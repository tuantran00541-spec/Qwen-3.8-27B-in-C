#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

static constexpr int N_LAYER = 64;

struct capture_state {
    bool enabled = false;
    std::map<std::string, std::vector<float>> tensors;
    std::string error;
};

static bool parse_layer_name(const char * name, const char * prefix, int & layer) {
    if (!name || !prefix) return false;
    const size_t n = std::strlen(prefix);
    if (std::strncmp(name, prefix, n) != 0) return false;
    char * end = nullptr;
    const long value = std::strtol(name + n, &end, 10);
    if (!end || *end != '\0' || value < 0 || value >= N_LAYER) return false;
    layer = (int) value;
    return true;
}

static bool wanted(const char * name) {
    if (!name) return false;
    if (std::strcmp(name, "result_norm") == 0 || std::strcmp(name, "result_output") == 0) return true;
    int layer = -1;
    if (parse_layer_name(name, "post_ffn-", layer) ||
        parse_layer_name(name, "attn_residual-", layer) ||
        parse_layer_name(name, "ffn_out-", layer)) return true;
    if (parse_layer_name(name, "Qcur-", layer) ||
        parse_layer_name(name, "Kcur-", layer) ||
        parse_layer_name(name, "Vcur-", layer)) return layer % 4 == 3;
    return false;
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * s = static_cast<capture_state *>(user_data);
    if (!s->enabled) return false;
    const char * name = ggml_get_name(t);
    const bool match = wanted(name);
    if (ask) return match;
    if (!match) return true;
    if (t->type != GGML_TYPE_F32) {
        s->error = std::string("checkpoint ") + (name ? name : "<unnamed>") +
            " type=" + ggml_type_name(t->type);
        return false;
    }
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    s->tensors[name] = std::move(v);
    return true;
}

static bool derive_add(capture_state & s, int layer) {
    const std::string a = "attn_residual-" + std::to_string(layer);
    const std::string b = "ffn_out-" + std::to_string(layer);
    const std::string o = "post_ffn-" + std::to_string(layer);
    auto ai = s.tensors.find(a), bi = s.tensors.find(b);
    if (ai == s.tensors.end() || bi == s.tensors.end()) return s.tensors.count(o) != 0;
    if (ai->second.size() != bi->second.size()) return false;
    std::vector<float> v(ai->second.size());
    for (size_t i = 0; i < v.size(); ++i) v[i] = ai->second[i] + bi->second[i];
    s.tensors[o] = std::move(v);
    return true;
}

static int argmax(const float * x, int n) {
    return (int) std::distance(x, std::max_element(x, x + n));
}

static void finalize(capture_state & s) {
    if (!s.error.empty()) return;
    for (int il = 0; il < N_LAYER; ++il) {
        if (!derive_add(s, il) || s.tensors["post_ffn-" + std::to_string(il)].size() != 5120) {
            s.error = "missing/invalid post_ffn-" + std::to_string(il);
            return;
        }
    }
    for (int il = 3; il < N_LAYER; il += 4) {
        for (const char * prefix : {"Qcur-", "Kcur-", "Vcur-"}) {
            const std::string name = std::string(prefix) + std::to_string(il);
            if (!s.tensors.count(name)) {
                s.error = "missing " + name;
                return;
            }
        }
    }
    if (!s.tensors.count("result_norm") || s.tensors["result_norm"].size() != 5120 ||
        !s.tensors.count("result_output")) {
        s.error = "missing final outputs";
    }
}

static void write_json(const char * path, llama_token bos, llama_token token1,
                       llama_token token2, int rc0, int rc1, const capture_state & s) {
    std::ofstream out(path);
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"schema\": \"qwen38-llama-full64-two-token-oracle-v1\",\n";
    out << "  \"llama_cpp_revision\": \"557614e0296ff4a5b6f649737a65ae2076eea2fd\",\n";
    out << "  \"bos_token\": " << bos << ",\n";
    out << "  \"token1\": " << token1 << ",\n";
    out << "  \"token2\": " << token2 << ",\n";
    out << "  \"position\": 1,\n";
    out << "  \"decode_rc0\": " << rc0 << ",\n";
    out << "  \"decode_rc1\": " << rc1 << ",\n";
    out << "  \"captured_complete\": " << (s.error.empty() ? "true" : "false") << ",\n";
    out << "  \"error\": \"" << s.error << "\",\n";
    out << "  \"checkpoints\": {\n";
    std::vector<std::pair<std::string, const std::vector<float> *>> emit;
    for (const auto & kv : s.tensors) {
        if (kv.first.rfind("attn_residual-", 0) == 0 || kv.first.rfind("ffn_out-", 0) == 0) continue;
        emit.push_back({kv.first, &kv.second});
    }
    for (size_t j = 0; j < emit.size(); ++j) {
        out << "    \"" << emit[j].first << "\": [";
        const auto & v = *emit[j].second;
        for (size_t i = 0; i < v.size(); ++i) { if (i) out << ','; out << v[i]; }
        out << "]" << (j + 1 == emit.size() ? "\n" : ",\n");
    }
    out << "  }\n}\n";
}

int main(int argc, char ** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: %s MODEL.gguf OUTPUT.json\n", argv[0]); return 2; }
    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp); if (!model) return 3;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token bos = llama_vocab_bos(vocab); if (bos == LLAMA_TOKEN_NULL) return 4;

    capture_state s;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 32; cp.n_batch = 1; cp.n_ubatch = 1; cp.n_threads = 2; cp.n_threads_batch = 2;
    cp.offload_kqv = false; cp.no_perf = true; cp.cb_eval = capture_cb; cp.cb_eval_user_data = &s;
    llama_context * ctx = llama_init_from_model(model, cp); if (!ctx) return 5;

    s.enabled = false;
    llama_batch b0 = llama_batch_get_one(&bos, 1);
    const int rc0 = llama_decode(ctx, b0);
    const int vocab_n = llama_vocab_n_tokens(vocab);
    const float * logits0 = llama_get_logits(ctx);
    llama_token token1 = (llama_token) argmax(logits0, vocab_n);

    s.enabled = true;
    llama_batch b1 = llama_batch_get_one(&token1, 1);
    const int rc1 = llama_decode(ctx, b1);
    finalize(s);
    llama_token token2 = LLAMA_TOKEN_NULL;
    if (rc1 == 0) token2 = (llama_token) argmax(llama_get_logits(ctx), vocab_n);
    write_json(argv[2], bos, token1, token2, rc0, rc1, s);

    llama_free(ctx); llama_model_free(model);
    if (rc0 != 0 || rc1 != 0) return 8;
    if (!s.error.empty()) { std::fprintf(stderr, "%s\n", s.error.c_str()); return 6; }
    std::fprintf(stderr, "QWEN38_TWO_TOKEN_ORACLE bos=%d token1=%d token2=%d checkpoints=%zu\n",
        (int) bos, (int) token1, (int) token2, s.tensors.size());
    return 0;
}