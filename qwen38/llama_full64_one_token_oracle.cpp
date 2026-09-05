#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

static constexpr int N_LAYER = 64;

struct capture_state {
    std::map<std::string, std::vector<float>> tensors;
    std::string error;
    bool done = false;
    int top_token = -1;
    float top_logit = 0.0f;
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
    if (std::strcmp(name, "result_norm") == 0 || std::strcmp(name, "result_output") == 0) {
        return true;
    }
    int layer = -1;
    return parse_layer_name(name, "post_ffn-", layer)
        || parse_layer_name(name, "attn_residual-", layer)
        || parse_layer_name(name, "ffn_out-", layer);
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<capture_state *>(user_data);
    const char * name = ggml_get_name(t);
    const bool match = wanted(name);
    if (ask) return match;
    if (!match) return true;

    if (t->type != GGML_TYPE_F32) {
        state->error = std::string("checkpoint ") + (name ? name : "<unnamed>")
            + " has unexpected type " + ggml_type_name(t->type);
        return false;
    }
    const size_t n = ggml_nelements(t);
    std::vector<float> values(n);
    ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
    state->tensors[name] = std::move(values);
    return true;
}

static bool derive_add(capture_state & state, int layer) {
    const std::string left_name = "attn_residual-" + std::to_string(layer);
    const std::string right_name = "ffn_out-" + std::to_string(layer);
    const std::string out_name = "post_ffn-" + std::to_string(layer);
    const auto left = state.tensors.find(left_name);
    const auto right = state.tensors.find(right_name);
    if (left == state.tensors.end() || right == state.tensors.end()) {
        return state.tensors.find(out_name) != state.tensors.end();
    }
    if (left->second.size() != right->second.size()) {
        state.error = left_name + " / " + right_name + " size mismatch";
        return false;
    }
    std::vector<float> out(left->second.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = left->second[i] + right->second[i];
    }
    // Prefer the explicit semantic add over an optimizer-aliased callback.
    state.tensors[out_name] = std::move(out);
    return true;
}

static void finalize(capture_state & state) {
    if (!state.error.empty()) return;
    for (int il = 0; il < N_LAYER; ++il) {
        if (!derive_add(state, il)) {
            state.error = "could not capture/derive post_ffn-" + std::to_string(il);
            return;
        }
        const auto & out = state.tensors["post_ffn-" + std::to_string(il)];
        if (out.size() != 5120) {
            state.error = "post_ffn-" + std::to_string(il) + " width mismatch";
            return;
        }
    }

    const auto norm = state.tensors.find("result_norm");
    const auto logits = state.tensors.find("result_output");
    if (norm == state.tensors.end() || norm->second.size() != 5120) {
        state.error = "result_norm missing or width mismatch";
        return;
    }
    if (logits == state.tensors.end() || logits->second.size() != 248320) {
        state.error = "result_output missing or width mismatch";
        return;
    }
    const auto it = std::max_element(logits->second.begin(), logits->second.end());
    state.top_token = (int) std::distance(logits->second.begin(), it);
    state.top_logit = *it;
    state.done = true;
}

static void write_json(const char * path, llama_token token, int decode_rc, const capture_state & state) {
    std::ofstream out(path);
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"schema\": \"qwen38-llama-full64-one-token-oracle-v1\",\n";
    out << "  \"llama_cpp_revision\": \"557614e0296ff4a5b6f649737a65ae2076eea2fd\",\n";
    out << "  \"token_id\": " << token << ",\n";
    out << "  \"position\": 0,\n";
    out << "  \"decode_returncode\": " << decode_rc << ",\n";
    out << "  \"captured_complete_model\": " << (state.done ? "true" : "false") << ",\n";
    out << "  \"top_token\": " << state.top_token << ",\n";
    out << "  \"top_logit\": " << state.top_logit << ",\n";
    out << "  \"post_ffn_semantics\": \"prefer_derived_fp32_add(attn_residual,ffn_out)\",\n";
    out << "  \"error\": \"" << state.error << "\",\n";
    out << "  \"checkpoints\": {\n";
    size_t ti = 0;
    for (const auto & kv : state.tensors) {
        // The residual/FFN component tensors are only capture provenance; keep
        // the artifact compact by emitting post_ffn-N plus the two final heads.
        if (kv.first.rfind("attn_residual-", 0) == 0 || kv.first.rfind("ffn_out-", 0) == 0) {
            continue;
        }
        ++ti;
    }
    size_t emitted = 0;
    for (const auto & kv : state.tensors) {
        if (kv.first.rfind("attn_residual-", 0) == 0 || kv.first.rfind("ffn_out-", 0) == 0) {
            continue;
        }
        out << "    \"" << kv.first << "\": [";
        for (size_t i = 0; i < kv.second.size(); ++i) {
            if (i) out << ',';
            out << kv.second[i];
        }
        out << "]" << (++emitted == ti ? "\n" : ",\n");
    }
    out << "  }\n";
    out << "}\n";
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUTPUT.json\n", argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 3;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_token token = llama_vocab_bos(vocab);
    if (token == LLAMA_TOKEN_NULL) {
        std::fprintf(stderr, "model has no BOS token\n");
        llama_model_free(model);
        return 4;
    }

    capture_state state;
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 32;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_threads = 2;
    cp.n_threads_batch = 2;
    cp.offload_kqv = false;
    cp.no_perf = true;
    cp.cb_eval = capture_cb;
    cp.cb_eval_user_data = &state;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 5;
    }

    llama_batch batch = llama_batch_get_one(&token, 1);
    const int rc = llama_decode(ctx, batch);
    finalize(state);
    write_json(argv[2], token, rc, state);

    llama_free(ctx);
    llama_model_free(model);

    if (!state.error.empty()) {
        std::fprintf(stderr, "%s\n", state.error.c_str());
        return 6;
    }
    if (rc != 0) {
        std::fprintf(stderr, "llama_decode failed rc=%d\n", rc);
        return 8;
    }
    if (!state.done) {
        std::fprintf(stderr, "complete full-model oracle was not captured\n");
        return 7;
    }

    std::fprintf(stderr,
        "QWEN38_LLAMA_FULL64_ORACLE token=%d checkpoints=%zu top_token=%d top_logit=%.9g\n",
        (int) token, state.tensors.size(), state.top_token, state.top_logit);
    return 0;
}
