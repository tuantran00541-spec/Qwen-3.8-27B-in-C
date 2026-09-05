#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>
#include <vector>

struct capture_state {
    std::map<std::string, std::vector<float>> tensors;
    std::string error;
    bool done = false;
    std::string layer3_input_source;
    std::string post_ffn_source;
};

static const char * TARGETS[] = {
    // Layer-3 input provenance. l_out-2 is preferred; the residual + FFN pair
    // lets us reconstruct it if graph optimization fuses/renames the final add.
    "l_out-2",
    "post_ffn-2",
    "attn_residual-2",
    "ffn_out-2",

    "attn_norm-3",
    "Qcur_full-3",
    "Qcur_reshaped-3",
    "Qcur_normed-3",
    "Kcur-3",
    "Kcur_normed-3",
    "gate_reshaped-3",
    "Vcur-3",
    "attn_pregate-3",
    "gate_sigmoid-3",
    "attn_gated-3",
    "attn_output-3",
    "attn_residual-3",
    "attn_post_norm-3",
    "ffn_out-3",
    "post_ffn-3",
};

static bool wanted(const char * name) {
    if (!name) return false;
    for (const char * target : TARGETS) {
        if (std::strcmp(name, target) == 0) return true;
    }
    return false;
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * state = static_cast<capture_state *>(user_data);
    const char * name = ggml_get_name(t);
    const bool match = wanted(name);
    if (ask) return match;
    if (!match) return true;

    if (t->type != GGML_TYPE_F32) {
        state->error = std::string("checkpoint ") + (name ? name : "<unnamed>") +
            " has unexpected type " + ggml_type_name(t->type);
        return false;
    }
    const size_t n = ggml_nelements(t);
    std::vector<float> values(n);
    ggml_backend_tensor_get(t, values.data(), 0, n * sizeof(float));
    state->tensors[name] = std::move(values);
    if (name && std::strcmp(name, "post_ffn-3") == 0) {
        state->done = true;
        state->post_ffn_source = "cb_eval";
    }
    return true;
}

static bool derive_add(
        capture_state & state,
        const char * left_name,
        const char * right_name,
        const char * out_name) {
    const auto left = state.tensors.find(left_name);
    const auto right = state.tensors.find(right_name);
    if (left == state.tensors.end() || right == state.tensors.end()) return false;
    if (left->second.size() != right->second.size()) {
        state.error = std::string(left_name) + " / " + right_name + " size mismatch";
        return false;
    }
    std::vector<float> out(left->second.size());
    for (size_t i = 0; i < out.size(); ++i) {
        out[i] = left->second[i] + right->second[i];
    }
    state.tensors[out_name] = std::move(out);
    return true;
}

static void finalize(capture_state & state) {
    if (!state.error.empty()) return;

    // No control vector is installed in this oracle context, so l_out-2 and
    // post_ffn-2 are semantically the same decoder output entering layer 3.
    auto lout = state.tensors.find("l_out-2");
    if (lout != state.tensors.end()) {
        state.tensors["layer3_input"] = lout->second;
        state.layer3_input_source = "cb_eval(l_out-2)";
    } else {
        auto post = state.tensors.find("post_ffn-2");
        if (post != state.tensors.end()) {
            state.tensors["layer3_input"] = post->second;
            state.layer3_input_source = "cb_eval(post_ffn-2; cvec identity)";
        } else if (derive_add(state, "attn_residual-2", "ffn_out-2", "layer3_input")) {
            state.layer3_input_source =
                "derived_fp32_add(attn_residual-2,ffn_out-2; cvec identity)";
        } else if (state.error.empty()) {
            state.error = "could not capture/derive layer3 input";
            return;
        }
    }

    if (!state.done) {
        if (derive_add(state, "attn_residual-3", "ffn_out-3", "post_ffn-3")) {
            state.done = true;
            state.post_ffn_source = "derived_fp32_add(attn_residual-3,ffn_out-3)";
        } else if (state.error.empty()) {
            state.error = "could not capture/derive post_ffn-3";
        }
    }
}

static void write_json(const char * path, llama_token token, int decode_rc, const capture_state & state) {
    std::ofstream out(path);
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"schema\": \"qwen38-llama-layer3-oracle-v1\",\n";
    out << "  \"llama_cpp_revision\": \"557614e0296ff4a5b6f649737a65ae2076eea2fd\",\n";
    out << "  \"token_id\": " << token << ",\n";
    out << "  \"decode_returncode\": " << decode_rc << ",\n";
    out << "  \"captured_complete_layer\": " << (state.done ? "true" : "false") << ",\n";
    out << "  \"layer3_input_source\": \"" << state.layer3_input_source << "\",\n";
    out << "  \"post_ffn_source\": \"" << state.post_ffn_source << "\",\n";
    out << "  \"error\": \"" << state.error << "\",\n";
    out << "  \"checkpoints\": {\n";
    size_t ti = 0;
    for (const auto & kv : state.tensors) {
        out << "    \"" << kv.first << "\": [";
        for (size_t i = 0; i < kv.second.size(); ++i) {
            if (i) out << ',';
            out << kv.second[i];
        }
        out << "]" << (++ti == state.tensors.size() ? "\n" : ",\n");
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
        std::fprintf(stderr, "complete layer-3 checkpoint was not captured/derived\n");
        return 7;
    }

    std::fprintf(stderr,
        "QWEN38_LLAMA_LAYER3_ORACLE token=%d checkpoints=%zu input_source=%s post_ffn_source=%s\n",
        (int) token,
        state.tensors.size(),
        state.layer3_input_source.c_str(),
        state.post_ffn_source.c_str());
    return 0;
}
