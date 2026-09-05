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
};

static const char * TARGETS[] = {
    "model.input_embed",
    "attn_residual-0", "ffn_out-0", "post_ffn-0",
    "attn_norm-1", "linear_attn_qkv_mixed-1", "z-1",
    "beta-1", "beta_sigmoid-1", "alpha-1", "a_softplus-1", "gate-1",
    "conv_output_silu-1", "q_conv_predelta-1", "k_conv_predelta-1", "v_conv_predelta-1",
    "linear_attn_out-1", "attn_residual-1", "attn_post_norm-1",
    "ffn_out-1", "post_ffn-1",
};

static bool wanted(const char * name) {
    if (!name) return false;
    for (const char * target : TARGETS) if (std::strcmp(name, target) == 0) return true;
    return false;
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * ud) {
    auto * s = static_cast<capture_state *>(ud);
    const char * name = ggml_get_name(t);
    const bool match = wanted(name);
    if (ask) return match;
    if (!match) return true;
    if (t->type != GGML_TYPE_F32) {
        s->error = std::string("checkpoint ") + (name ? name : "<unnamed>") + " type=" + ggml_type_name(t->type);
        return false;
    }
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size()*sizeof(float));
    s->tensors[name] = std::move(v);
    return true;
}

static bool derive_add(capture_state & s, int il) {
    const std::string l = "attn_residual-" + std::to_string(il);
    const std::string r = "ffn_out-" + std::to_string(il);
    const std::string o = "post_ffn-" + std::to_string(il);
    auto li=s.tensors.find(l), ri=s.tensors.find(r);
    if (li == s.tensors.end() || ri == s.tensors.end()) return s.tensors.find(o) != s.tensors.end();
    if (li->second.size() != ri->second.size()) { s.error = o + " add size mismatch"; return false; }
    std::vector<float> out(li->second.size());
    for (size_t i=0;i<out.size();++i) out[i] = li->second[i] + ri->second[i];
    s.tensors[o] = std::move(out);
    return true;
}

static bool normalize_qk(capture_state & s) {
    constexpr size_t fused = 128*16, tiled = 128*48;
    for (const char * name : {"q_conv_predelta-1", "k_conv_predelta-1"}) {
        auto it=s.tensors.find(name); if (it==s.tensors.end()) return false;
        if (it->second.size()==tiled) continue;
        if (it->second.size()!=fused) { s.error=std::string(name)+" width mismatch"; return false; }
        const auto src=it->second; std::vector<float> out; out.reserve(tiled);
        for (int r=0;r<3;++r) out.insert(out.end(), src.begin(), src.end());
        it->second=std::move(out);
    }
    return true;
}

static void finalize(capture_state & s) {
    if (!s.error.empty()) return;
    if (!derive_add(s,0) || !derive_add(s,1)) { s.error="could not derive layer boundary"; return; }
    if (!normalize_qk(s)) { if (s.error.empty()) s.error="q/k predelta missing"; return; }
    if (s.tensors["post_ffn-0"].size()!=5120 || s.tensors["post_ffn-1"].size()!=5120) {
        s.error="layer boundary width mismatch"; return;
    }
    s.done=true;
}

static void write_json(const char * path, llama_token token, int rc, const capture_state & s) {
    std::ofstream out(path); out << std::setprecision(9);
    out << "{\n  \"schema\": \"qwen38-llama-layer1-composition-oracle-v1\",\n";
    out << "  \"llama_cpp_revision\": \"557614e0296ff4a5b6f649737a65ae2076eea2fd\",\n";
    out << "  \"token_id\": " << token << ",\n  \"decode_returncode\": " << rc << ",\n";
    out << "  \"captured_complete\": " << (s.done?"true":"false") << ",\n  \"error\": \"" << s.error << "\",\n";
    out << "  \"checkpoints\": {\n"; size_t n=0;
    for (const auto & kv:s.tensors) {
        out << "    \"" << kv.first << "\": [";
        for (size_t i=0;i<kv.second.size();++i) { if(i) out << ','; out << kv.second[i]; }
        out << "]" << (++n==s.tensors.size()?"\n":",\n");
    }
    out << "  }\n}\n";
}

int main(int argc,char **argv) {
    if(argc!=3){std::fprintf(stderr,"usage: %s MODEL.gguf OUTPUT.json\n",argv[0]);return 2;}
    ggml_backend_load_all(); llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
    llama_model * model=llama_model_load_from_file(argv[1],mp); if(!model)return 3;
    const llama_vocab * vocab=llama_model_get_vocab(model); llama_token token=llama_vocab_bos(vocab);
    if(token==LLAMA_TOKEN_NULL){llama_model_free(model);return 4;}
    capture_state s; llama_context_params cp=llama_context_default_params();
    cp.n_ctx=32; cp.n_batch=1; cp.n_ubatch=1; cp.n_threads=2; cp.n_threads_batch=2;
    cp.offload_kqv=false; cp.no_perf=true; cp.cb_eval=capture_cb; cp.cb_eval_user_data=&s;
    llama_context * ctx=llama_init_from_model(model,cp); if(!ctx){llama_model_free(model);return 5;}
    llama_batch batch=llama_batch_get_one(&token,1); const int rc=llama_decode(ctx,batch);
    finalize(s); write_json(argv[2],token,rc,s); llama_free(ctx); llama_model_free(model);
    if(!s.error.empty()){std::fprintf(stderr,"%s\n",s.error.c_str());return 6;}
    if(rc!=0)return 8; if(!s.done)return 7;
    std::fprintf(stderr,"QWEN38_LAYER1_COMPOSITION_ORACLE token=%d checkpoints=%zu\n",(int)token,s.tensors.size());
    return 0;
}
