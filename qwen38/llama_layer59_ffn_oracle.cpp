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
    std::string error, input_source, output_source;
    bool done = false;
};

static const char * TARGETS[] = {
    "l_out-58", "post_ffn-58", "attn_residual-58", "ffn_out-58",
    "attn_norm-59", "Qcur_full-59", "gate_reshaped-59", "Vcur-59",
    "attn_pregate-59", "gate_sigmoid-59", "attn_gated-59", "attn_output-59",
    "attn_residual-59", "attn_post_norm-59", "ffn_gate-59", "ffn_up-59",
    "ffn_swiglu-59", "ffn_out-59", "post_ffn-59",
};

static bool wanted(const char * name) {
    if (!name) return false;
    for (const char * t : TARGETS) if (std::strcmp(name, t) == 0) return true;
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

static bool derive_add(capture_state & s, const char * a, const char * b, const char * out) {
    auto ai=s.tensors.find(a), bi=s.tensors.find(b);
    if (ai==s.tensors.end() || bi==s.tensors.end()) return false;
    if (ai->second.size()!=bi->second.size()) { s.error=std::string(out)+" add size mismatch"; return false; }
    std::vector<float> v(ai->second.size());
    for (size_t i=0;i<v.size();++i) v[i]=ai->second[i]+bi->second[i];
    s.tensors[out]=std::move(v); return true;
}

static void finalize(capture_state & s) {
    if (!s.error.empty()) return;
    if (derive_add(s,"attn_residual-58","ffn_out-58","layer59_input")) s.input_source="derived_fp32_add(attn_residual-58,ffn_out-58)";
    else if (s.tensors.count("l_out-58")) { s.tensors["layer59_input"]=s.tensors["l_out-58"]; s.input_source="cb_eval(l_out-58)"; }
    else if (s.tensors.count("post_ffn-58")) { s.tensors["layer59_input"]=s.tensors["post_ffn-58"]; s.input_source="cb_eval(post_ffn-58)"; }
    else { s.error="could not derive layer59 input"; return; }

    if (derive_add(s,"attn_residual-59","ffn_out-59","post_ffn-59")) s.output_source="derived_fp32_add(attn_residual-59,ffn_out-59)";
    else if (s.tensors.count("post_ffn-59")) s.output_source="cb_eval(post_ffn-59)";
    else { s.error="could not derive post_ffn-59"; return; }

    const char * req[] = {"layer59_input","attn_norm-59","gate_reshaped-59","Vcur-59","attn_pregate-59",
        "gate_sigmoid-59","attn_gated-59","attn_output-59","attn_residual-59","attn_post_norm-59",
        "ffn_gate-59","ffn_up-59","ffn_swiglu-59","ffn_out-59","post_ffn-59"};
    for (const char * n:req) if (!s.tensors.count(n)) { s.error=std::string("missing checkpoint ")+n; return; }
    if (s.tensors["layer59_input"].size()!=5120 || s.tensors["post_ffn-59"].size()!=5120) { s.error="layer59 boundary width mismatch"; return; }
    s.done=true;
}

static void write_json(const char * path, llama_token token, int rc, const capture_state & s) {
    std::ofstream out(path); out<<std::setprecision(9);
    out<<"{\n  \"schema\": \"qwen38-llama-layer59-ffn-oracle-v1\",\n";
    out<<"  \"llama_cpp_revision\": \"557614e0296ff4a5b6f649737a65ae2076eea2fd\",\n";
    out<<"  \"token_id\": "<<token<<",\n  \"decode_returncode\": "<<rc<<",\n  \"captured_complete\": "<<(s.done?"true":"false")<<",\n";
    out<<"  \"input_source\": \""<<s.input_source<<"\",\n  \"output_source\": \""<<s.output_source<<"\",\n  \"error\": \""<<s.error<<"\",\n  \"checkpoints\": {\n";
    size_t n=0; for (const auto & kv:s.tensors) { out<<"    \""<<kv.first<<"\": ["; for(size_t i=0;i<kv.second.size();++i){if(i)out<<',';out<<kv.second[i];} out<<"]"<<(++n==s.tensors.size()?"\n":",\n"); }
    out<<"  }\n}\n";
}

int main(int argc,char **argv) {
    if(argc!=3){std::fprintf(stderr,"usage: %s MODEL.gguf OUTPUT.json\n",argv[0]);return 2;}
    ggml_backend_load_all(); llama_model_params mp=llama_model_default_params(); mp.n_gpu_layers=0;
    llama_model * model=llama_model_load_from_file(argv[1],mp); if(!model)return 3;
    const llama_vocab * vocab=llama_model_get_vocab(model); llama_token token=llama_vocab_bos(vocab); if(token==LLAMA_TOKEN_NULL){llama_model_free(model);return 4;}
    capture_state s; llama_context_params cp=llama_context_default_params(); cp.n_ctx=32; cp.n_batch=1; cp.n_ubatch=1; cp.n_threads=2; cp.n_threads_batch=2; cp.offload_kqv=false; cp.no_perf=true; cp.cb_eval=capture_cb; cp.cb_eval_user_data=&s;
    llama_context * ctx=llama_init_from_model(model,cp); if(!ctx){llama_model_free(model);return 5;}
    llama_batch batch=llama_batch_get_one(&token,1); int rc=llama_decode(ctx,batch); finalize(s); write_json(argv[2],token,rc,s); llama_free(ctx); llama_model_free(model);
    if(!s.error.empty()){std::fprintf(stderr,"%s\n",s.error.c_str());return 6;} if(rc!=0)return 8; if(!s.done)return 7;
    std::fprintf(stderr,"QWEN38_LAYER59_FFN_ORACLE token=%d checkpoints=%zu\n",(int)token,s.tensors.size()); return 0;
}
