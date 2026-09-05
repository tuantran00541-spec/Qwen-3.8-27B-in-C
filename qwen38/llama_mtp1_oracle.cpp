#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

static constexpr llama_token PROMPT_TOKEN = 12675; // raw "Hi" in pinned Qwen3.8 tokenizer
static constexpr int N_EMBD = 5120;

static int argmax(const float * x, int n) {
    return (int) std::distance(x, std::max_element(x, x + n));
}

static std::vector<std::pair<int, float>> topk(const float * x, int n, int k) {
    std::vector<int> ids(n);
    for (int i = 0; i < n; ++i) ids[i] = i;
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
        [&](int a, int b) { return x[a] > x[b]; });
    std::vector<std::pair<int, float>> out;
    for (int i = 0; i < k; ++i) out.push_back({ids[i], x[ids[i]]});
    return out;
}

static void write_topk(std::ofstream & out, const char * key,
                       const std::vector<std::pair<int, float>> & values,
                       bool comma = true) {
    out << "  \"" << key << "\": [";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << "{\"token\":" << values[i].first << ",\"logit\":" << values[i].second << "}";
    }
    out << "]" << (comma ? "," : "") << "\n";
}

static void mtp_batch_set_one(llama_batch & batch, llama_token token, llama_pos pos,
                              const float * h, bool logits) {
    batch.n_tokens = 1;
    batch.token[0] = token;
    std::memcpy(batch.embd, h, N_EMBD * sizeof(float));
    batch.pos[0] = pos;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = logits ? 1 : 0;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUTPUT.json\n", argv[0]);
        return 2;
    }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mtp = true;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 3;

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (llama_model_n_embd(model) != N_EMBD) {
        std::fprintf(stderr, "unexpected n_embd=%d\n", llama_model_n_embd(model));
        llama_model_free(model);
        return 4;
    }

    llama_context_params cp_tgt = llama_context_default_params();
    cp_tgt.n_ctx = 32;
    cp_tgt.n_batch = 4;
    cp_tgt.n_ubatch = 4;
    cp_tgt.n_threads = 2;
    cp_tgt.n_threads_batch = 2;
    cp_tgt.offload_kqv = false;
    cp_tgt.no_perf = true;
    llama_context * ctx_tgt = llama_init_from_model(model, cp_tgt);
    if (!ctx_tgt) {
        llama_model_free(model);
        return 5;
    }
    llama_set_embeddings_nextn(ctx_tgt, true, false);

    llama_context_params cp_mtp = llama_context_default_params();
    cp_mtp.n_ctx = 32;
    cp_mtp.n_batch = 4;
    cp_mtp.n_ubatch = 4;
    cp_mtp.n_threads = 2;
    cp_mtp.n_threads_batch = 2;
    cp_mtp.offload_kqv = false;
    cp_mtp.no_perf = true;
    cp_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cp_mtp.ctx_other = ctx_tgt;
    llama_context * ctx_mtp = llama_init_from_model(model, cp_mtp);
    if (!ctx_mtp) {
        llama_free(ctx_tgt);
        llama_model_free(model);
        return 6;
    }
    llama_set_embeddings_nextn(ctx_mtp, true, true);

    // Target position 0: raw token "Hi". Capture its post-output-norm h_nextn,
    // which is the h input paired with the target's sampled token at MTP pos 1.
    llama_token tok0 = PROMPT_TOKEN;
    llama_batch tgt0 = llama_batch_get_one(&tok0, 1);
    const int rc_t0 = llama_decode(ctx_tgt, tgt0);
    if (rc_t0 != 0) return 7;
    const float * logits0 = llama_get_logits(ctx_tgt);
    if (!logits0) return 8;
    llama_token token1 = (llama_token) argmax(logits0, n_vocab);
    const auto target0_top5 = topk(logits0, n_vocab, 5);

    const float * h0_ptr = llama_get_embeddings_nextn_ith(ctx_tgt, 0);
    if (!h0_ptr) return 9;
    std::vector<float> h0(h0_ptr, h0_ptr + N_EMBD);

    // MTP catch-up pos 0: token[0] + zero h, matching upstream's shifted-h process().
    llama_batch mtp = llama_batch_init(4, N_EMBD, 1);
    mtp.token = (llama_token *) std::malloc(sizeof(llama_token) * 4);
    if (!mtp.token) return 10;
    std::vector<float> zero_h(N_EMBD, 0.0f);
    mtp_batch_set_one(mtp, tok0, 0, zero_h.data(), false);
    const int rc_m0 = llama_decode(ctx_mtp, mtp);
    if (rc_m0 != 0) return 11;

    // MTP draft pos 1: target's freshest sampled token + target h_nextn[pos0].
    mtp_batch_set_one(mtp, token1, 1, h0.data(), true);
    const int rc_m1 = llama_decode(ctx_mtp, mtp);
    if (rc_m1 != 0) return 12;
    const float * mtp_logits = llama_get_logits(ctx_mtp);
    if (!mtp_logits) return 13;
    const llama_token mtp_token = (llama_token) argmax(mtp_logits, n_vocab);
    const auto mtp_top5 = topk(mtp_logits, n_vocab, 5);

    // Exact target oracle for the same proposed next position.
    llama_batch tgt1 = llama_batch_get_one(&token1, 1);
    const int rc_t1 = llama_decode(ctx_tgt, tgt1);
    if (rc_t1 != 0) return 14;
    const float * logits1 = llama_get_logits(ctx_tgt);
    if (!logits1) return 15;
    const llama_token token2 = (llama_token) argmax(logits1, n_vocab);
    const auto target1_top5 = topk(logits1, n_vocab, 5);

    std::ofstream out(argv[2]);
    out << std::setprecision(9);
    out << "{\n";
    out << "  \"schema\": \"qwen38-llama-mtp1-oracle-v1\",\n";
    out << "  \"llama_cpp_revision\": \"b81c99b479d4c24e5eeca10de99032ebd343ef8f\",\n";
    out << "  \"prompt_token\": " << tok0 << ",\n";
    out << "  \"target_first_token\": " << token1 << ",\n";
    out << "  \"mtp_draft_token\": " << mtp_token << ",\n";
    out << "  \"target_verify_token\": " << token2 << ",\n";
    out << "  \"mtp1_accepted\": " << (mtp_token == token2 ? "true" : "false") << ",\n";
    out << "  \"decode_rc_target0\": " << rc_t0 << ",\n";
    out << "  \"decode_rc_mtp_catchup\": " << rc_m0 << ",\n";
    out << "  \"decode_rc_mtp_draft\": " << rc_m1 << ",\n";
    out << "  \"decode_rc_target1\": " << rc_t1 << ",\n";
    write_topk(out, "target_first_top5", target0_top5);
    write_topk(out, "mtp_draft_top5", mtp_top5);
    write_topk(out, "target_verify_top5", target1_top5, false);
    out << "}\n";

    std::fprintf(stderr,
        "QWEN38_LLAMA_MTP1_ORACLE target1=%d mtp=%d target2=%d accepted=%d\n",
        (int) token1, (int) mtp_token, (int) token2, mtp_token == token2 ? 1 : 0);

    std::free(mtp.token);
    mtp.token = nullptr;
    llama_batch_free(mtp);
    llama_free(ctx_mtp);
    llama_free(ctx_tgt);
    llama_model_free(model);
    return 0;
}
