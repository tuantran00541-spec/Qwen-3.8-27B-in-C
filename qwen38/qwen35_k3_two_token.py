#!/usr/bin/env python3
"""Stateful two-token Qwen3.8-27B K3 executor.

Milestone path:
  token0 = BOS builds real recurrent/conv/KV state;
  token1 = pinned llama.cpp token0 argmax is consumed at position 1;
  the custom runtime emits token2 logits.

Recurrent Gated-DeltaNet state is stored as real F32 matrices in a native C
kernel, matching pinned llama.cpp autoregressive equations. Full-attention K/V
uses the default F16 cache. Text-only position 1 MRoPE reduces to ordinary
partial NeoX RoPE because all multimodal position axes are equal for text.
"""
from __future__ import annotations

import argparse, ctypes, json, math, resource, time
from pathlib import Path
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_full_attn_layer3_gate as attn
import qwen35_k3_full64_one_token as base
import qwen35_k3_full64_ggml_exact as exact
import qwen35_k3_full64_ggml_rmsnorm as rmswrap
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

N_LAYER = 64
ROPE_N_ROT = 64                 # head_dim 256 * partial_rotary_factor 0.25
ROPE_THETA = 10_000_000.0       # official pinned Qwen3.8 config
STATE_ELEMS = gdn.V_HEADS * gdn.HEAD_DIM * gdn.HEAD_DIM
STATE_BYTES_PER_LAYER = STATE_ELEMS * 4
SCALE_GDN = 1.0 / math.sqrt(gdn.HEAD_DIM)
SCALE_ATTN = 1.0 / math.sqrt(attn.HEAD_DIM)


def rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def f32(x: float) -> float: return exact.f32(x)
def addf(a: float, b: float) -> float: return f32(f32(a) + f32(b))
def mulf(a: float, b: float) -> float: return exact.mul_f32(a, b)
def siluf(x: float) -> float: return mulf(x, exact.sigmoid_f32(x))

def softplusf(x: float) -> float:
    xf = f32(x)
    if xf > 20.0: return xf
    if xf < -20.0: return f32(exact.expf(xf))
    return f32(math.log1p(float(exact.expf(xf))))


def load_state_lib(path: Path):
    lib = ctypes.CDLL(str(path))
    fp = ctypes.POINTER(ctypes.c_float)
    lib.qwen_gdn_ar_step_f32.argtypes = [fp, fp, fp, fp, fp, fp, fp]
    lib.qwen_gdn_ar_step_f32.restype = ctypes.c_int
    return lib


def carr(values: Sequence[float]):
    return (ctypes.c_float * len(values))(*map(float, values))


def repeat_k_heads(values: Sequence[float]) -> list[float]:
    heads = gdn.split_heads(values, gdn.K_HEADS)
    reps = gdn.V_HEADS // gdn.K_HEADS
    return gdn.flatten([h for _ in range(reps) for h in heads])


def rope_text_neox(values: Sequence[float], heads: int, pos: int) -> list[float]:
    if pos == 0: return list(map(float, values))
    hs = attn.split_heads(values, heads)
    half = ROPE_N_ROT // 2
    out = []
    for head in hs:
        h = list(map(float, head))
        src = h[:]
        for i in range(half):
            freq = ROPE_THETA ** (-(2.0 * i) / ROPE_N_ROT)
            angle = float(pos) * freq
            c = f32(math.cos(angle)); s = f32(math.sin(angle))
            x0 = f32(src[i]); x1 = f32(src[i + half])
            h[i] = addf(mulf(x0, c), -mulf(x1, s))
            h[i + half] = addf(mulf(x0, s), mulf(x1, c))
        out.extend(h)
    return out


def ffn(runtime, view, metas, prefix: str, x: Sequence[float]) -> list[float]:
    prepared = runtime.quantize(x, "Q6_K")
    gate = runtime.matvec(view("ffn_gate.weight"), metas[f"{prefix}.ffn_gate.weight"], x, prepared)
    up = runtime.matvec(view("ffn_up.weight"), metas[f"{prefix}.ffn_up.weight"], x, prepared)
    sw = [mulf(siluf(gate[i]), up[i]) for i in range(gdn.INTERMEDIATE)]
    return runtime.matvec(view("ffn_down.weight"), metas[f"{prefix}.ffn_down.weight"], sw)


def recurrent_step(runtime, state_lib, state, prev_qkv, view, metas, vec,
                   hidden: Sequence[float], layer: int, token_index: int):
    p = f"blk.{layer}"
    x = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    qkv = runtime.matvec(view("attn_qkv.weight"), metas[f"{p}.attn_qkv.weight"], x)
    z = runtime.matvec(view("attn_gate.weight"), metas[f"{p}.attn_gate.weight"], x)
    beta_raw = runtime.matvec(view("ssm_beta.weight"), metas[f"{p}.ssm_beta.weight"], x)
    alpha = runtime.matvec(view("ssm_alpha.weight"), metas[f"{p}.ssm_alpha.weight"], x)
    beta = [exact.sigmoid_f32(v) for v in beta_raw]
    dt = vec("ssm_dt.bias"); aa = vec("ssm_a")
    gate = [mulf(aa[h], softplusf(addf(alpha[h], dt[h]))) for h in range(gdn.V_HEADS)]

    kernels = vec("ssm_conv1d.weight")
    conv = [0.0] * gdn.CONV_DIM
    for c in range(gdn.CONV_DIM):
        cur = mulf(qkv[c], kernels[c*gdn.CONV_KERNEL + 3])
        if token_index > 0:
            cur = addf(mulf(prev_qkv[c], kernels[c*gdn.CONV_KERNEL + 2]), cur)
        conv[c] = siluf(cur)

    q = conv[:gdn.KEY_DIM]; k = conv[gdn.KEY_DIM:2*gdn.KEY_DIM]; v = conv[2*gdn.KEY_DIM:]
    qn = gdn.flatten([gdn.l2_norm(h) for h in gdn.split_heads(q, gdn.K_HEADS)])
    kn = gdn.flatten([gdn.l2_norm(h) for h in gdn.split_heads(k, gdn.K_HEADS)])
    q48 = [mulf(vv, SCALE_GDN) for vv in repeat_k_heads(qn)]
    k48 = repeat_k_heads(kn)

    out_buf = (ctypes.c_float * gdn.VALUE_DIM)()
    rc = state_lib.qwen_gdn_ar_step_f32(state, carr(q48), carr(k48), carr(v), carr(gate), carr(beta), out_buf)
    if rc != 0: raise RuntimeError(f"layer {layer}: GDN state kernel rc={rc}")
    core = [float(out_buf[i]) for i in range(gdn.VALUE_DIM)]

    norm_w = vec("ssm_norm.weight")
    core_h = gdn.split_heads(core, gdn.V_HEADS); z_h = gdn.split_heads(z, gdn.V_HEADS)
    gated = []
    for ch, zh in zip(core_h, z_h):
        nh = rmswrap.ggml_rms_norm(ch, norm_w, gdn.RMS_EPS)
        gated.extend(mulf(nh[d], siluf(zh[d])) for d in range(gdn.HEAD_DIM))
    linear = runtime.matvec(view("ssm_out.weight"), metas[f"{p}.ssm_out.weight"], gated)
    residual = [addf(hidden[i], linear[i]) for i in range(gdn.HIDDEN)]
    post = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    fo = ffn(runtime, view, metas, p, post)
    return [addf(residual[i], fo[i]) for i in range(gdn.HIDDEN)], qkv


def softmax2(a: float, b: float) -> tuple[float, float]:
    m = max(float(a), float(b))
    ea = f32(exact.expf(f32(a - m))); eb = f32(exact.expf(f32(b - m)))
    d = addf(ea, eb)
    return f32(ea / d), f32(eb / d)


def full_attn_step(runtime, cache, view, metas, vec, hidden: Sequence[float], layer: int, token_index: int):
    p = f"blk.{layer}"
    x = gdn.rms_norm(hidden, vec("attn_norm.weight"))
    qg = runtime.matvec(view("attn_q.weight"), metas[f"{p}.attn_q.weight"], x)
    q, gate = attn.split_q_gate(qg)
    q = attn.rms_norm_heads(q, attn.N_HEAD, vec("attn_q_norm.weight"))
    k = runtime.matvec(view("attn_k.weight"), metas[f"{p}.attn_k.weight"], x)
    v = runtime.matvec(view("attn_v.weight"), metas[f"{p}.attn_v.weight"], x)
    k = attn.rms_norm_heads(k, attn.N_HEAD_KV, vec("attn_k_norm.weight"))
    q_rope = rope_text_neox(q, attn.N_HEAD, token_index)
    k_rope = rope_text_neox(k, attn.N_HEAD_KV, token_index)
    k_cache = attn.f16_roundtrip(k_rope); v_cache = attn.f16_roundtrip(v)
    cache["k"].append(k_cache); cache["v"].append(v_cache)

    qh = attn.split_heads(q_rope, attn.N_HEAD)
    pregate = []
    for qidx in range(attn.N_HEAD):
        kvh = qidx // attn.GQA_REPEAT
        qv = qh[qidx]
        if token_index == 0:
            pregate.extend(cache["v"][0][kvh*attn.HEAD_DIM:(kvh+1)*attn.HEAD_DIM])
            continue
        kh0 = cache["k"][0][kvh*attn.HEAD_DIM:(kvh+1)*attn.HEAD_DIM]
        kh1 = cache["k"][1][kvh*attn.HEAD_DIM:(kvh+1)*attn.HEAD_DIM]
        s0 = f32(math.fsum(float(qv[d])*float(kh0[d]) for d in range(attn.HEAD_DIM)) * SCALE_ATTN)
        s1 = f32(math.fsum(float(qv[d])*float(kh1[d]) for d in range(attn.HEAD_DIM)) * SCALE_ATTN)
        p0,p1 = softmax2(s0,s1)
        vv0 = cache["v"][0][kvh*attn.HEAD_DIM:(kvh+1)*attn.HEAD_DIM]
        vv1 = cache["v"][1][kvh*attn.HEAD_DIM:(kvh+1)*attn.HEAD_DIM]
        pregate.extend(addf(mulf(p0,vv0[d]), mulf(p1,vv1[d])) for d in range(attn.HEAD_DIM))

    gs = [exact.sigmoid_f32(vv) for vv in gate]
    gated = [mulf(pregate[i], gs[i]) for i in range(attn.Q_DIM)]
    ao = runtime.matvec(view("attn_output.weight"), metas[f"{p}.attn_output.weight"], gated)
    residual = [addf(hidden[i], ao[i]) for i in range(gdn.HIDDEN)]
    post = gdn.rms_norm(residual, vec("post_attention_norm.weight"))
    fo = ffn(runtime, view, metas, p, post)
    final = [addf(residual[i], fo[i]) for i in range(gdn.HIDDEN)]
    return final, {"Qcur": q_rope, "Kcur": k_rope, "Vcur": v}


def execute(model: Path, native_lib: Path, state_lib_path: Path, inventory_json: Path,
            oracle_json: Path, work_dir: Path, output: Path):
    started=time.monotonic(); exact.install()
    inv=json.loads(inventory_json.read_text()); oracle=json.loads(oracle_json.read_text())
    if inv.get("status")!="PASS" or inv.get("sha256")!=gdn.SHA256: raise RuntimeError("inventory not PASS")
    if oracle.get("schema")!="qwen38-llama-full64-two-token-oracle-v1" or not oracle.get("captured_complete"): raise RuntimeError("two-token oracle incomplete")
    ref=oracle["checkpoints"]; bos=int(oracle["bos_token"]); token1=int(oracle["token1"]); oracle_token2=int(oracle["token2"])
    directory=parse_gguf(model)
    trunk=work_dir/"decoder64.k3.bin"; manifest_path=work_dir/"decoder64.k3.json"
    manifest=pack_gguf_layers(directory,trunk,manifest_path,layers=range(N_LAYER),model_id=gdn.MODEL_ID,revision=gdn.REVISION,source_sha256=gdn.SHA256,expected_layers=N_LAYER)
    max_layer=max(int(x["read_bytes"]) for x in manifest["layers"])
    runtime=gdn.QuantRuntime(gdn._load_native(native_lib)); state_lib=load_state_lib(state_lib_path)
    states={il:(ctypes.c_float*STATE_ELEMS)() for il in range(N_LAYER) if il%4!=3}
    prev_qkv:dict[int,list[float]]={}; caches={il:{"k":[],"v":[]} for il in range(N_LAYER) if il%4==3}

    # token0 builds persistent state; token1 consumes the proven first argmax.
    hidden0=gdn._embedding_row(model,directory,bos)
    hidden1=gdn._embedding_row(model,directory,token1)
    layer1_metrics={}; rope_metrics={}
    with K3Trunk(trunk,manifest_path,budget_bytes=2*max_layer,want_ring=2,max_pinned=0,prefer_direct_io=True) as reader:
        for token_index in (0,1):
            hidden=hidden0 if token_index==0 else hidden1
            for il in range(N_LAYER):
                bound=reader.bind(il)
                if il+1<N_LAYER: reader.prefetch(il+1)
                metas=base._layer_meta(manifest,il); p=f"blk.{il}"
                def view(s): return reader.tensor_view(bound,f"{p}.{s}")
                def vec(s): return gdn.f32_vector(view(s))
                if il%4==3:
                    hidden, diag=full_attn_step(runtime,caches[il],view,metas,vec,hidden,il,token_index)
                    if token_index==1:
                        for name in ("Qcur","Kcur","Vcur"):
                            r=ref.get(f"{name}-{il}")
                            if r is not None: rope_metrics[f"{name}-{il}"]=base.metrics(r,diag[name])
                else:
                    hidden,qkv=recurrent_step(runtime,state_lib,states[il],prev_qkv.get(il,[0.0]*gdn.CONV_DIM),view,metas,vec,hidden,il,token_index)
                    prev_qkv[il]=qkv
                if token_index==1:
                    r=ref.get(f"post_ffn-{il}")
                    if r is not None: layer1_metrics[str(il)]=base.metrics(r,hidden)
                bound.release()
            if token_index==1: token1_final=hidden
        reader_report=reader.report()

    tensors=directory.by_name(); normw=base._read_f32_tensor(model,tensors["output_norm.weight"])
    result_norm=gdn.rms_norm(token1_final,normw)
    logits=base._stream_q8_logits(model,tensors["output.weight"],runtime,result_norm)
    top10=base._topk(logits,10); token2=int(top10[0]["token"])
    ref_logits=ref["result_output"]; oracle_top10=base._topk(ref_logits,10)
    first_bad=None
    for il in range(N_LAYER):
        m=layer1_metrics.get(str(il))
        if m and base._over_limit(m,(2e-2,5e-3)):
            first_bad=il; break
    result={
        "schema":"qwen38-k3-two-token-v1","status":"PASS" if token2==oracle_token2 else "FAIL",
        "bos_token":bos,"token1":token1,"candidate_token2":token2,"oracle_token2":oracle_token2,
        "candidate_top10":top10,"oracle_top10":oracle_top10,
        "top5_overlap":len({x['token'] for x in top10[:5]} & {x['token'] for x in oracle_top10[:5]}),
        "position":1,"first_bad_layer_loose":first_bad,"token1_layer_metrics":layer1_metrics,
        "rope_checkpoint_metrics":rope_metrics,
        "final_metrics":{"post_ffn-63":base.metrics(ref["post_ffn-63"],token1_final),"result_norm":base.metrics(ref["result_norm"],result_norm),"result_output":base.metrics(ref_logits,logits)},
        "state":{"recurrent_layers":48,"gdn_state_bytes_f32":48*STATE_BYTES_PER_LAYER,"conv_history_current_bytes_f32":48*gdn.CONV_DIM*4,"attention_kv_bytes_f16":16*2*2*attn.KV_DIM},
        "candidate":{"reader_report":reader_report,"activation_quantizations":runtime.activation_quantizations,"matvec_rows":runtime.matvec_rows,"native_gdn_state_kernel":True},
        "elapsed_seconds":time.monotonic()-started,"max_rss_gib":rss_gib(),
    }
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))
    if token2!=oracle_token2: raise SystemExit(1)
    return result


def sanity():
    exact.install()
    if ROPE_N_ROT!=64 or abs(rope_text_neox([1.0]*attn.HEAD_DIM,1,0)[0]-1.0)>0: raise SystemExit("rope sanity")
    if STATE_BYTES_PER_LAYER!=3145728: raise SystemExit("state geometry")
    print(json.dumps({"schema":"qwen38-k3-two-token-sanity-v1","status":"PASS","state_bytes":48*STATE_BYTES_PER_LAYER,"rope_theta":ROPE_THETA,"rope_n_rot":ROPE_N_ROT},indent=2))


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True); sub.add_parser("sanity")
    r=sub.add_parser("run"); r.add_argument("--model",type=Path,required=True); r.add_argument("--native-lib",type=Path,required=True); r.add_argument("--state-lib",type=Path,required=True); r.add_argument("--inventory",type=Path,required=True); r.add_argument("--oracle",type=Path,required=True); r.add_argument("--work-dir",type=Path,required=True); r.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    if a.cmd=="sanity": sanity()
    else: a.work_dir.mkdir(parents=True,exist_ok=True); execute(a.model,a.native_lib,a.state_lib,a.inventory,a.oracle,a.work_dir,a.output)
if __name__=="__main__": main()
