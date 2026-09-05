#!/usr/bin/env python3
"""Diagnose the remaining isolated full64 local spike at layer 59.

Attention uses the two already-proven pinned-GGML fixes (RMSNorm and F32
sigmoid/gate multiplication).  This probe then tests llama.cpp's F32 SwiGLU
materialization before the Q6_K FFN down projection, with oracle injections to
separate FFN pointwise arithmetic from the down kernel and residual adds.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
from typing import Any, Sequence

import qwen35_gdn_quant_layer_gate as gdn
import qwen35_full_attn_layer3_gate as attn
import qwen35_k3_full64_one_token as full64
import qwen35_k3_full64_ggml_rmsnorm as rmswrap
import qwen35_k3_full64_ggml_exact as exact
from gguf_k3_layout import pack_gguf_layers
from gguf_stream import parse_gguf
from k3_stream import K3Trunk

LAYER=59; PREV=58
MODES=("baseline_exact_attention","f32_swiglu","inject_ffn_swiglu","inject_ffn_out","f32_all_pointwise")

def f32(x: float)->float: return exact.f32(x)
def silu_f32(x: float)->float:
    xf=f32(x); denom=f32(f32(1.0)+f32(exact.expf(f32(-xf))))
    return f32(xf/denom)
def add_f32(a:float,b:float)->float: return f32(f32(a)+f32(b))
def metrics(ref:Sequence[float],cand:Sequence[float])->dict[str,float]: return full64.metrics(ref,cand)

def _layer_meta(manifest:dict[str,Any])->dict[str,dict[str,Any]]:
    e=next(x for x in manifest["layers"] if int(x["layer"])==LAYER)
    return {t["name"]:t for t in e["tensors"]}

def run_layer(runtime,view,metas,vec,hidden:Sequence[float],ref:dict[str,list[float]],mode:str)->dict[str,list[float]]:
    if mode not in MODES: raise ValueError(mode)
    p=f"blk.{LAYER}"; cp:dict[str,list[float]]={}
    attn_norm=gdn.rms_norm(hidden,vec("attn_norm.weight")); cp["attn_norm-59"]=attn_norm
    qg=runtime.matvec(view("attn_q.weight"),metas[f"{p}.attn_q.weight"],attn_norm)
    _q,gate=attn.split_q_gate(qg)
    v=runtime.matvec(view("attn_v.weight"),metas[f"{p}.attn_v.weight"],attn_norm)
    cp["Qcur_full-59"]=qg; cp["gate_reshaped-59"]=gate; cp["Vcur-59"]=v
    pregate=attn.gqa_one_key_attention(attn.f16_roundtrip(v)); cp["attn_pregate-59"]=pregate
    gate_sig=[exact.sigmoid_f32(x) for x in gate]; cp["gate_sigmoid-59"]=gate_sig
    gated=[exact.mul_f32(pregate[i],gate_sig[i]) for i in range(attn.Q_DIM)]; cp["attn_gated-59"]=gated
    attn_out=runtime.matvec(view("attn_output.weight"),metas[f"{p}.attn_output.weight"],gated); cp["attn_output-59"]=attn_out
    use_all=(mode=="f32_all_pointwise")
    residual=[add_f32(hidden[i],attn_out[i]) if use_all else float(hidden[i])+attn_out[i] for i in range(gdn.HIDDEN)]
    cp["attn_residual-59"]=residual
    post_norm=gdn.rms_norm(residual,vec("post_attention_norm.weight")); cp["attn_post_norm-59"]=post_norm
    prepared=runtime.quantize(post_norm,"Q6_K")
    fg=runtime.matvec(view("ffn_gate.weight"),metas[f"{p}.ffn_gate.weight"],post_norm,prepared)
    fu=runtime.matvec(view("ffn_up.weight"),metas[f"{p}.ffn_up.weight"],post_norm,prepared)
    cp["ffn_gate-59"]=fg; cp["ffn_up-59"]=fu
    if mode in ("f32_swiglu","f32_all_pointwise"):
        sw=[exact.mul_f32(silu_f32(fg[i]),fu[i]) for i in range(gdn.INTERMEDIATE)]
    else:
        sw=[gdn.silu(fg[i])*fu[i] for i in range(gdn.INTERMEDIATE)]
    if mode=="inject_ffn_swiglu": sw=list(map(float,ref["ffn_swiglu-59"]))
    cp["ffn_swiglu-59"]=sw
    ffn_out=runtime.matvec(view("ffn_down.weight"),metas[f"{p}.ffn_down.weight"],sw)
    if mode=="inject_ffn_out": ffn_out=list(map(float,ref["ffn_out-59"]))
    cp["ffn_out-59"]=ffn_out
    cp["post_ffn-59"]=[add_f32(residual[i],ffn_out[i]) if use_all else residual[i]+ffn_out[i] for i in range(gdn.HIDDEN)]
    return cp

def execute(model:Path,native_lib:Path,oracle_json:Path,work_dir:Path,output:Path)->dict[str,Any]:
    started=time.monotonic(); exact.install()
    oracle=json.loads(oracle_json.read_text(encoding="utf-8"))
    if oracle.get("schema")!="qwen38-llama-layer59-ffn-oracle-v1" or not oracle.get("captured_complete"): raise RuntimeError("layer59 oracle incomplete")
    ref=oracle["checkpoints"]; hidden=list(map(float,ref["layer59_input"]))
    directory=parse_gguf(model); trunk=work_dir/"layer59.k3.bin"; manifest_path=work_dir/"layer59.k3.json"
    manifest=pack_gguf_layers(directory,trunk,manifest_path,layers=[LAYER],model_id=gdn.MODEL_ID,revision=gdn.REVISION,source_sha256=gdn.SHA256,expected_layers=gdn.DECODER_LAYERS)
    runtime=gdn.QuantRuntime(gdn._load_native(native_lib)); budget=int(manifest["layers"][0]["read_bytes"])
    with K3Trunk(trunk,manifest_path,budget_bytes=budget,want_ring=1,max_pinned=0,prefer_direct_io=True) as reader:
        bound=reader.bind(LAYER); metas=_layer_meta(manifest)
        def view(s:str): return reader.tensor_view(bound,f"blk.{LAYER}.{s}")
        def vec(s:str): return gdn.f32_vector(view(s))
        ordered=["attn_norm-59","gate_reshaped-59","Vcur-59","attn_pregate-59","gate_sigmoid-59","attn_gated-59","attn_output-59","attn_residual-59","attn_post_norm-59","ffn_gate-59","ffn_up-59","ffn_swiglu-59","ffn_out-59","post_ffn-59"]
        modes={}
        for mode in MODES:
            cp=run_layer(runtime,view,metas,vec,hidden,ref,mode)
            comp={k:metrics(ref[k],cp[k]) for k in ordered if k in ref and k in cp}
            modes[mode]={"checkpoint_metrics":comp,"output_metrics":comp["post_ffn-59"]}
        report=reader.report(); bound.release()
    out_rel={m:modes[m]["output_metrics"]["relative_l2"] for m in MODES}; out_max={m:modes[m]["output_metrics"]["max_abs"] for m in MODES}
    result={"schema":"qwen38-full-attn-layer59-ffn-probe-v1","status":"PASS","diagnostic_only":True,"token_id":int(oracle["token_id"]),"input_source":oracle.get("input_source"),"modes":modes,"output_relative_l2":out_rel,"output_max_abs":out_max,"best_mode_relative_l2":min(out_rel,key=out_rel.get),"best_mode_max_abs":min(out_max,key=out_max.get),"reader_report":report,"elapsed_seconds":time.monotonic()-started,"swiglu_contract":"ggml_silu_f32(x)=x/(1.0f+expf(-x)); F32 multiply before FFN down projection"}
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2,sort_keys=True)); return result

def sanity()->None:
    exact.install(); x=f32(0.25); y=silu_f32(x); z=exact.mul_f32(y,f32(0.5))
    if not all(math.isfinite(v) for v in (x,y,z)) or LAYER%4!=3 or PREV!=58 or len(MODES)!=5: raise SystemExit("layer59 FFN sanity failed")
    print(json.dumps({"schema":"qwen38-full-attn-layer59-ffn-sanity-v1","status":"PASS","layer":LAYER,"modes":MODES},indent=2))

def main()->None:
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True); sub.add_parser("sanity")
    r=sub.add_parser("run"); r.add_argument("--model",type=Path,required=True); r.add_argument("--native-lib",type=Path,required=True); r.add_argument("--oracle",type=Path,required=True); r.add_argument("--work-dir",type=Path,required=True); r.add_argument("--output",type=Path,required=True)
    a=ap.parse_args();
    if a.cmd=="sanity": sanity()
    else: a.work_dir.mkdir(parents=True,exist_ok=True); execute(a.model,a.native_lib,a.oracle,a.work_dir,a.output)
if __name__=="__main__": main()
