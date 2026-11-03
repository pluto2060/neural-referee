"""End-to-end demo: request -> Llama writes C -> neural computer runs it -> referee checks -> Llama explains.

python demo/run_demo.py "print the sum of the digits of the input number" --input 98765 \
    [--fault act:pos=1200,sublayer=7,dim=3,bit=60 | --fault kv:pos=900,layer=2,which=v,dim=1,bit=55 | --fault weight:seed=4]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee import faults, llama_front, protocol, tvm  # noqa: E402


def parse_fault(s, golden):
    if not s:
        return golden, None, None
    kind, _, rest = s.partition(":")
    kv = dict(x.split("=") for x in rest.split(",") if x)
    if kind == "weight":
        bad, info = faults.inject(golden, random.Random(int(kv.get("seed", 0))), stratify=True)
        return bad, None, info
    rf = {"type": kind}
    for k, v in kv.items():
        rf[k] = v if k == "which" else int(v)
    return golden, rf, rf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("--input", default="")
    ap.add_argument("--fault", default="")
    ap.add_argument("--name", default="llama_prog")
    ap.add_argument("--out", default="results/demo_last.json")
    ap.add_argument("--expect", default=None, help="expected program output; wrong programs go back to Llama")
    a = ap.parse_args()

    t0 = time.time()
    if a.expect is not None:
        code, txt, ref_txt, log = llama_front.write_checked_program(a.request, a.input, a.expect, a.name)
    else:
        code, log = llama_front.write_program(a.request)
        txt = ref_txt = None
    print(f"[1] Llama wrote a program ({len(log)} round(s), {time.time() - t0:.0f}s)")
    print(json.dumps(log, indent=1, default=str))
    if code is None:
        sys.exit(1)
    print(code)
    if txt is None:
        txt, ref_txt = llama_front.build(code, a.input, a.name)
    golden, all_tokens, t2i = tvm.load()
    prefix = [t2i[t] for t in open(txt).read().split()]
    ref = [t2i[t] for t in open(ref_txt).read().split()]
    model, rf, finfo = parse_fault(a.fault, golden)
    t0 = time.time()
    toks, R = tvm.run_prover(model, prefix, len(ref) - len(prefix) + 50, runtime_fault=rf)
    print(f"[2] neural computer ran {len(toks)} steps in {time.time() - t0:.1f}s; fault: {finfo}")
    v = protocol.referee(golden, toks, R, ref)
    v["prover_output"] = llama_front.decode_output([all_tokens[i] for i in toks])
    v["reference_output"] = llama_front.decode_output([all_tokens[i] for i in ref])
    print(f"[3] referee: {v['verdict']} step={v.get('first_wrong_step')} sublayer={v.get('faulty_sublayer')} "
          f"position={v.get('fault_origin_position')} hash_comparisons={v['hash_comparisons']}")
    print(f"    output: neural={v['prover_output']!r} reference={v['reference_output']!r}")
    expl = llama_front.explain(v)
    print("[4] Llama explains:\n" + expl)
    json.dump({"request": a.request, "input": a.input, "code": code, "llama_log": log, "fault": finfo,
               "referee": {k: v[k] for k in v if k != "sublayer_checks"}, "explanation": expl},
              open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
