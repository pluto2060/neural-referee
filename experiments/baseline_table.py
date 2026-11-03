"""Referee vs baselines on three fault classes.

Classes
  weight : one corrupted weight (as in h_r.py, stratified over sublayers)
  act    : one bit flipped in a residual during execution (compute error); weights untouched
  kv     : one bit flipped in a key or value written to attention memory (memory error); weights untouched
Methods
  output_only : compare final trace with the reference (what transformer-vm / QwenPU report today)
  weight_diff : compare prover weight files with golden weights
  referee     : this project
Only harmful faults (trace differs from the reference) are counted.
Usage: python experiments/baseline_table.py [--n 15] [--seed 7]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee import baselines, faults, protocol, tvm  # noqa: E402

BITS = {"sign": [63], "exponent": list(range(52, 62)), "mantissa_high": list(range(44, 52))}


def runtime_fault(rng, cls, n_pos, n_layers, d_model):
    kind = rng.choice(list(BITS))
    bit = rng.choice(BITS[kind])
    pos = rng.randrange(1, n_pos - 1)
    if cls == "act":
        s = rng.randrange(0, 2 * n_layers + 1)
        return {"type": "act", "pos": pos, "sublayer": s, "dim": rng.randrange(d_model), "bit": bit, "kind": kind}
    return {"type": "kv", "pos": pos, "layer": rng.randrange(n_layers), "which": rng.choice("kv"),
            "dim": rng.randrange(d_model), "bit": bit, "kind": kind}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--programs", default="hello,addition,fibonacci")
    ap.add_argument("--max-tries", type=int, default=600)
    ap.add_argument("--out", default="results/baseline_table.json")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    golden, _, t2i = tvm.load()
    names = tvm.sublayer_names(len(golden.attn))
    n_layers, d_model = len(golden.attn), golden.tok.weight.shape[1]
    progs = {p: tvm.program(p, t2i) for p in a.programs.split(",")}
    rows = []
    for cls in ["weight", "act", "kv"]:
        harmful = benign = tries = 0
        while harmful < a.n and tries < a.max_tries:
            tries += 1
            prog = rng.choice(list(progs))
            prefix, ref = progs[prog]
            if cls == "weight":
                model, info = faults.inject(golden, rng, kind=rng.choice(faults.KINDS), stratify=True)
                rf = None
                true_sub, true_pos = info["true_sublayer"], None
            else:
                model = golden
                rf = runtime_fault(rng, cls, len(ref), n_layers, d_model)
                info = dict(rf)
                true_sub = names[rf["sublayer"]] if cls == "act" else f"L{rf['layer']}.attn"
                true_pos = rf["pos"]
            toks, R = tvm.run_prover(model, prefix, len(ref) - len(prefix) + 50, runtime_fault=rf)
            if toks == ref:
                benign += 1
                continue
            harmful += 1
            t0 = time.time()
            v = protocol.referee(golden, toks, R, ref)
            ref_s = time.time() - t0
            oracle_k = protocol.linear_first_divergence(toks, ref)
            wd = baselines.weight_diff(golden, model)
            pos_ok = True if true_pos is None else (
                v.get("fault_origin_position") == true_pos if cls == "act"
                else (v.get("fault_origin_position") is not None and v["fault_origin_position"] >= true_pos))
            row = {"class": cls, "program": prog, "true_sublayer": true_sub, "true_position": true_pos,
                   "fault": info if cls != "weight" else {k: info[k] for k in ("kind", "index", "bit")},
                   "output_only_detects": not baselines.output_only(toks, ref),
                   "weight_diff_sublayer": wd,
                   "weight_diff_correct": wd == true_sub,
                   "referee_step_correct": v["first_wrong_step"] == oracle_k,
                   "referee_sublayer": v["faulty_sublayer"], "referee_stage": v.get("stage"),
                   "referee_position": v.get("fault_origin_position"),
                   "referee_correct": v["first_wrong_step"] == oracle_k and v["faulty_sublayer"] == true_sub and pos_ok,
                   "referee_seconds": round(ref_s, 2)}
            rows.append(row)
            print(f"[{cls} {harmful}/{a.n}] {prog} true={true_sub}@{true_pos} referee={v['faulty_sublayer']}@"
                  f"{v.get('fault_origin_position')} [{v.get('stage')}] weight_diff={wd} ok={row['referee_correct']}", flush=True)
        rows.append({"class": cls, "summary": True, "harmful": harmful, "benign": benign, "tries": tries})

    table = []
    for cls in ["weight", "act", "kv"]:
        rs = [r for r in rows if r["class"] == cls and not r.get("summary")]
        n = len(rs) or 1
        table.append({"class": cls, "harmful": len(rs),
                      "output_only_detects": sum(r["output_only_detects"] for r in rs) / n,
                      "output_only_locates": 0.0,
                      "weight_diff_locates": sum(r["weight_diff_correct"] for r in rs) / n,
                      "referee_locates": sum(r["referee_correct"] for r in rs) / n})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"table": table, "rows": rows, "seed": a.seed}, open(a.out, "w"), indent=1)
    print(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
