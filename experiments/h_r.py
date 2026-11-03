"""H-R test: inject random weight faults until N change the program's output, then check whether the
referee (a) finds the same first wrong step as a full linear comparison, (b) names the sublayer that
actually holds the corrupted weight, (c) re-executes only one step.

Pass rule (fixed before running, from the M2 sheet): >= 9 of 10 harmful faults located exactly.
Usage: python experiments/h_r.py [--n 10] [--seed 1] [--programs hello,addition,fibonacci]
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee import faults, protocol, tvm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--programs", default="hello,addition,fibonacci")
    ap.add_argument("--max-tries", type=int, default=200)
    ap.add_argument("--out", default="results/h_r.json")
    ap.add_argument("--stratify", action="store_true")
    a = ap.parse_args()

    rng = random.Random(a.seed)
    golden, all_tokens, t2i = tvm.load()
    progs = {p: tvm.program(p, t2i) for p in a.programs.split(",")}
    harmful, benign, tries = [], [], 0
    while len(harmful) < a.n and tries < a.max_tries:
        tries += 1
        name = rng.choice(list(progs))
        prefix, ref = progs[name]
        kind = faults.KINDS[tries % len(faults.KINDS)]
        bad, info = faults.inject(golden, rng, kind=kind, stratify=a.stratify)
        t0 = time.time()
        toks, R = tvm.run_prover(bad, prefix, len(ref) - len(prefix) + 50)
        prover_s = time.time() - t0
        info.update(program=name, try_no=tries)
        if toks == ref:
            benign.append(info)
            continue
        t0 = time.time()
        v = protocol.referee(golden, toks, R, ref)
        ref_s = time.time() - t0
        oracle_k = protocol.linear_first_divergence(toks, ref)
        row = {**info,
               "oracle_first_wrong_step": oracle_k,
               "referee_first_wrong_step": v["first_wrong_step"],
               "referee_sublayer": v["faulty_sublayer"],
               "stage": v.get("stage"), "fault_origin_position": v.get("fault_origin_position"),
               "step_correct": v["first_wrong_step"] == oracle_k,
               "sublayer_correct": v["faulty_sublayer"] == info["true_sublayer"],
               "hash_comparisons": v["hash_comparisons"],
               "trace_len": v["trace_len"],
               "steps_reexecuted": v.get("steps_reexecuted"),
               "prover_seconds": round(prover_s, 2),
               "referee_seconds": round(ref_s, 2),
               "prover_token": all_tokens[v["prover_token"]] if v.get("prover_token") is not None else None,
               "reference_token": all_tokens[v["reference_token"]] if v.get("reference_token") is not None else None}
        harmful.append(row)
        print(f"[{len(harmful)}/{a.n}] {name} {info['kind']:13s} true={info['true_sublayer']:9s} "
              f"found={v['faulty_sublayer']} [{v.get('stage')}] step {v['first_wrong_step']} (oracle {oracle_k}) "
              f"cmp={v['hash_comparisons']} prover {prover_s:.1f}s referee {ref_s:.2f}s", flush=True)

    both = sum(r["step_correct"] and r["sublayer_correct"] for r in harmful)
    summary = {"harmful": len(harmful), "benign": len(benign), "tries": tries,
               "located_exactly": both,
               "pass_rule": ">= 9 of 10 located exactly",
               "pass": len(harmful) >= a.n and both >= round(0.9 * a.n),
               "found_at_wrong_step": sum(r["stage"] == "step" for r in harmful),
               "needed_earlier_scan": sum(r["stage"] == "scan" for r in harmful),
               "referee_to_prover_time_ratio": round(sum(r["referee_seconds"] for r in harmful) / max(sum(r["prover_seconds"] for r in harmful), 1e-9), 3),
               "mean_prover_seconds": round(sum(r["prover_seconds"] for r in harmful) / max(len(harmful), 1), 2),
               "mean_referee_seconds": round(sum(r["referee_seconds"] for r in harmful) / max(len(harmful), 1), 2)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": summary, "stratify": a.stratify, "harmful": harmful, "benign": benign, "seed": a.seed}, open(a.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
