"""Finish the stress test: the first run stopped in the Collatz group because the model's program halves by
repeated subtraction (11.3M steps for n < 1000). Run 3 Collatz tasks with small n (5..40) under the same
fault model (8 bit flips per run) and append them to results/agent_stress.json."""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee import agent, llama_front, tvm  # noqa: E402
from experiments.agent_stress import collatz  # noqa: E402

K = 8


def main():
    rng = random.Random(19)
    golden, toks, t2i = tvm.load()
    L, D = len(golden.attn), golden.tok.weight.shape[1]

    def fault_fn(n_pre, n_ref):
        fs = []
        for _ in range(K):
            f = {"type": rng.choice(["act", "kv"]), "pos": rng.randrange(n_pre, n_ref - 1), "dim": rng.randrange(D), "bit": rng.choice(range(52, 62))}
            if f["type"] == "act":
                f["sublayer"] = rng.randrange(2 * L + 1)
            else:
                f["layer"], f["which"] = rng.randrange(L), rng.choice("kv")
            fs.append(f)
        return fs

    data = json.load(open("results/agent_stress.json"))
    rows = data["rows"]
    task = "Print the number of Collatz steps for the input number to reach 1 (n -> n/2 if even, 3n+1 if odd)."
    for i in range(3):
        n = rng.randrange(5, 40)
        t0 = time.time()
        log = agent.run(task, str(n), golden, toks, t2i, fault_fn, f"agc{i}")
        truth = str(collatz(n))
        first = log["attempts"][0]["output"] if log["attempts"] else None
        row = {"task": task[:60] + f" n={n}", "truth": truth, "reference_output": log.get("reference_output"),
               "program_correct": log.get("reference_output") == truth,
               "unverified_delivered": first, "unverified_wrong": first is not None and first != truth,
               "agent_delivered": log["delivered"], "agent_wrong": log["delivered"] is not None and log["delivered"] != truth,
               "agent_abstained": log["delivered"] is None, "attempts": len(log["attempts"]), "repairs": log["repairs"],
               "decisions": [d.get("action") for d in log["decisions"]], "receipt": log.get("receipt", "")[:16],
               "refused": log.get("refused"), "seconds": round(time.time() - t0)}
        print(json.dumps(row), flush=True)
        rows.append({**row, "log": log})
    s = {"n": len(rows), "k_faults_per_run": K,
         "unverified_wrong_delivered": sum(r["unverified_wrong"] for r in rows),
         "agent_wrong_delivered": sum(r["agent_wrong"] for r in rows),
         "agent_correct_delivered": sum(r["agent_delivered"] == r["truth"] for r in rows),
         "agent_abstained": sum(r["agent_abstained"] for r in rows),
         "program_repairs": sum(r["repairs"] for r in rows),
         "executions_total": sum(r["attempts"] for r in rows),
         "muse_retry_decisions": sum(d == "retry" for r in rows for d in r["decisions"]),
         "note": "first 9 tasks from agent_stress.py (seed 9); the Collatz group was rerun with n in 5..40 (seed 19) after the n<1000 run hit an 11.3M-step trace"}
    data["summary"] = s
    json.dump(data, open("results/agent_stress.json", "w"), indent=1, default=str)
    print(json.dumps(s))


if __name__ == "__main__":
    main()
