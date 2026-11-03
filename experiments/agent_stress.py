"""Stress test: every execution on the transformer computer is hit by K random bit flips (activations or
attention memory), like silent data corruption on a bad machine. Compare what gets delivered:
  unverified pipeline: first execution's output, no checking
  Muse agent loop: referee + self-tests + retry/repair, delivers only PASS answers with a receipt
Ground truth is computed in Python. Usage: python experiments/agent_stress.py [--k 8] [--seed 9]
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee import agent, llama_front, tvm  # noqa: E402


def collatz(n):
    s = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1; s += 1
    return s


def tasks(rng):
    out = []
    for _ in range(3):
        s = "".join(rng.choice("abcdefghij") for _ in range(120)); ch = rng.choice("abcdefghij")
        out.append((f"Count how many times the letter '{ch}' appears in the input string and print the count.", s, str(s.count(ch))))
    for _ in range(3):
        x = str(rng.randrange(10**29, 10**30))
        out.append(("Print the sum of the decimal digits of the input number.", x, str(sum(map(int, x)))))
    for _ in range(3):
        a, b = rng.randrange(10**19, 10**20), rng.randrange(10**19, 10**20)
        out.append(("The input is two 20-digit numbers separated by '+'. Print their exact sum.", f"{a}+{b}", str(a + b)))
    for _ in range(3):
        n = rng.randrange(100, 1000)
        out.append(("Print the number of Collatz steps for the input number to reach 1 (n -> n/2 if even, 3n+1 if odd).", str(n), str(collatz(n))))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--k", type=int, default=8); ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    golden, toks, t2i = tvm.load()
    L, D = len(golden.attn), golden.tok.weight.shape[1]

    def fault_fn(n_pre, n_ref):
        fs = []
        for _ in range(a.k):
            f = {"type": rng.choice(["act", "kv"]), "pos": rng.randrange(n_pre, n_ref - 1), "dim": rng.randrange(D), "bit": rng.choice(range(52, 62))}
            if f["type"] == "act":
                f["sublayer"] = rng.randrange(2 * L + 1)
            else:
                f["layer"], f["which"] = rng.randrange(L), rng.choice("kv")
            fs.append(f)
        return fs

    rows = []
    for i, (task, inp, truth) in enumerate(tasks(rng)):
        t0 = time.time()
        log = agent.run(task, inp, golden, toks, t2i, fault_fn, f"ag{i:02d}")
        first = log["attempts"][0]["output"] if log["attempts"] else None
        row = {"task": task[:60], "truth": truth, "reference_output": log.get("reference_output"),
               "program_correct": log.get("reference_output") == truth,
               "unverified_delivered": first, "unverified_wrong": first is not None and first != truth,
               "agent_delivered": log["delivered"], "agent_wrong": log["delivered"] is not None and log["delivered"] != truth,
               "agent_abstained": log["delivered"] is None, "attempts": len(log["attempts"]), "repairs": log["repairs"],
               "decisions": [d.get("action") for d in log["decisions"]], "receipt": log.get("receipt", "")[:16],
               "seconds": round(time.time() - t0)}
        rows.append({**row, "log": log})
        print(json.dumps(row), flush=True)
    n = len(rows)
    summ = {"n": n, "k_faults_per_run": a.k,
            "unverified_wrong_delivered": sum(r["unverified_wrong"] for r in rows),
            "agent_wrong_delivered": sum(r["agent_wrong"] for r in rows),
            "agent_correct_delivered": sum(r["agent_delivered"] == r["truth"] for r in rows),
            "agent_abstained": sum(r["agent_abstained"] for r in rows),
            "program_repairs": sum(r["repairs"] for r in rows),
            "executions_total": sum(r["attempts"] for r in rows),
            "executions_failed_by_referee": sum(1 for r in rows for x in r["log"]["attempts"] if x["verdict"] == "FAIL"),
            "muse_retry_decisions": sum(d == "retry" for r in rows for d in r["decisions"]),
            "backend": llama_front.backend()}
    json.dump({"summary": summ, "rows": rows, "usage": llama_front._USAGE}, open("results/agent_stress.json", "w"), indent=1, default=str)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
