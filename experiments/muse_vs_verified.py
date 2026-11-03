"""Does the verified path beat Muse answering directly?

20 exact-answer tasks with one correct output (computed in Python):
  count a letter in a 300-char string, add two 40-digit numbers, digit sum of a 60-digit number,
  Collatz step count for n < 10,000.
Arm A (direct): Muse answers in text.
Arm B (verified): Muse writes a C program (it never sees the expected answer) -> the program runs inside
  the transformer computer (transformer-vm) -> the referee checks the trace against the WASM reference ->
  the printed output is the answer, with a hash-chain receipt.
Usage: python experiments/muse_vs_verified.py [--n-per-kind 5]
"""
import argparse, json, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee import llama_front, protocol, tvm  # noqa: E402


def collatz(n):
    s = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        s += 1
    return s


def tasks(rng, k):
    out = []
    for _ in range(k):
        s = "".join(rng.choice("abcdefghij") for _ in range(300)); ch = rng.choice("abcdefghij")
        out.append(("count", f"How many times does the letter '{ch}' appear in the input string?", s, str(s.count(ch))))
    for _ in range(k):
        a, b = rng.randrange(10**39, 10**40), rng.randrange(10**39, 10**40)
        out.append(("add40", "The input is two 40-digit numbers separated by '+'. Print their exact sum.", f"{a}+{b}", str(a + b)))
    for _ in range(k):
        x = str(rng.randrange(10**59, 10**60))
        out.append(("digitsum60", "Print the sum of the decimal digits of the input number.", x, str(sum(map(int, x)))))
    for _ in range(k):
        n = rng.randrange(1000, 10000)
        out.append(("collatz", "Print the number of Collatz steps for the input number to reach 1 (n -> n/2 if even, 3n+1 if odd).", str(n), str(collatz(n))))
    return out


def direct(question, inp):
    text = llama_front.chat("Answer exactly. Do not write code. End with one line: ANSWER: <answer>",
                            f"{question}\nInput: {inp}", max_tokens=32768)
    m = re.findall(r"ANSWER:\s*([^\n]+)", text or "")
    return m[-1].strip().strip("`*") if m else (text or "").strip()[-40:]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-per-kind", type=int, default=5); ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    golden, toks, t2i = tvm.load()
    rows = []
    for i, (kind, q, inp, exp) in enumerate(tasks(rng, a.n_per_kind)):
        t0 = time.time(); got_a = direct(q, inp); ta = time.time() - t0
        t0 = time.time()
        code, _ = llama_front.write_program(q + " The input string is passed as `input`.")
        got_b = verdict = steps = receipt = None
        if code:
            try:
                txt, ref_txt = llama_front.build(code, inp, f"mv{i:02d}")
                pre = [t2i[t] for t in open(txt).read().split()]; ref = [t2i[t] for t in open(ref_txt).read().split()]
                tr, R = tvm.run_prover(golden, pre, len(ref) - len(pre) + 50)
                v = protocol.referee(golden, tr, R, ref)
                verdict, steps = v["verdict"], len(tr)
                receipt = protocol.hash_chain(tr)[-1].hex()[:16]
                got_b = llama_front.decode_output([toks[x] for x in tr]).strip()
            except Exception as e:
                verdict = "build_error: " + str(e)[-80:]
        tb = time.time() - t0
        row = {"kind": kind, "expected": exp, "direct": got_a, "direct_ok": got_a == exp, "verified": got_b,
               "verified_ok": got_b == exp, "referee": verdict, "steps": steps, "receipt": receipt,
               "sec_direct": round(ta, 1), "sec_verified": round(tb, 1)}
        rows.append(row); print(json.dumps(row), flush=True)
    kinds = sorted({r["kind"] for r in rows})
    summ = {k: {"n": sum(r["kind"] == k for r in rows), "direct_ok": sum(r["direct_ok"] for r in rows if r["kind"] == k),
                "verified_ok": sum(r["verified_ok"] for r in rows if r["kind"] == k)} for k in kinds}
    summ["total"] = {"n": len(rows), "direct_ok": sum(r["direct_ok"] for r in rows), "verified_ok": sum(r["verified_ok"] for r in rows)}
    json.dump({"summary": summ, "rows": rows, "backend": llama_front.backend(), "usage": llama_front._USAGE},
              open("results/muse_vs_verified.json", "w"), indent=1)
    print(json.dumps(summ))


if __name__ == "__main__":
    main()
