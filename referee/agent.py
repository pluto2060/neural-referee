"""Muse agent loop: write -> self-test -> run on the neural computer -> referee -> decide -> receipt.

1. Muse writes the program and three small self-tests it can work out by hand (input -> output).
2. Self-tests run on the plain WebAssembly reference; failures go back to Muse for repair.
3. The real input runs on the transformer computer (possibly hit by hardware faults).
4. The referee returns PASS with a receipt, or FAIL with the first wrong step and sublayer.
5. On FAIL, Muse reads the verdict and chooses: retry the execution, repair the program, or give up.
Only PASS answers are delivered, each with the hash of its full trace.
"""
import json
import re

from referee import llama_front, protocol, tvm

# The prover keeps every residual (16 x 38 float64, about 5 KB per step) for the referee.
# An 11.3M-step trace (a Collatz program that halves by repeated subtraction) used >20 GB of RAM,
# so refuse traces longer than this and report it instead of running.
MAX_STEPS = 200_000

SYS_TESTS = ("You write test cases for a tiny program. Given a task, output JSON only: "
             '{"tests": [{"input": "...", "output": "..."}, ...]} with exactly 3 small inputs whose correct '
             "output you are certain of. Output is exactly what the program prints, without the trailing newline.")
SYS_DECIDE = ("You operate a computer that runs inside a transformer and a referee that checks its execution. "
              "Given the task, the self-test results and the referee report, reply with JSON only: "
              '{"action": "retry" | "repair" | "give_up", "reason": "<one sentence>"}. '
              "retry = the program passed its tests but this execution diverged from the reference (hardware fault); "
              "repair = the program itself is wrong; give_up = cannot be solved.")


def _json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    return json.loads(m.group(0)) if m else {}


def self_test(code, tests, name):
    fails = []
    for j, t in enumerate(tests):
        try:
            _, ref_txt = llama_front.build(code, t["input"], f"{name}_t{j}")
            got = llama_front.decode_output(open(ref_txt).read().split()).strip()
        except Exception as e:
            got = "error: " + str(e)[-120:]
        if got != str(t["output"]).strip():
            fails.append({"input": t["input"], "expected": t["output"], "got": got})
    return fails


def run(task, inp, golden, toks, t2i, fault_fn, name, max_attempts=4, max_repairs=2):
    log = {"task": task, "input": inp, "repairs": 0, "attempts": [], "decisions": []}
    tests = _json(llama_front.chat(SYS_TESTS, task, max_tokens=4096)).get("tests", [])[:3]
    log["tests"] = tests
    req = task + " The input string is passed as `input`."
    code, _ = llama_front.write_program(req)
    for _ in range(max_repairs + 1):
        if code is None:
            break
        fails = self_test(code, tests, name) if tests else []
        log["self_test_fails"] = fails
        if not fails:
            break
        if log["repairs"] >= max_repairs:
            break
        log["repairs"] += 1
        code, _ = llama_front.write_program(f"{req}\nYour previous program:\n```c\n{code}\n```\nfailed these tests: "
                                            f"{json.dumps(fails)}. Write a corrected program.")
    log["code"] = code
    if code is None:
        log["delivered"] = None
        return log
    txt, ref_txt = llama_front.build(code, inp, name)
    pre = [t2i[t] for t in open(txt).read().split()]
    ref = [t2i[t] for t in open(ref_txt).read().split()]
    log["reference_output"] = llama_front.decode_output([toks[x] for x in ref]).strip()
    if len(ref) > MAX_STEPS:
        log["delivered"] = None
        log["refused"] = f"trace has {len(ref)} steps, above MAX_STEPS={MAX_STEPS}"
        return log
    for a in range(max_attempts):
        faults = fault_fn(len(pre), len(ref))
        tr, R = tvm.run_prover(golden, pre, len(ref) - len(pre) + 50, runtime_fault=faults)
        v = protocol.referee(golden, tr, R, ref)
        out = llama_front.decode_output([toks[x] for x in tr]).strip()
        att = {"n_faults": len(faults), "verdict": v["verdict"], "output": out, "steps": len(tr),
               "first_wrong_step": v.get("first_wrong_step"), "faulty_sublayer": v.get("faulty_sublayer"),
               "receipt": protocol.hash_chain(tr)[-1].hex()}
        log["attempts"].append(att)
        if v["verdict"] == "PASS":
            log["delivered"] = out
            log["receipt"] = att["receipt"]
            return log
        d = _json(llama_front.chat(SYS_DECIDE, json.dumps({"task": task, "self_test_failures": log.get("self_test_fails", []),
              "referee": {k: att[k] for k in ("verdict", "first_wrong_step", "faulty_sublayer", "steps")}}), max_tokens=4096))
        log["decisions"].append(d)
        if d.get("action") != "retry":
            break
    log["delivered"] = None
    return log
