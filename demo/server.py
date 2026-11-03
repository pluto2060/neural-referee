"""Local web demo: ask -> Muse writes a program -> it runs inside the transformer computer ->
optional fault -> referee verdict -> Muse explains. No framework; python demo/server.py, open :8765."""
import json, os, random, sys, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee import agent, llama_front, protocol, tvm  # noqa: E402

GOLDEN, TOKS, T2I = tvm.load()
NAMES = tvm.sublayer_names(len(GOLDEN.attn))
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
CACHE = {}


def build(request, inp, fresh=False):
    key = (request, inp)
    if fresh or key not in CACHE:
        code, log = llama_front.write_program(request + " The input string is passed as `input`.")
        if code is None:
            raise RuntimeError("program did not compile: " + json.dumps(log)[-300:])
        txt, ref_txt = llama_front.build(code, inp, f"web{abs(hash(key)) % 10**8}")
        pre = [T2I[t] for t in open(txt).read().split()]
        ref = [T2I[t] for t in open(ref_txt).read().split()]
        CACHE[key] = (code, pre, ref)
    return CACHE[key]


def harmful_fault(pre, ref, seed):
    rng = random.Random(seed)
    L = len(GOLDEN.attn)
    for _ in range(300):
        rf = {"type": rng.choice(["act", "kv"]), "pos": rng.randrange(len(pre), len(ref) - 1),
              "dim": rng.randrange(GOLDEN.tok.weight.shape[1]), "bit": rng.choice(range(52, 62))}
        if rf["type"] == "act":
            rf["sublayer"] = rng.randrange(2 * L + 1)
        else:
            rf["layer"], rf["which"] = rng.randrange(L), rng.choice("kv")
        tr, R = tvm.run_prover(GOLDEN, pre, len(ref) - len(pre) + 50, runtime_fault=rf)
        if tr != ref:
            return rf, tr, R
    return None, None, None


def execute(pre, ref, tr, R, fault):
    v = protocol.referee(GOLDEN, tr, R, ref)
    return {"verdict": v["verdict"], "steps": len(tr), "trace_len": len(ref),
            "first_wrong_step": v.get("first_wrong_step"), "faulty_sublayer": v.get("faulty_sublayer"),
            "hash_comparisons": v["hash_comparisons"], "fault": fault,
            "output": llama_front.decode_output([TOKS[x] for x in tr]).strip(),
            "receipt": protocol.hash_chain(tr)[-1].hex()}


def run(body):
    """One request = the whole loop. With fault=true the first execution is hit by a harmful bit flip;
    the model reads the referee report and decides what to do; a clean re-run is delivered with a receipt."""
    t0 = time.time()
    code, pre, ref = build(body["request"], body.get("input", ""), fresh=bool(body.get("fresh")))
    t_write = time.time() - t0
    if len(ref) > agent.MAX_STEPS:
        raise RuntimeError(f"program needs {len(ref):,} steps; the demo limit is {agent.MAX_STEPS:,} to keep memory bounded")
    attempts, decisions = [], []
    if body.get("fault"):
        rf, tr, R = harmful_fault(pre, ref, int(body.get("seed", 1)))
        if rf is not None:
            att = execute(pre, ref, tr, R, rf)
            attempts.append(att)
            d = agent._json(llama_front.chat(agent.SYS_DECIDE, json.dumps({"task": body["request"], "self_test_failures": [],
                "referee": {k: att[k] for k in ("verdict", "first_wrong_step", "faulty_sublayer", "steps")}}), max_tokens=4096))
            decisions.append(d)
            if d.get("action") != "retry":
                return {"code": code, "attempts": attempts, "decisions": decisions, "delivered": None,
                        "sublayers": NAMES, "seconds": {"write": round(t_write, 1), "total": round(time.time() - t0, 1)}}
    tr, R = tvm.run_prover(GOLDEN, pre, len(ref) - len(pre) + 50)
    att = execute(pre, ref, tr, R, None)
    attempts.append(att)
    first = attempts[0]
    report = {"verdict": first["verdict"], "first_wrong_step": first["first_wrong_step"], "faulty_sublayer": first["faulty_sublayer"],
              "trace_len": first["trace_len"], "hash_comparisons": first["hash_comparisons"],
              "prover_output": first["output"], "reference_output": llama_front.decode_output([TOKS[x] for x in ref]).strip()}
    if len(attempts) > 1:
        report["after_retry"] = {"verdict": att["verdict"], "receipt": att["receipt"][:16]}
    expl = llama_front.explain(report)
    return {"code": code, "attempts": attempts, "decisions": decisions,
            "delivered": att["output"] if att["verdict"] == "PASS" else None, "receipt": att["receipt"],
            "explanation": expl, "sublayers": NAMES,
            "seconds": {"write": round(t_write, 1), "total": round(time.time() - t0, 1)}}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = "index.html" if self.path in ("/", "") else self.path.lstrip("/")
        f = os.path.join(WEB, os.path.basename(p))
        if os.path.exists(f):
            self._send(200, open(f, "rb").read(), "text/html; charset=utf-8" if f.endswith(".html") else "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self._send(200, json.dumps(run(body)).encode(), "application/json")
        except Exception as e:
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(e)[-400:]}).encode(), "application/json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"http://127.0.0.1:{port}  backend={llama_front.backend()}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
