"""Local web demo: ask -> Muse writes a program -> it runs inside the transformer computer ->
optional fault -> referee verdict -> Muse explains. No framework; python demo/server.py, open :8765."""
import json, os, random, sys, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from referee import llama_front, protocol, tvm  # noqa: E402

GOLDEN, TOKS, T2I = tvm.load()
NAMES = tvm.sublayer_names(len(GOLDEN.attn))
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
CACHE = {}


def build(request, inp):
    key = (request, inp)
    if key not in CACHE:
        code, log = llama_front.write_program(request + " The input string is passed as `input`.")
        if code is None:
            raise RuntimeError("program did not compile: " + json.dumps(log)[-300:])
        txt, ref_txt = llama_front.build(code, inp, f"web{abs(hash(key)) % 10**8}")
        pre = [T2I[t] for t in open(txt).read().split()]
        ref = [T2I[t] for t in open(ref_txt).read().split()]
        CACHE[key] = (code, pre, ref)
    return CACHE[key]


def run(body):
    t0 = time.time()
    code, pre, ref = build(body["request"], body.get("input", ""))
    t_write = time.time() - t0
    fault = None
    if body.get("fault"):
        rng = random.Random(int(body.get("seed", 1)))
        L = len(GOLDEN.attn)
        for _ in range(200):  # look for a fault that actually changes the trace
            rf = {"type": rng.choice(["act", "kv"]), "pos": rng.randrange(len(pre), len(ref) - 1),
                  "dim": rng.randrange(GOLDEN.tok.weight.shape[1]), "bit": rng.choice(range(52, 62))}
            if rf["type"] == "act":
                rf["sublayer"] = rng.randrange(2 * L + 1)
            else:
                rf["layer"], rf["which"] = rng.randrange(L), rng.choice("kv")
            tr, R = tvm.run_prover(GOLDEN, pre, len(ref) - len(pre) + 50, runtime_fault=rf)
            if tr != ref:
                fault = rf
                break
    if fault is None:
        tr, R = tvm.run_prover(GOLDEN, pre, len(ref) - len(pre) + 50)
    t_run = time.time() - t0 - t_write
    v = protocol.referee(GOLDEN, tr, R, ref)
    v["prover_output"] = llama_front.decode_output([TOKS[x] for x in tr])
    v["reference_output"] = llama_front.decode_output([TOKS[x] for x in ref])
    expl = llama_front.explain(v) if body.get("explain", True) else ""
    return {"code": code, "steps": len(tr), "program_tokens": len(pre), "fault": fault,
            "verdict": v["verdict"], "first_wrong_step": v.get("first_wrong_step"),
            "faulty_sublayer": v.get("faulty_sublayer"), "origin_position": v.get("fault_origin_position"),
            "stage": v.get("stage"), "hash_comparisons": v["hash_comparisons"],
            "receipt": protocol.hash_chain(tr)[-1].hex(), "sublayers": NAMES,
            "output": v["prover_output"], "reference_output": v["reference_output"],
            "explanation": expl, "backend": llama_front.backend(),
            "seconds": {"write": round(t_write, 1), "run": round(t_run, 1), "total": round(time.time() - t0, 1)}}


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
