"""One-shot runner: start the demo server (with Meta key) if not already up, then
record the slide-style demo video, then stop the server."""
import glob, os, re, socket, subprocess, sys, time

ROOT = "/home/desk-1/Project/neural-referee"
env = dict(os.environ)
for p in ["/home/desk-1/.claude/env/shared.env"]:
    try:
        for line in open(p):
            m = re.match(r'\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', line)
            if m:
                k, v = m.group(1), m.group(2).strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                env.setdefault(k, v)
    except FileNotFoundError:
        pass
env["CHROME_PATH"] = sorted(glob.glob("/home/desk-1/.cache/ms-playwright/chromium-*/chrome-linux64/chrome"))[-1]
print("key present:", bool(env.get("META_MODEL_API_KEY")), flush=True)


def up():
    s = socket.socket(); s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", 8765)); s.close(); return True
    except OSError:
        return False


srv = None
if not up():
    log = open(os.path.join(ROOT, "results", "demo_server_slide.log"), "w")
    srv = subprocess.Popen([sys.executable, "demo/server.py"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        if up():
            break
        time.sleep(1)
    print("server up:", up(), flush=True)

try:
    r = subprocess.run([sys.executable, "demo/record_video_slide.py"], cwd=ROOT, env=env)
    print("recorder rc", r.returncode, flush=True)
finally:
    if srv:
        srv.terminate()
        try: srv.wait(timeout=10)
        except Exception: srv.kill()
print("DONE", flush=True)
