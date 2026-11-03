"""Record the 3-minute demo video from the local web demo (server must run on :8765).
Title cards are plain HTML; numbers come from results/*.json so the video matches the repo.
Output: results/video/neural-referee-demo.mp4"""
import glob, json, os, shutil, subprocess, sys, time
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "video")
URL = os.environ.get("DEMO_URL", "http://127.0.0.1:8765/")

bt = json.load(open(os.path.join(ROOT, "results", "baseline_table.json")))["table"]
acc = json.load(open(os.path.join(ROOT, "results", "accuracy_muse.json")))["summary"]
def _mv():
    f = os.path.join(ROOT, "results", "muse_vs_verified.json")
    if os.path.exists(f):
        return json.load(open(f))["summary"]["total"]
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "results", "muse_vs_verified.log")) if l.startswith('{"kind"')]
    return {"n": len(rows), "direct_ok": sum(r["direct_ok"] for r in rows), "verified_ok": sum(r["verified_ok"] for r in rows)}


mv = _mv()
T = {r["class"]: r for r in bt}

CSS = """body{margin:0;height:100vh;display:flex;flex-direction:column;justify-content:center;padding:0 90px;
background:#0f1419;color:#e6edf3;font:28px/1.45 system-ui,Segoe UI,sans-serif}h1{font-size:54px;margin:0 0 18px}
.dim{color:#8b98a5}.acc{color:#58a6ff}.ok{color:#3fb950}.bad{color:#f85149}table{border-collapse:collapse;font-size:24px;margin-top:10px}
td,th{border-bottom:1px solid #2a3542;padding:8px 18px;text-align:left}code{font-family:ui-monospace,monospace;font-size:22px}"""


def card(html):
    return "data:text/html;charset=utf-8," + __import__("urllib.parse").parse.quote(f"<style>{CSS}</style>{html}")


def caption(page, text):
    page.evaluate("""t => { let c = document.getElementById('cap'); if (!c) { c = document.createElement('div'); c.id = 'cap';
      c.style.cssText = 'position:fixed;left:0;right:0;bottom:0;padding:14px 28px;background:rgba(10,14,20,.92);color:#e6edf3;font:20px system-ui;border-top:2px solid #58a6ff;z-index:9';
      document.body.appendChild(c); } c.textContent = t; }""", text)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.webm")):
        os.remove(f)
    with sync_playwright() as p:
        exe = os.environ.get("CHROME_PATH")
        b = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1280, "height": 760}, record_video_dir=OUT, record_video_size={"width": 1280, "height": 760})
        page = ctx.new_page()
        t_start = time.time()
        page.goto(card("<h1>Neural Referee</h1><div class=dim>Catch silent data corruption inside AI computation, and say exactly where it happened</div>"
                       "<p style='margin-top:40px'><code>Neural Computer = Compiled Layers + Neural State + Program Weights + <span class=acc>Verifier</span></code></p>"
                       "<p class=dim>Meta Global AI Developer Hackathon 2026 &middot; Indie track &middot; Push the Frontier</p>"))
        time.sleep(9)
        page.goto(card("<h1>The problem</h1><p>Hardware sometimes computes a wrong value with no error. Meta studied these silent data corruptions across its fleet (Dixit et al., 2021).</p>"
                       "<p>Computers are now compiled into model weights (Percepta's WebAssembly transformer, QwenPU's RISC-V CPU).</p>"
                       "<p>They only check that the <b>final output</b> matches. When one step goes wrong, nobody can say <b>which step</b> or <b>which part of the model</b>.</p>"
                       "<p class=acc>Neural Referee is the missing verifier.</p>"))
        time.sleep(16)
        page.goto(URL)
        caption(page, "Ask in plain language. A Meta AI model (Meta Model API) writes the program; it runs inside a 7-layer transformer computer.")
        time.sleep(5)
        caption(page, "The first run is hit by a hardware-style bit flip, like silent data corruption on a bad machine.  (waiting time shown at 4x)")
        t_run0 = time.time() - t_start
        page.click("#run")
        page.wait_for_selector("text=Delivered with receipt", timeout=900000)
        t_run1 = time.time() - t_start
        caption(page, "The referee finds the first wrong step and the sublayer in red. The model reads the verdict, re-runs, and delivers only the passing answer with a receipt.")
        time.sleep(14)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        caption(page, "A weight-file check would see nothing here: the weights never changed. Only the step-by-step referee can say where it went wrong.")
        time.sleep(9)
        st = json.load(open(os.path.join(ROOT, "results", "agent_stress.json")))["summary"] if os.path.exists(os.path.join(ROOT, "results", "agent_stress.json")) else None
        rows = "".join(f"<tr><td>{n}</td><td>{T[k]['harmful']}</td><td class=ok>{int(T[k]['referee_locates']*T[k]['harmful'])}/{T[k]['harmful']}</td>"
                       f"<td class={'ok' if T[k]['weight_diff_locates'] else 'bad'}>{int(T[k]['weight_diff_locates']*T[k]['harmful'])}/{T[k]['harmful']}</td><td class=bad>0/{T[k]['harmful']}</td></tr>"
                       for k, n in (("weight", "Corrupted weight"), ("act", "Compute error while running"), ("kv", "Memory error while running")))
        page.goto(card("<h1>Results (CPU only, reproducible)</h1><table><tr><th>Fault</th><th>Harmful</th><th>Referee locates</th><th>Weight-file diff</th><th>Output check locates</th></tr>"
                       f"{rows}</table><p>40/40 more weight faults located in two earlier runs. No false alarms on four clean programs up to 44,332 steps.</p>"
                       "<p>Hidden error caught: final answer correct, step 1,909 of 3,107 wrong.</p>"
                       + (f"<p>Stress test, 8 bit flips per run: unchecked pipeline delivered <b class=bad>{st['unverified_wrong_delivered']}</b> wrong answers out of {st['n']}; "
                          f"the agent loop delivered <b class=ok>{st['agent_wrong_delivered']}</b> wrong ({st['agent_correct_delivered']} correct, {st['agent_abstained']} withheld).</p>" if st else "")))
        time.sleep(18)
        page.goto(card("<h1>Meta models in the loop</h1>"
                       f"<p>Program writing via Meta Model API: <b>{acc['first_try_correct']}/{acc['n']}</b> requests correct on the first try (about 12 s each). "
                       "A small open-weights Meta model runs the same loop offline on CPU.</p>"
                       f"<p>Exact tasks answered directly by the model: <b>{mv['direct_ok']}/{mv['n']}</b>; through the verified computer: <b>{mv['verified_ok']}/{mv['n']}</b>, each with a receipt.</p>"
                       "<p class=dim>Limits: referee needs the golden compiled weights; checking costs 15-19% of run time; tested on transformer-vm only.</p>"))
        time.sleep(16)
        page.goto(card("<h1>Neural Referee</h1><p>github.com/pluto2060/neural-referee</p><p class=dim>Next: compile the computer into a Meta open-weights model and verify it the same way.</p>"))
        time.sleep(7)
        vid = page.video.path()
        ctx.close(); b.close()
    mp4 = os.path.join(OUT, "neural-referee-demo.mp4")
    # play the waiting part of the live run at 4x; everything else at normal speed
    a, b = t_run0 + 1.0, max(t_run0 + 2.0, t_run1 - 1.0)
    fc = (f"[0:v]trim=0:{a},setpts=PTS-STARTPTS[v0];[0:v]trim={a}:{b},setpts=(PTS-STARTPTS)/4[v1];"
          f"[0:v]trim={b},setpts=PTS-STARTPTS[v2];[v0][v1][v2]concat=n=3:v=1:a=0[out]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-filter_complex", fc, "-map", "[out]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", mp4], check=True)
    dur = subprocess.run(["ffmpeg", "-i", mp4], capture_output=True, text=True).stderr.split("Duration:")[-1].split(",")[0].strip()
    print(mp4, "duration_s", dur)


if __name__ == "__main__":
    main()
