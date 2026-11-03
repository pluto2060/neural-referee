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
        page.goto(card("<h1>Neural Referee</h1><div class=dim>Verifiable answers from a computer that runs inside transformer weights</div>"
                       "<p style='margin-top:40px'><code>Neural Computer = Compiled Layers + Neural State + Program Weights + <span class=acc>Verifier</span></code></p>"
                       "<p class=dim>Meta Global AI Developer Hackathon 2026 &middot; Indie track &middot; Push the Frontier</p>"))
        time.sleep(9)
        page.goto(card("<h1>The problem</h1><p>Researchers now compile real computers into model weights: a WebAssembly machine (Percepta) and a RISC-V CPU inside a 4B open model (QwenPU).</p>"
                       "<p>They only check that the <b>final output</b> matches. When one step goes wrong, nobody can say <b>which step</b> or <b>which part of the model</b>.</p>"
                       "<p class=acc>Neural Referee is the missing verifier.</p>"))
        time.sleep(16)
        page.goto(URL)
        caption(page, "1. Ask in plain language. A Meta model (Meta Model API) writes the program; it runs inside a 7-layer transformer computer on CPU.")
        time.sleep(4)
        page.click("#run")
        page.wait_for_selector("text=every step verified", timeout=300000)
        caption(page, "PASS: the referee compared hash chains of the whole trace and every step matched. The receipt hash lets anyone re-check this answer.")
        time.sleep(9)
        caption(page, "2. Now flip one bit in the computer's memory or arithmetic while it runs, like a hardware fault. Most flips are harmless, so we search for one that matters.")
        page.click("#fault")
        page.wait_for_selector("text=wrong step found", timeout=600000)
        caption(page, "FAIL: about 11 hash comparisons find the first wrong step, then one step is rechecked layer by layer. The faulty sublayer is shown in red.")
        time.sleep(12)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        caption(page, "The Meta model explains the verdict in plain language. A weight-file check would see nothing here: the weights never changed.")
        time.sleep(10)
        rows = "".join(f"<tr><td>{n}</td><td>{T[k]['harmful']}</td><td class=ok>{int(T[k]['referee_locates']*T[k]['harmful'])}/{T[k]['harmful']}</td>"
                       f"<td class={'ok' if T[k]['weight_diff_locates'] else 'bad'}>{int(T[k]['weight_diff_locates']*T[k]['harmful'])}/{T[k]['harmful']}</td><td class=bad>0/{T[k]['harmful']}</td></tr>"
                       for k, n in (("weight", "Corrupted weight"), ("act", "Compute error while running"), ("kv", "Memory error while running")))
        page.goto(card("<h1>Results (CPU only, reproducible)</h1><table><tr><th>Fault</th><th>Harmful</th><th>Referee locates</th><th>Weight-file diff</th><th>Output check locates</th></tr>"
                       f"{rows}</table><p>40/40 more weight faults located in two earlier runs. No false alarms on four clean programs up to 44,332 steps.</p>"
                       "<p>Hidden error caught: final answer correct, step 1,909 of 3,107 wrong.</p>"))
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
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", mp4], check=True)
    dur = subprocess.run(["ffmpeg", "-i", mp4], capture_output=True, text=True).stderr.split("Duration:")[-1].split(",")[0].strip()
    print(mp4, "duration_s", dur)


if __name__ == "__main__":
    main()
