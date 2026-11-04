"""Record the demo video with the new slide-style title cards (white, giant bold
headline, Meta-blue highlight block). Same structure as record_video.py; only the
title/section cards and the caption bar changed. Live demo + how-it-works page are
unchanged (they are product screens, not slides).
Output: results/video/neural-referee-demo-slide.mp4"""
import glob, json, os, subprocess, time
from playwright.sync_api import sync_playwright

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "video")
URL = os.environ.get("DEMO_URL", "http://127.0.0.1:8765/")
CHROME = (os.environ.get("CHROME_PATH")
          or (sorted(glob.glob("/home/desk-1/.cache/ms-playwright/chromium-*/chrome-linux64/chrome")) or [None])[-1])

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

# ---- new slide language: white, black, Meta blue #0866FF, flat, sharp corners ----
CSS = """*{box-sizing:border-box}
body{margin:0;height:100vh;display:flex;flex-direction:column;justify-content:center;padding:0 88px;
background:#fff;color:#000;font:24px/1.45 "Helvetica Neue","Optimistic Text","Segoe UI",Roboto,Helvetica,Arial,sans-serif;letter-spacing:-0.012em}
.eb{font-size:15px;font-weight:800;text-transform:uppercase;letter-spacing:.24em;margin:0 0 30px}
h1{font-size:64px;line-height:.98;letter-spacing:-0.038em;margin:0 0 28px;font-weight:800}
.hl{display:inline-block;background:#0866FF;color:#fff;padding:.02em .12em .05em}
p{font-size:24px;line-height:1.5;margin:0 0 16px;max-width:66ch}
.dim{color:#666}.acc{color:#0866FF;font-weight:800}
code{font-family:ui-monospace,Menlo,monospace;font-size:19px;background:#EDF2FF;color:#0846B0;padding:6px 12px;border:1px solid #C9D8FF}
table{border-collapse:collapse;font-size:21px;margin-top:6px;width:100%}
th{color:#666;font-size:13px;text-transform:uppercase;letter-spacing:.1em;text-align:left;border-bottom:3px solid #000;padding:8px 14px}
td{border-bottom:1px solid #E7E7E7;padding:9px 14px;text-align:left}
td.a{background:#EDF2FF;font-weight:700}"""


def card(html):
    return "data:text/html;charset=utf-8," + __import__("urllib.parse").parse.quote(f"<style>{CSS}</style>{html}")


def caption(page, text):
    page.evaluate("""t => { let c = document.getElementById('cap'); if (!c) { c = document.createElement('div'); c.id = 'cap';
      c.style.cssText = 'position:fixed;left:0;right:0;bottom:0;padding:16px 30px;background:#FFFFFF;color:#000;font:800 20px system-ui;border-top:4px solid #0866FF;z-index:9';
      document.body.appendChild(c); } c.style.zoom = 1 / parseFloat(document.body.style.zoom || 1); c.textContent = t; }""", text)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "*.webm")):
        os.remove(f)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"]) if CHROME else p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1280, "height": 760}, record_video_dir=OUT, record_video_size={"width": 1280, "height": 760})
        page = ctx.new_page()
        t_start = time.time()
        page.goto(card('<div class=eb>Meta Global AI Developer Hackathon 2026 &middot; Indie track &middot; Push the Frontier</div>'
                       '<h1>Neural <span class=hl>Referee</span></h1>'
                       '<p>Catch silent data corruption inside AI computation &mdash; and say exactly where it happened.</p>'
                       '<p class=dim style="margin-top:18px"><code>Neural Computer = Compiled Layers + Neural State + Program Weights + <span class=acc>Verifier</span></code></p>'))
        time.sleep(9)
        page.goto(card('<div class=eb>The problem</div><h1>A wrong value. <span class=hl>No error.</span></h1>'
                       '<p>Hardware sometimes computes a wrong value with no error. Meta studied these silent data corruptions across its fleet (Dixit et al., 2021).</p>'
                       '<p>Computers are now compiled into model weights (Percepta\'s WebAssembly transformer, QwenPU\'s RISC-V CPU).</p>'
                       '<p>They only check that the <b>final output</b> matches. When one step goes wrong, nobody can say <b>which step</b> or <b>which part of the model</b>.</p>'
                       '<p class=acc>Neural Referee is the missing verifier.</p>'))
        time.sleep(13)
        # how it works: the flow diagram, one stage at a time (product diagram, unchanged)
        page.goto("file://" + os.path.join(ROOT, "demo", "web", "how-it-works.html") + "?v=2")
        page.evaluate("document.body.style.zoom = 0.7")
        focus = lambda k: page.evaluate("""k => { const b = document.body; b.classList.remove('focus', 'sub3');
            document.querySelectorAll('.on').forEach(e => e.classList.remove('on')); if (!k) return;
            b.classList.add('focus'); document.querySelector(`.card[data-s="${k[0]}"]`).classList.add('on');
            if (k.length > 1) { b.classList.add('sub3'); document.querySelector(`.step[data-p="${k}"]`).classList.add('on'); } }""", k)
        for k, txt, sec in [
            ("", "How it works: four stages. A Meta AI model writes the program, a computer inside transformer weights runs it, the referee checks it, the model decides.", 5),
            ("1", "1. Write: the Meta AI model turns a plain-language request into a C program plus its own tests. Build errors go back to the model.", 5),
            ("2", "2. Run inside the weights: every generated token is one machine step, through 16 sublayers. A hardware bit flip corrupts one step; the weight file stays clean.", 7),
            ("3a", "3a. Hash-chain bisection: compare running hashes with a golden reference and halve the range. 3,107 steps need only 12 comparisons to find step 1,542.", 7),
            ("3b", "3b. Recheck just that one step with the golden weights, sublayer by sublayer: the first one that disagrees is the fault (L2.attn).", 6),
            ("3c", "3c. If that step is clean, the fault spread through attention memory: scan earlier positions to find where it started.", 5),
            ("4", "4. The model reads the verdict: retry, repair or give up. Only an answer that passes every step is delivered, with the final hash as its receipt.", 6)]:
            focus(k)
            caption(page, txt)
            time.sleep(sec)
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
        n_rt = T["act"]["harmful"] + T["kv"]["harmful"]
        n_all = n_rt + T["weight"]["harmful"]
        wd_rt = int(T["act"]["weight_diff_locates"] * T["act"]["harmful"] + T["kv"]["weight_diff_locates"] * T["kv"]["harmful"])
        ref_all = int(sum(T[k]["referee_locates"] * T[k]["harmful"] for k in T))
        ba = [("Fault located (step + sublayer)", "output check", f"0/{n_all}", f"{ref_all}/{n_all}"),
              ("Compute or memory error while running", "weight-file diff", f"{wd_rt}/{n_rt}", f"{n_rt}/{n_rt}")]
        if st:
            ba.append(("Wrong answers delivered, 8 bit flips per run", "unchecked pipeline", f"{st['unverified_wrong_delivered']}/{st['n']}", f"{st['agent_wrong_delivered']}/{st['n']}"))
        ba += [("Hidden error: right answer, wrong step", "output check", "missed", "caught (step 1,909 of 3,107)"),
               ("Cost of checking", "full re-run", "100% of run time", "15-19% of run time")]
        rows = "".join(f"<tr><td>{m}</td><td class=dim>{w}</td><td>{x}</td><td class=a>{y}</td></tr>" for m, w, x, y in ba)
        page.goto(card('<div class=eb>Before / after &middot; CPU only &middot; reproducible</div><h1>Before and <span class=hl>after</span></h1>'
                       '<table><tr><th>Measure</th><th>Before: method</th><th>Before</th><th>With Neural Referee</th></tr>'
                       f'{rows}</table><p class=dim style="font-size:21px;margin-top:14px">No false alarms on four clean programs up to 44,332 steps. '
                       + (f"Agent loop: {st['agent_correct_delivered']} correct with receipts, {st['agent_abstained']} withheld.</p>" if st else "</p>")))
        time.sleep(18)
        page.goto(card('<div class=eb>Meta models in the loop</div><h1>Muse writes. <span class=hl>Llama explains.</span></h1>'
                       f'<p>Program writing via Meta Model API: <b>{acc["first_try_correct"]}/{acc["n"]}</b> requests correct on the first try (about 12 s each). A small open-weights Meta model runs the same loop offline on CPU.</p>'
                       f'<p>Exact tasks answered directly by the model: <b>{mv["direct_ok"]}/{mv["n"]}</b>; through the verified computer: <b>{mv["verified_ok"]}/{mv["n"]}</b>, each with a receipt.</p>'
                       '<p class=dim>Limits: referee needs the golden compiled weights; checking costs 15-19% of run time; tested on transformer-vm only.</p>'))
        time.sleep(12)
        page.goto(card('<div class=eb>github.com/pluto2060/neural-referee</div><h1>Prove the <span class=hl>run.</span></h1>'
                       '<p>Neural Referee &mdash; verified computation for AI agents.</p>'
                       '<p class=dim>Next: compile the computer into a Meta open-weights model and verify it the same way.</p>'))
        time.sleep(7)
        vid = page.video.path()
        ctx.close(); b.close()
    mp4 = os.path.join(OUT, "neural-referee-demo-slide.mp4")
    a, b2 = t_run0 + 1.0, max(t_run0 + 2.0, t_run1 - 1.0)
    fc = (f"[0:v]trim=0:{a},setpts=PTS-STARTPTS[v0];[0:v]trim={a}:{b2},setpts=(PTS-STARTPTS)/4[v1];"
          f"[0:v]trim={b2},setpts=PTS-STARTPTS[v2];[v0][v1][v2]concat=n=3:v=1:a=0[out]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vid, "-filter_complex", fc, "-map", "[out]",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", mp4], check=True)
    dur = subprocess.run(["ffmpeg", "-i", mp4], capture_output=True, text=True).stderr.split("Duration:")[-1].split(",")[0].strip()
    print(mp4, "duration_s", dur, flush=True)


if __name__ == "__main__":
    main()
