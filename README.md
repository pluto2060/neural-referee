# Neural Computer Referee

**Neural Computer = Compiled Layers + Neural State + Program Weights + Verifier.**
Recent work compiles real computers into transformer weights (Percepta transformer-vm, QwenPU).
They show the machine works by comparing final output with a reference. This project is the missing
fourth term: a referee that, when such a machine goes wrong, finds **the first wrong step and the
sublayer that caused it**, without re-running the program.

Built for the Meta Global AI Developer Hackathon (Oct 26 - Nov 4, 2026), Indie track. All code in this
repository is new work from 2026-11-03.

## Hypothesis H-R
When a computer that runs inside transformer weights produces a wrong result, its own execution record
is enough to find the first wrong step and the faulty sublayer.
Pass rule, fixed before running: at least 9 of 10 harmful injected faults located exactly
(step equal to a full linear comparison, sublayer equal to where the corrupted weight lives).

## How it works
1. **Commit.** The prover (the neural computer, possibly faulty) runs a program and publishes a SHA-256
   hash chain over its token trace plus its residual activations after every sublayer.
   The reference executor (a plain WebAssembly interpreter) publishes a hash chain over its trace.
2. **Bisect.** The referee compares chain values at O(log n) indices to find the first step k where the
   traces differ (about 10-14 comparisons for 1,000-9,000-step traces).
3. **Recheck one step.** At the position that produced step k, the referee recomputes every sublayer
   (embedding, attention and FFN of each layer, output head) with the golden compiled weights from the
   prover's committed inputs. The first sublayer whose committed output differs is the faulty one.
4. **Earlier-position scan (fallback).** If every sublayer at step k is consistent, the fault changed an
   earlier position whose state later reached step k through attention. The referee checks every
   sublayer at all earlier positions in one parallel pass per sublayer and reports the earliest
   inconsistent position and sublayer.

## Results (CPU only, no GPU)
| Run | Harmful faults | Located exactly | Found at step k | Needed scan | Benign faults (no output change) | Referee time / prover time |
|---|---|---|---|---|---|---|
| `results/h_r_stratified.json` (seed 2, faults spread evenly over 16 sublayers) | 20 | 20 | 18 | 2 | 27 | 0.149 |
| `results/h_r_stratified_seed3.json` (seed 3, same setup) | 20 | 20 | 15 | 5 | 20 | 0.194 |
| `results/h_r.json` (seed 1, faults weighted by tensor size, first version) | 10 | 10 | - | - | 13 | - |

No false alarms: the golden (unmodified) model passes on `hello`, `addition`, `fibonacci` and `collatz`
(1,034 to 44,332 steps), see `results/run_seed3_and_golden.log`.

Fault kinds: sign flip, exponent bit flip, high mantissa bit flip, zeroed weight, on a random nonzero
weight. Programs: `hello`, `addition`, `fibonacci` from transformer-vm (1,034-9,037 steps).
More than half of random single-weight faults changed nothing in the output; the referee reports PASS
for those, because the trace matches the reference.

## Honest limits
- The referee needs the golden compiled weights. For compiled (not trained) layers this is cheap: the
  compiler output is deterministic and can be rebuilt by anyone.
- Recomputing one attention sublayer still reads keys at all earlier positions (O(n) dot products, no
  autoregressive generation). Measured cost is 15-19% of the prover's run time, not 1%.
- The first seed-1 run sampled faults by tensor size, so 6 of 10 landed in the output head; the
  stratified run is the one to cite.
- Tested only on transformer-vm (7 layers, d_model 38). QwenPU needs a 40 GB GPU and was not run.

## Run it
```
# 1. transformer-vm (Apache-2.0) in a sibling folder, with a clang that has the wasm32 target
git clone https://github.com/Percepta-Core/transformer-vm ../_ext/transformer-vm
cd ../_ext/transformer-vm && uv sync && CLANG_PATH=<clang with wasm32> uv run wasm-run   # builds model.bin
# 2. this repo, using that environment
export TVM_ROOT=../_ext/transformer-vm
PYTHONPATH=. $TVM_ROOT/.venv/bin/python -m pytest tests -q
PYTHONPATH=. $TVM_ROOT/.venv/bin/python experiments/h_r.py --n 20 --seed 2 --stratify
```
No GPU or API key is needed for the referee.

## Layout
- `referee/tvm.py` - prover run with activation commitments; golden recompute of one sublayer
- `referee/protocol.py` - hash chain, bisection, one-step recheck, earlier-position scan
- `referee/faults.py` - weight fault injection with the true sublayer recorded
- `experiments/h_r.py` - the H-R test
