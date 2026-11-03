# 3-minute demo script (draft; follows the H-R pass rule)

| Time | Screen | Narration |
|---|---|---|
| 0:00-0:20 | Formula slide: Neural Computer = Compiled Layers + Neural State + Program Weights + Verifier | Computers are now being compiled into model weights. The missing piece is the verifier. |
| 0:20-0:45 | `wasm-run` on fibonacci: PASS, steps/s on CPU | This 7-layer transformer is a WebAssembly computer. Its weights were calculated, not trained. |
| 0:45-1:20 | Flip one weight bit; run again; output is wrong | One corrupted weight. The program now prints the wrong answer. Where did it break? |
| 1:20-2:10 | Referee output: hash comparisons (about 13), first wrong step, per-sublayer check table, faulty sublayer | The referee compares about 13 hashes, not 9,000 steps, then rechecks one step layer by layer and names the faulty sublayer. |
| 2:10-2:40 | Results table: 20 of 20 located, 18 at the wrong step, 2 by earlier-position scan, 27 benign faults | Fixed pass rule: 9 of 10. Result: 20 of 20 with faults spread over every sublayer. |
| 2:40-3:00 | Limits + next: Meta model as the natural-language front end and verdict explainer | Next: Llama writes the program and explains the verdict. |

Open item: the Meta model step (Meta Model API or open-weights Llama) is not built yet.
