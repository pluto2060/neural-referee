"""Referee protocol.

1. Commit: the prover publishes a hash chain over its token trace (h_i covers tokens 0..i).
   The reference executor (plain WASM interpreter, wasm-reference) publishes the same for its trace.
2. Bisect: compare chain values at O(log n) indices to find the first step k where traces differ.
3. Recheck one step: at position p=k-1, recompute each sublayer with golden weights from the
   prover's committed inputs. The first sublayer whose committed output differs is the faulty one.
No program is re-run: the referee does O(log n) hash comparisons and one step of sublayer math.
"""
import hashlib
import math

import torch

from referee import tvm


def hash_chain(tokens):
    h = b"\x00" * 32
    out = []
    for t in tokens:
        h = hashlib.sha256(h + int(t).to_bytes(4, "little")).digest()
        out.append(h)
    return out


def bisect_first_divergence(chain_a, chain_b):
    """Smallest i with chain_a[i] != chain_b[i] (or the shorter length if one is a prefix).
    Returns (i, number_of_comparisons)."""
    n = min(len(chain_a), len(chain_b))
    lo, hi, comps = 0, n, 0  # invariant: all indices < lo agree; answer in [lo, hi]
    while lo < hi:
        mid = (lo + hi) // 2
        comps += 1
        if chain_a[mid] == chain_b[mid]:
            lo = mid + 1
        else:
            hi = mid
    return lo, comps


def linear_first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def locate_sublayer(golden, R, tokens, k):
    """Recheck step k (computed at position p=k-1). Returns list of per-sublayer verdicts and the
    first faulty sublayer name (None if every sublayer matches)."""
    p = k - 1
    n_layers = len(golden.attn)
    names = tvm.sublayer_names(n_layers)
    verdicts, first = [], None
    for s, name in enumerate(names):
        g = tvm.recompute_sublayer(golden, R, tokens, p, s)
        if name == "head":
            ok = g == tokens[k]
            diff = 0.0 if ok else float("inf")
        else:
            diff = (g - R[p, s]).abs().max().item()
            ok = torch.equal(g, R[p, s])
        verdicts.append({"sublayer": name, "match": bool(ok), "max_abs_diff": diff})
        if not ok and first is None:
            first = name
    return verdicts, first


def scan_earlier(golden, R, tokens, k):
    """Fallback when step k itself is consistent: the fault changed an earlier position whose
    residual later reached step k through attention. Check every sublayer at positions 0..k-1
    in one parallel pass per sublayer and return the earliest inconsistent (position, sublayer)."""
    p = k - 1
    names = tvm.sublayer_names(len(golden.attn))
    best = None
    for s, name in enumerate(names[:-1]):  # head is already checked at step k
        g = tvm.recompute_sublayer_all(golden, R, tokens, p, s)
        bad = (g != R[: p + 1, s]).any(dim=-1).nonzero()
        if len(bad):
            pos = int(bad[0])
            if best is None or pos < best[0]:
                best = (pos, name)
    return best


def referee(golden, prover_tokens, R, ref_tokens):
    """Full verdict for one prover run against the reference trace."""
    k, comps = bisect_first_divergence(hash_chain(prover_tokens), hash_chain(ref_tokens))
    n = len(ref_tokens)
    if k >= min(len(prover_tokens), n) and len(prover_tokens) == n:
        return {"verdict": "PASS", "first_wrong_step": None, "hash_comparisons": comps, "trace_len": n}
    verdicts, first = locate_sublayer(golden, R, prover_tokens, k) if k < len(prover_tokens) else ([], None)
    stage, origin_pos = "step", k - 1
    if first is None and k < len(prover_tokens):
        hit = scan_earlier(golden, R, prover_tokens, k)
        if hit:
            stage, origin_pos, first = "scan", hit[0], hit[1]
    return {
        "stage": stage,
        "fault_origin_position": origin_pos,
        "verdict": "FAIL",
        "first_wrong_step": k,
        "prover_token": prover_tokens[k] if k < len(prover_tokens) else None,
        "reference_token": ref_tokens[k] if k < n else None,
        "faulty_sublayer": first,
        "sublayer_checks": verdicts,
        "hash_comparisons": comps,
        "log2_trace_len": round(math.log2(max(n, 2)), 1),
        "steps_reexecuted": 1 if stage == "step" else None,
        "trace_len": n,
    }
