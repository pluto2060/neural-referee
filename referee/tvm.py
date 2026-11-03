"""Thin layer over Percepta transformer-vm: load weights, run the prover, recompute one sublayer.

Sublayer order (index into the residual record R[pos, s, :]):
  s=0            residual after token embedding + position encoding
  s=2l+1         residual after attention of layer l
  s=2l+2         residual after FFN of layer l
The head (argmax over vocab) reads R[pos, 2L, :].
"""
import os
import sys

TVM_ROOT = os.environ.get("TVM_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "_ext", "transformer-vm"))
sys.path.insert(0, TVM_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformer_vm.attention import HullKVCache  # noqa: E402
from transformer_vm.model.transformer import add_position_encoding  # noqa: E402
from transformer_vm.model.weights import load_weights  # noqa: E402

DATA = os.path.join(TVM_ROOT, "transformer_vm", "data")
MODEL_BIN = os.path.join(TVM_ROOT, "model.bin")


def load(model_path=MODEL_BIN):
    model, all_tokens, tok_to_idx = load_weights(model_path)
    model.eval()
    return model, all_tokens, tok_to_idx


def program(name, tok_to_idx):
    """Return (prefix ids, reference ids) for a compiled example program."""
    with open(os.path.join(DATA, f"{name}.txt")) as f:
        prefix = [tok_to_idx[t] for t in f.read().split()]
    with open(os.path.join(DATA, f"{name}_ref.txt")) as f:
        ref = [tok_to_idx[t] for t in f.read().split()]
    return prefix, ref


def sublayer_names(n_layers):
    names = ["embed"]
    for layer in range(n_layers):
        names += [f"L{layer}.attn", f"L{layer}.ffn"]
    return names + ["head"]


def _new_cache(model, n_layers):
    cache = HullKVCache(n_layers, model.attn[0].num_heads)
    if hasattr(model, "head_tiebreak"):
        for li in range(n_layers):
            for h in range(model.attn[0].num_heads):
                if model.head_tiebreak[li][h]:
                    cache.set_tiebreak(li, h, True)
    return cache


def _flip_bit(t, idx, bit):
    """Return a copy of float64 tensor t with one bit of element idx flipped."""
    t = t.clone()
    b = t[idx : idx + 1].view(torch.int64)
    b ^= torch.tensor([1], dtype=torch.int64) << bit
    return t


@torch.no_grad()
def run_prover(model, prefix, max_new, runtime_fault=None):
    """Autoregressive run that also commits every residual it computes.

    Same arithmetic as VanillaTransformer.generate_with_cache, plus a record R of the
    residual after each sublayer at each position (the prover's activation commitment).

    runtime_fault: one fault dict or a list of them (weights untouched, simulates hardware errors during
    execution):
      {"type": "act", "pos": p, "sublayer": s, "dim": d, "bit": b}
          flip one bit of the residual right after sublayer s at position p (compute error)
      {"type": "kv", "pos": p, "layer": l, "which": "k"|"v", "dim": d, "bit": b}
          flip one bit of the key or value written to attention memory (memory error)
    The prover commits what it actually computed, so a corrupted residual is committed as is.
    """
    faults = runtime_fault if isinstance(runtime_fault, list) else ([runtime_fault] if runtime_fault else [])
    acts = {(f["pos"], f["sublayer"]): f for f in faults if f.get("type") == "act"}
    kvs = {(f["pos"], f["layer"]): f for f in faults if f.get("type") == "kv"}

    def act(x, pos, s):
        f = acts.get((pos, s))
        return _flip_bit(x, f["dim"], f["bit"]) if f else x

    n_layers = len(model.attn)
    cache = _new_cache(model, n_layers)
    ids = list(prefix)
    rec = []
    pos = 0
    while pos < len(ids) and pos < len(prefix) + max_new:
        x = model.tok.weight[ids[pos]].clone()
        add_position_encoding(x, pos)
        x = act(x, pos, 0)
        row = [x.clone()]
        for li, (attn, ff_in, ff_out) in enumerate(zip(model.attn, model.ff_in, model.ff_out)):
            q, k, v = (attn.in_proj_weight @ x).chunk(3, dim=-1)
            f = kvs.get((pos, li))
            if f:
                if f["which"] == "k":
                    k = _flip_bit(k, f["dim"], f["bit"])
                else:
                    v = _flip_bit(v, f["dim"], f["bit"])
            out = cache.layer_step(li, k.contiguous(), q.contiguous(), v.contiguous())
            x = act(x + attn.out_proj(out), pos, 2 * li + 1)
            row.append(x.clone())
            gate, val = ff_in(x).chunk(2, dim=-1)
            x = act(x + ff_out(F.relu(gate) * val), pos, 2 * li + 2)
            row.append(x.clone())
        rec.append(torch.stack(row))
        if pos + 1 == len(ids):
            nxt = model.head(x).argmax().item()
            ids.append(nxt)
            if nxt == model.stop_token_id:
                break  # the stop token is never fed back, so rec index == position still holds
        pos += 1
    return ids, torch.stack(rec)


@torch.no_grad()
def recompute_sublayer(golden, R, tokens, p, s):
    """Recompute sublayer s at position p with golden weights from the prover's committed inputs.

    Returns the golden value of R[p, s] (or, for the head, the golden next-token id).
    Attention needs keys/values at positions 0..p, rebuilt from committed inputs R[0..p, s-1].
    """
    n_layers = len(golden.attn)
    if s == 0:
        x = golden.tok.weight[tokens[p]].clone()
        add_position_encoding(x, p)
        return x
    if s == 2 * n_layers + 1:  # head
        return golden.head(R[p, 2 * n_layers]).argmax().item()
    layer, is_ffn = (s - 1) // 2, (s - 1) % 2 == 1
    x_in = R[p, s - 1]
    if is_ffn:
        gate, val = golden.ff_in[layer](x_in).chunk(2, dim=-1)
        return x_in + golden.ff_out[layer](F.relu(gate) * val)
    attn = golden.attn[layer]
    cache = _new_cache(golden, n_layers)
    out = None
    for i in range(p + 1):
        q, k, v = (attn.in_proj_weight @ R[i, s - 1]).chunk(3, dim=-1)
        out = cache.layer_step(layer, k, q, v)
    return x_in + attn.out_proj(out)


@torch.no_grad()
def recompute_sublayer_all(golden, R, tokens, p, s):
    """Golden value of sublayer s at every position 0..p from committed inputs (no autoregressive
    generation). Uses exactly the same per-vector operations as the prover: batched matmuls round
    differently and caused a false hit in the first baseline run. Not used for the head."""
    n_layers = len(golden.attn)
    if s == 0:
        out = []
        for i in range(p + 1):
            x = golden.tok.weight[tokens[i]].clone()
            add_position_encoding(x, i)
            out.append(x)
        return torch.stack(out)
    layer, is_ffn = (s - 1) // 2, (s - 1) % 2 == 1
    if is_ffn:
        out = []
        for i in range(p + 1):
            x_in = R[i, s - 1]
            gate, val = golden.ff_in[layer](x_in).chunk(2, dim=-1)
            out.append(x_in + golden.ff_out[layer](F.relu(gate) * val))
        return torch.stack(out)
    attn = golden.attn[layer]
    cache = _new_cache(golden, n_layers)
    out = []
    for i in range(p + 1):
        x_in = R[i, s - 1]
        q, k, v = (attn.in_proj_weight @ x_in).chunk(3, dim=-1)
        o = cache.layer_step(layer, k.contiguous(), q.contiguous(), v.contiguous())
        out.append(x_in + attn.out_proj(o))
    return torch.stack(out)
