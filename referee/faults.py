"""Fault injection into a copy of the golden model. Each fault records its true sublayer."""
import copy
import random

import torch

KINDS = ["sign", "exponent", "mantissa_high", "zero"]


def _params(model):
    """(sublayer name, tensor) for every weight tensor."""
    out = [("embed", model.tok.weight)]
    for li in range(len(model.attn)):
        out.append((f"L{li}.attn", model.attn[li].in_proj_weight))
        out.append((f"L{li}.attn", model.attn[li].out_proj.weight))
        out.append((f"L{li}.ffn", model.ff_in[li].weight))
        out.append((f"L{li}.ffn", model.ff_out[li].weight))
    out.append(("head", model.head.weight))
    return out


def _flip(value, kind, rng):
    bits = torch.tensor([value], dtype=torch.float64).view(torch.int64)
    if kind == "sign":
        bit = 63
    elif kind == "exponent":
        bit = rng.randint(52, 61)
    elif kind == "mantissa_high":
        bit = rng.randint(44, 51)
    else:
        return 0.0, None
    bits ^= torch.tensor([1], dtype=torch.int64) << bit
    return bits.view(torch.float64).item(), bit


@torch.no_grad()
def inject(golden, rng: random.Random, kind=None, stratify=False):
    """Return (faulty_model, fault_info). Picks a random nonzero weight.
    stratify=False: weighted by tensor size (large tensors such as the head dominate).
    stratify=True: first pick a sublayer uniformly, then a tensor and weight inside it."""
    model = copy.deepcopy(golden)
    params = _params(model)
    if stratify:
        nonzero = {n for n, t in params if int((t != 0).sum()) > 0}
        names = sorted(nonzero, key=lambda n: [x for x, _ in params].index(n))
        target = rng.choice(names)
        params = [(n, t) for n, t in params if n == target]
    sizes = [int((t != 0).sum()) for _, t in params]
    name, t = rng.choices(params, weights=sizes)[0]
    nz = (t != 0).nonzero()
    idx = tuple(nz[rng.randrange(len(nz))].tolist())
    kind = kind or rng.choice(KINDS)
    old = t[idx].item()
    new, bit = _flip(old, kind, rng)
    t[idx] = new
    return model, {"true_sublayer": name, "index": list(idx), "kind": kind, "bit": bit,
                   "old": old, "new": new}
