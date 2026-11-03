"""Baselines the referee is compared against."""
import torch


@torch.no_grad()
def weight_diff(golden, prover_model):
    """Compare the prover's weight files with the golden ones. Returns the first differing sublayer
    name, or None. Free and exact for weight corruption; blind to anything that happens at run time."""
    from referee.faults import _params
    for (name, g), (_, p) in zip(_params(golden), _params(prover_model)):
        if not torch.equal(g, p):
            return name
    return None


def output_only(prover_tokens, ref_tokens):
    """What transformer-vm and QwenPU check today: does the output match? No location."""
    return prover_tokens == ref_tokens
