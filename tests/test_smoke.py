"""Golden prover passes; a large fault is caught at the right sublayer. Run with the transformer-vm venv."""
import random

from referee import faults, protocol, tvm


def test_golden_pass_and_fault_located():
    golden, _, t2i = tvm.load()
    prefix, ref = tvm.program("hello", t2i)
    toks, R = tvm.run_prover(golden, prefix, len(ref))
    assert toks == ref
    assert protocol.referee(golden, toks, R, ref)["verdict"] == "PASS"
    rng = random.Random(0)
    for _ in range(30):
        bad, info = faults.inject(golden, rng, kind="exponent")
        toks, R = tvm.run_prover(bad, prefix, len(ref) + 50)
        if toks != ref:
            v = protocol.referee(golden, toks, R, ref)
            assert v["verdict"] == "FAIL"
            assert v["first_wrong_step"] == protocol.linear_first_divergence(toks, ref)
            assert v["faulty_sublayer"] == info["true_sublayer"], (v, info)
            return
    raise AssertionError("no harmful fault in 30 tries")
