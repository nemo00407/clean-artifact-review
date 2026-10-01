"""Regression guards for the headline evaluation claims (E1, E3).

These run the real harness on the generated dataset; skipped if the SISA
data has not been produced yet (harness/sisa_train.py).
"""
import pathlib

import pytest

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
pytestmark = pytest.mark.skipif(
    not (DATA / "deployment.json").exists(),
    reason="run harness/sisa_train.py to generate data first")


def test_e1_noninterference_and_controls():
    from harness import two_world
    delta = two_world._pick(0, 2, 1)
    cf = two_world.build_counterfactual(0, delta, DATA / "shard_cf_test.npz")
    h_a = two_world.run_world(DATA / "shard_0.npz", delta, 2)
    h_b = two_world.run_world(cf, [], 2)
    assert h_a == h_b, "physical exclusion must match the counterfactual world"
    assert two_world.loss_mask_train(0, delta, 2) != h_b, \
        "loss-masking must NOT satisfy non-interference"
    assert two_world.tainted_finetune(0, delta) != h_b, \
        "tainted fine-tune must NOT satisfy non-interference"


def test_e3_all_attacks_rejected_ablations_expose():
    from harness import attacks
    _, rows = attacks.run()
    ok_r, _ = attacks.randomness_manipulation()
    assert all(full == 0 for _, full, _, _ in rows), "an attack was accepted"
    assert ok_r, "randomness manipulation not blocked at API"
    assert all(weak == 1 for _, _, weak, _ in rows if weak is not None), \
        "a structural ablation failed to re-expose its attack"
