"""Compound erasure: a deletion must survive the deletions that follow it."""
from harness import compound


def test_sequential_deletions_verify_and_resurrection_is_refused():
    r = compound.run()
    assert r["first"]["accept"] == 1, r["first"]["reason"]
    assert r["second"]["accept"] == 1, r["second"]["reason"]
    # the ladder the first deletion issued keeps the second one local
    assert r["second"]["slices_retrained"] < r["first"]["slices_retrained"]
    assert r["resurrection_refused"], \
        "resuming from a checkpoint the record forbids was not refused"
    assert r["record_rollback_refused"], \
        "a rolled-back deletion record was not refused"
    assert r["verifier_rejects"], \
        "the auditor accepted a run that restored a deleted record"
    assert r["ablation_accepts"], \
        "the compound-erasure ablation failed to re-expose the attack"
