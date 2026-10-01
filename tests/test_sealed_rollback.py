"""A restored sealed state must not buy a certificate for the world it restores.

Sealing binds persistent state to the enclave identity and not to the moment it
was written, so a host that keeps a copy of the sealed deletion record from
before a deletion can put it back and let the enclave resume from it. The
enclave has no way to tell; everything it is handed decrypts and hashes exactly
as it should, and the certificate it issues is genuine.

What the restored world does not match is the auditor's own memory. Every
certificate binds the record head as it stood, so an auditor that carries a head
forward from an earlier session can require the served record to still pass
through it, which no restored earlier state does. These tests pin both halves:
the attack reaches a signed certificate, and the carried head refuses it.
"""
from harness import compound, run_unlearn
from clean_runtime import dlog


def _rollback_case():
    with compound.isolated_data():
        snapshot = compound.sealed_snapshot()
        first = compound._delete(compound.SHARD,
                                 run_unlearn.pick_delta(compound.SHARD, 2, 1),
                                 "rollback-test", slices_expected=2)
        erased = dlog.deleted_hashes(run_unlearn.load_record(compound.SHARD))
        yield first, compound.rollback_case(
            snapshot, bytes.fromhex(first["h_record"]), erased)


def test_restored_state_certifies_and_the_carried_head_refuses_it():
    for first, case in _rollback_case():
        assert first["accept"] == 1
        # The restored world does put a deleted record back: the checkpoint it
        # resumes from is one the deletion had superseded.
        assert case["restored"], "nothing was restored, so nothing was tested"
        # An auditor with no earlier head is seeing this shard for the first
        # time and has nothing to compare against.
        assert case["first_time_accepts"]
        # One that carried its head forward does.
        assert case["stateful_rejects"]


def test_the_freshness_check_is_what_refuses_it():
    for _, case in _rollback_case():
        assert case["ablation_accepts"], (
            "removing the freshness check must let the certificate through, "
            "or something else is doing the refusing")
