"""A deletion must not come back through rows compaction has not reached yet.

Entries of the deletion record name records by hash, and the rows they name stay
in host storage until the compaction daemon runs. Admission therefore has to
withhold every record the shard has been asked to forget, not only the request in
hand: a later invocation whose retraining region covers an earlier deletion's row
would otherwise train that record straight back into the adapter.

The checkpoint's trained prefix is the other half of the same obligation, and
`test_compound.py` covers it. This test covers the retraining region, which needs
two deletions to land in the *same* slice --- if the earlier one falls in the
prefix of the checkpoint the later one resumes from, the prefix check catches it
first and this path is never exercised.
"""
import json

import pytest

from harness import common, compound, run_unlearn
from clean_runtime import commit, gateway, runtime

SHARD = 0
SLICE = 3


def _two_requests_in_one_slice():
    manifest = json.loads(
        (common.DATA_DIR / f"shard_{SHARD}.manifest.json").read_text())
    lo = SLICE * common.RECORDS_PER_SLICE
    first = [(int(manifest[lo][0]), manifest[lo][1])]
    second = [(int(manifest[lo + 1][0]), manifest[lo + 1][1])]
    return first, second


def test_earlier_deletion_is_not_readmitted_from_uncompacted_rows():
    with compound.isolated_data():
        config = json.loads((common.DATA_DIR / "config.json").read_text())
        first, second = _two_requests_in_one_slice()
        gone = int(first[0][0])

        run_unlearn.run(shard_id=SHARD, target_slice=SLICE,
                        clean_version=SLICE, delta=first, out_name=None,
                        persist=True, nonce_label="uncompacted-1")
        record = run_unlearn.load_record(SHARD)
        assert len(record) == 1, "the first deletion was not recorded"

        enclave = run_unlearn.load_enclave()
        nonce = common.harness_r("uncompacted-2")
        inv = gateway.open_invocation(
            common.DATA_DIR / "base.npz",
            common.DATA_DIR / f"cp_{SHARD}_v{SLICE}.npz",
            common.DATA_DIR / f"shard_{SHARD}.npz",
            second, enclave.derive_r(nonce), config, nonce=nonce,
            enclave=enclave, record=record)

        admitted = {int(i) for i in inv.data.indices}
        assert gone not in admitted, (
            f"record {gone}, deleted by the previous invocation, was admitted "
            f"again into the retraining region")

        # The honest path still completes and still verifies, so the exclusion
        # is not simply refusing the invocation.
        _, arc, _ = runtime.unlearn(inv, enclave)
        accept, reason, _ = run_unlearn.verify_run(
            arc, shard_id=SHARD, nonce_label="uncompacted-2", delta=second,
            clean_version=SLICE, record=record)
        assert accept == 1, reason


def test_the_verifier_refuses_a_run_that_readmitted_one():
    """The auditor catches it too, without seeing the runtime's state.

    `Verify` recomputes $H(\\mathcal{D}_{clean})$ from the public manifest, so it
    has to withhold the same set the gateway does. Handing it a certificate whose
    $H(\\mathcal{D}_{clean})$ covers the readmitted row must fail that check.
    """
    from verifier import verify
    with compound.isolated_data():
        config = json.loads((common.DATA_DIR / "config.json").read_text())
        first, second = _two_requests_in_one_slice()

        run_unlearn.run(shard_id=SHARD, target_slice=SLICE,
                        clean_version=SLICE, delta=first, out_name=None,
                        persist=True, nonce_label="uncompacted-3")
        record = run_unlearn.load_record(SHARD)

        enclave = run_unlearn.load_enclave()
        nonce = common.harness_r("uncompacted-4")
        inv = gateway.open_invocation(
            common.DATA_DIR / "base.npz",
            common.DATA_DIR / f"cp_{SHARD}_v{SLICE}.npz",
            common.DATA_DIR / f"shard_{SHARD}.npz",
            second, enclave.derive_r(nonce), config, nonce=nonce,
            enclave=enclave, record=record)
        _, arc, _ = runtime.unlearn(inv, enclave)

        # A host that withheld only the current request would commit to this
        # root instead. Nothing else about the certificate changes.
        manifest = json.loads(
            (common.DATA_DIR / f"shard_{SHARD}.manifest.json").read_text())
        bounds = json.loads(
            (common.DATA_DIR / f"shard_{SHARD}.npz.meta.json").read_text()
        )["slice_bounds"]
        cp_meta = json.loads(
            (common.DATA_DIR / f"cp_{SHARD}_v{SLICE}.npz.meta.json").read_text())
        prefix_end = bounds[cp_meta["slice_idx"]][1]
        deleted = {int(i) for i, _ in second}
        lax = [(int(ix), bytes.fromhex(h))
               for row, (ix, h) in enumerate(manifest)
               if row >= prefix_end and int(ix) not in deleted]
        forged = dict(arc, h_dclean=verify._root_of(lax))

        dep = json.loads((common.DATA_DIR / "deployment.json").read_text())
        accept, reason = verify.verify_arc(
            bytes.fromhex(dep["pk"]), forged, record=record,
            nonce=nonce, delta=second, manifest=manifest,
            slice_bounds=bounds, cp_meta=cp_meta,
            cp_cert=bytes.fromhex(cp_meta["cert"]),
            c_base=bytes.fromhex(dep["c_base"]),
            h_config=bytes.fromhex(dep["h_config"]),
            mrenclave=bytes.fromhex(dep["mrenclave"]))
        assert accept == 0, "the auditor accepted a readmitted deletion"
        assert "deletion record" in reason, reason
