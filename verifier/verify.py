"""Client-side verifier (non-TCB, spec section 6).

Deliberately independent: this module re-implements the canonical hashing,
Merkle and message encodings and imports nothing from clean_runtime, so a
verification pass never trusts runtime code. The verifier reconstructs the
entire ARC payload from its *own* expectations (session nonce, deletion
request, shard manifest, deployment commitments) plus the certificate's
claimed output bindings; any host deviation either fails a structural
check or invalidates the signature.
"""
import hashlib
import struct

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

CP_TAG = b"CLEAN-CP-v1"
DLOG_TAG = b"CLEAN-DLOG-v1"
ARC_TAG = b"CLEAN-ARC-v1"


def _sha256(b):
    return hashlib.sha256(b).digest()


def _int(v):
    return struct.pack(">q", int(v))


def _encode(tag, fields):
    out = bytearray(tag)
    for f in fields:
        f = bytes(f)
        out += struct.pack(">I", len(f)) + f
    return bytes(out)


def _merkle(leaves):
    if len(leaves) == 0:
        return _sha256(b"")
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [_sha256(level[i] + level[i + 1])
                 for i in range(0, len(level), 2)]
    return level[0]


def _root_of(entries):
    """Merkle root over (index, hash) pairs, ordered by ascending index."""
    return _merkle([h for _, h in sorted(entries)])


def _dlog_genesis(shard_id):
    return _sha256(DLOG_TAG + b"genesis"
                   + int(shard_id).to_bytes(8, "big"))


def _dlog_extend(head, record_hashes):
    leaves = sorted(bytes(h) for h in record_hashes)
    return _sha256(DLOG_TAG + bytes(head) + _merkle(leaves))


def _dlog_head(shard_id, entries):
    head = _dlog_genesis(shard_id)
    for entry in entries:
        head = _dlog_extend(head, entry)
    return head


def _prefix_end(slice_bounds, slice_idx):
    return 0 if slice_idx < 0 else slice_bounds[slice_idx][1]


def verify_cp_cert(pk_bytes, cert_sig, h_cp, cp_meta, mrenclave):
    msg = _encode(CP_TAG, [h_cp, bytes.fromhex(cp_meta["h_d_prefix"]),
                           bytes.fromhex(cp_meta.get("h_record", "")),
                           _int(cp_meta["shard_id"]),
                           _int(cp_meta["slice_idx"]),
                           _int(cp_meta["version"]), mrenclave])
    try:
        Ed25519PublicKey.from_public_bytes(pk_bytes).verify(cert_sig, msg)
    except InvalidSignature:
        return False
    return True


def verify_arc(pk_bytes, arc, *, nonce, delta, manifest, slice_bounds,
               cp_meta, cp_cert, c_base, h_config, mrenclave, record=(),
               known_head=None, unsafe_skip=frozenset()):
    """Full spec-section-6 verification. Returns (accept: 0/1, reason).

    `record` is the shard's deletion record as published by the provider:
    the deletions performed before this one, oldest first, each a list of
    record hashes. The auditor recomputes its head and re-runs the
    compound-erasure test, so a provider that truncates the record cannot
    hide an earlier deletion the resumed checkpoint would restore.

    `known_head` is the freshness token: a record head this auditor saw
    in an earlier session, carried forward. The served record must still
    pass through it. An auditor holding none is auditing this shard for the
    first time and the check does not apply.

    `unsafe_skip` disables named checks ("nonce", "cp_cert", "clean",
    "config", "dclean", "delta_binding", "record", "freshness") and exists
    ONLY to build the weakened verifier ablations of the evaluation; never
    set it in deployment."""
    delta = [(int(ix), bytes.fromhex(h) if isinstance(h, str) else bytes(h))
             for ix, h in delta]
    row_of = {int(ix): row for row, (ix, _) in enumerate(manifest)}
    man_hash = {int(ix): bytes.fromhex(h) for ix, h in manifest}

    # (3) deletion binding: request must reference real records
    if "delta_binding" not in unsafe_skip:
        for ix, h in delta:
            if ix not in row_of:
                return 0, f"delta index {ix} not in manifest"
            if man_hash[ix] != h:
                return 0, f"delta hash mismatch at index {ix}"
    h_delta = arc["h_delta"] if "delta_binding" in unsafe_skip \
        else _root_of(delta)

    # (4) checkpoint certificate and clean-rollback judgment
    if "cp_cert" not in unsafe_skip:
        if not verify_cp_cert(pk_bytes, cp_cert, arc["h_l_ckpt"], cp_meta,
                              mrenclave):
            return 0, "invalid checkpoint certificate"
        if (int(cp_meta["shard_id"]) != int(arc["shard_id"])
                or int(cp_meta["version"]) != int(arc["cp_version"])):
            return 0, "checkpoint identity does not match ARC"
    if "clean" not in unsafe_skip:
        prefix_end = _prefix_end(slice_bounds, cp_meta["slice_idx"])
        for ix, _ in delta:
            if ix in row_of and row_of[ix] < prefix_end:
                return 0, f"checkpoint not clean: index {ix} in trained prefix"

    record = [list(e) for e in record]

    # (4b) freshness: the auditor's own state anchors the deletion record.
    # Sealing binds persistent state to the enclave identity, not to time, so
    # a host can restore a sealed record from before a deletion it already
    # certified and obtain a fresh signature over the older world. An auditor
    # carrying a head from an earlier session requires the served record to
    # still pass through it, which no restored earlier state does.
    if "freshness" not in unsafe_skip and known_head is not None:
        known = (bytes.fromhex(known_head) if isinstance(known_head, str)
                 else bytes(known_head))
        running = _dlog_genesis(arc["shard_id"])
        seen = running == known
        for entry in record:
            running = _dlog_extend(running, entry)
            seen = seen or running == known
        if not seen:
            return 0, ("deletion record does not extend the head this "
                       "auditor last saw")

    # (4c) compound erasure: the resumed state must also be free of everything
    # deleted before this request, or this invocation restores it.
    h_record = _dlog_extend(_dlog_head(arc["shard_id"], record),
                            [h for _, h in delta])
    if "record" not in unsafe_skip:
        # Only deletions appended after the checkpoint was certified can still
        # be inside its prefix; its certificate names the head it carried.
        running = _dlog_genesis(arc["shard_id"])
        target = bytes.fromhex(cp_meta.get("h_record", ""))
        later, seen = None, running == target
        if seen:
            later = list(record)
        for i, entry in enumerate(record):
            running = _dlog_extend(running, entry)
            if running == target:
                later, seen = list(record[i + 1:]), True
                break
        if not seen:
            return 0, ("checkpoint's record head is not a prefix of the "
                       "deletion record")
        forgotten = {bytes(h) for e in later for h in e}
        prefix_end = _prefix_end(slice_bounds, cp_meta["slice_idx"])
        for row, (ix, h) in enumerate(manifest):
            if row >= prefix_end:
                break
            if bytes.fromhex(h) in forgotten:
                return 0, (f"checkpoint not clean: index {ix} of its trained "
                           f"prefix was deleted by an earlier invocation")
    else:
        h_record = arc["h_record"]

    # (5) verifiable physical exclusion. The retraining region must withhold
    # this request and every record the shard was asked to forget before it:
    # until compaction runs those rows are still in the manifest, and admitting
    # one again would retrain a deleted record back into the adapter.
    if "dclean" not in unsafe_skip:
        prefix_end = _prefix_end(slice_bounds, cp_meta["slice_idx"])
        deleted = {ix for ix, _ in delta}
        erased = {bytes(h) for entry in record for h in entry}
        retained = [(int(ix), bytes.fromhex(h))
                    for row, (ix, h) in enumerate(manifest)
                    if row >= prefix_end and int(ix) not in deleted
                    and bytes.fromhex(h) not in erased]
        if _root_of(retained) != arc["h_dclean"]:
            return 0, ("H(D_clean) does not match the manifest minus the "
                       "request and the deletion record")

    # (1)+(2)+(6) reconstruct the full payload from expected values and
    # verify the signature (covers freshness, identity and deployment
    # bindings in one shot)
    session_nonce = arc["nonce"] if "nonce" in unsafe_skip else nonce
    cfg = arc["h_config"] if "config" in unsafe_skip else h_config
    msg = _encode(ARC_TAG, [c_base, arc["h_l_ckpt"], arc["h_l_new"],
                            arc["h_dclean"], h_delta, h_record, arc["h_r"], cfg,
                            _int(arc["shard_id"]), _int(arc["cp_version"]),
                            mrenclave, session_nonce])
    try:
        Ed25519PublicKey.from_public_bytes(pk_bytes).verify(
            arc["signature"], msg)
    except InvalidSignature:
        return 0, "invalid ARC signature"
    return 1, "accept"
