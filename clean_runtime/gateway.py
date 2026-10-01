"""Secure Input Gateway - the sole mediation point of the runtime.

Complete mediation: every piece of information entering an unlearning
invocation is loaded here, validated, and frozen (read-only) before the
trainer runs; the trainer itself performs no I/O. The deleted set Delta is
*physically excluded* at this boundary: the DatasetHandle handed to the
trainer contains no deleted row, and Delta payloads are referenced only by
their record hashes. This module is also the only writer of runtime outputs.

Admission also fixes the values in time. Inputs that leave enclave memory
are blocked and committed here (see `store`), so a later fetch that returns
anything other than what was admitted aborts the invocation instead of
silently entering the state transition. The gateway is likewise the only
component that moves values across the device boundary (see `device`): the
payload runs on the far side of that boundary and has no path back.
"""
import json
import pathlib
from types import MappingProxyType

import numpy as np

from . import commit, device as device_mod, dlog, store
from .rng import TeeRng


ATTESTATION = pathlib.Path("/dev/attestation")


def runtime_measurement() -> bytes:
    """MRENCLAVE: the measurement the platform assigns to this image.

    On SGX the value is read from the enclave's own report, where the hardware
    placed it: bytes 64..96 of `sgx_report_body_t`. It covers the loader, the
    interpreter, the libraries and the runtime source, and the manifest that
    configured them, so it names a deployed image and not a directory of files.
    Writing `target_info` is what makes the report readable; a report about
    ourselves needs no other target.

    Off SGX the same identity is emulated by hashing the runtime source tree,
    which is what the prototype used before it had hardware to ask. Either way
    measurement requires file access, which is why it lives in the gateway.
    """
    report = ATTESTATION / "report"
    if report.exists():
        (ATTESTATION / "target_info").write_bytes(
            (ATTESTATION / "my_target_info").read_bytes())
        return bytes(report.read_bytes()[64:96])
    root = pathlib.Path(__file__).resolve().parent
    h_input = bytearray()
    for path in sorted(root.glob("*.py")):
        h_input += path.name.encode() + path.read_bytes()
    return commit.sha256(bytes(h_input))


def _freeze(arr):
    a = np.ascontiguousarray(arr).copy()
    a.setflags(write=False)
    return a


class BaseModelHandle:
    def __init__(self, params):
        self.params = MappingProxyType({k: _freeze(v) for k, v in params.items()})


class CheckpointHandle:
    def __init__(self, lora, meta):
        self.lora = MappingProxyType({k: _freeze(v) for k, v in lora.items()})
        self.meta = MappingProxyType(dict(meta))


class DatasetHandle:
    """Clean training rows only (retrain region minus Delta).

    `tokens` is the one input re-read on every batch of the state transition,
    so it is admitted through `store`: the rows are blocked, committed, and
    handed to `backend`, and each fetch during training is checked against
    the commitment taken here. The index and salt columns are consumed once
    at admission and stay enclave-resident.
    """

    def __init__(self, indices, salts, tokens, h_dclean, backend=None,
                 verify=True, path=None):
        self.indices = _freeze(indices)
        self.salts = _freeze(salts)
        self.h_dclean = h_dclean
        self.backend = backend if backend is not None \
            else store.EnclaveResidentBackend()
        self.tokens = store.admit(self.backend, "dclean/tokens", tokens,
                                  verify=verify, path=path)
        self.slice_spans = ()
        self.prefix_roots = {}

    def __len__(self):
        return len(self.indices)


class Invocation:
    """Frozen input tuple I = (base, CP, D_clean, r, c) plus bindings."""

    def __init__(self, base, cp, data, tee_rng, config, h_delta,
                 delta_indices, nonce):
        self.base = base
        self.cp = cp
        self.data = data
        self.rng = tee_rng
        self.config = MappingProxyType(dict(config))
        self.h_config = commit.config_hash(dict(config))
        self.h_delta = h_delta
        self.delta_indices = _freeze(delta_indices)
        self.nonce = bytes(nonce)
        self.device = None
        self.h_record = b""
        self.delta_hashes = ()
        self.issued = []


def load_base(path) -> BaseModelHandle:
    with np.load(path) as npz:
        return BaseModelHandle({k: npz[k] for k in npz.files})


def load_checkpoint(path) -> CheckpointHandle:
    with open(str(path) + ".meta.json") as f:
        meta = json.load(f)
    with np.load(path) as npz:
        return CheckpointHandle({k: npz[k] for k in npz.files}, meta)


def _load_shard(path):
    with open(str(path) + ".meta.json") as f:
        meta = json.load(f)
    with np.load(path) as npz:
        return npz["index"].copy(), npz["salt"].copy(), npz["tokens"].copy(), meta


def _slice_of_row(slice_bounds, row):
    for s, (lo, hi) in enumerate(slice_bounds):
        if lo <= row < hi:
            return s
    raise ValueError("row outside all slice bounds")


def open_invocation(base_path, cp_path, shard_path, delta, r, config,
                    nonce=b"", backend=None, verify=True,
                    device=None, enclave=None, record=()) -> Invocation:
    """Build the frozen invocation. `delta` is a list of (index, record_hash)
    pairs; the gateway never exposes deleted payloads to the trainer."""
    dev = device if device is not None else device_mod.DeviceContext()
    base = load_base(base_path)
    # the frozen base model crosses the boundary once and stays resident;
    # device memory is integrity-protected, so it is not re-checked per batch
    dev.place("base", base)
    cp = load_checkpoint(cp_path)
    index, salt, tokens, shard_meta = _load_shard(shard_path)
    bounds = shard_meta["slice_bounds"]
    if cp.meta["shard_id"] != shard_meta["shard_id"]:
        raise ValueError("checkpoint does not belong to this shard")

    # The deletion record is host-held, so it is believed only once it hashes
    # to the head the enclave retained across invocations.
    record = [list(e) for e in record]
    if enclave is not None:
        if dlog.head_of(shard_meta["shard_id"],
                        record) != enclave.record_head(shard_meta["shard_id"]):
            raise ValueError(
                "deletion record does not match the head retained by the "
                "enclave; it was truncated, reordered or rolled back")
    # A checkpoint was trained on what survived the deletions recorded when it
    # was certified, so only deletions appended after it can still be inside
    # its prefix. Its certificate names the head it was issued under.
    later = dlog.entries_after(shard_meta["shard_id"], record,
                               bytes.fromhex(cp.meta["h_record"]))
    if later is None:
        raise ValueError(
            "the checkpoint's record head is not a prefix of this shard's "
            "deletion record; the record was forked or truncated")
    forgotten = dlog.deleted_hashes(later)
    # Everything this shard has ever been asked to forget. `forgotten` answers
    # whether the checkpoint's prefix is clean; this answers which rows of the
    # retraining region must be withheld. The two differ: a deletion recorded
    # before the checkpoint was certified is absent from the state we resume
    # from, yet its row survives in host storage until compaction runs, and
    # re-admitting it now would train it straight back in.
    erased = dlog.deleted_hashes(record)

    delta_idx = np.array(sorted(int(i) for i, _ in delta), dtype=np.uint32)
    delta_hash = {int(i): bytes.fromhex(h) if isinstance(h, str) else bytes(h)
                  for i, h in delta}
    row_of = {int(ix): row for row, ix in enumerate(index)}

    retrain_lo = bounds[cp.meta["slice_idx"] + 1][0]

    # Compound erasure: the state we resume from must be free of everything
    # this shard has already been asked to forget, not only of the request in
    # hand. Without this a later deletion can resume from a checkpoint that
    # predates an earlier one and carry its records back into the deployment.
    for row in range(retrain_lo):
        if commit.record_hash(salt[row], tokens[row]) in forgotten:
            raise ValueError(
                f"checkpoint is not clean: row {row} of its trained prefix "
                f"was deleted by an earlier invocation")

    # Cleanliness guard: every deleted record must lie strictly after the
    # checkpoint's trained prefix, and its hash must match the stored record.
    for ix in delta_idx:
        ix = int(ix)
        if ix not in row_of:
            raise ValueError(f"deleted index {ix} not present in shard")
        row = row_of[ix]
        if _slice_of_row(bounds, row) <= cp.meta["slice_idx"]:
            raise ValueError(
                f"checkpoint is not clean: deleted index {ix} is inside "
                f"its trained prefix (slice_idx={cp.meta['slice_idx']})")
        if commit.record_hash(salt[row], tokens[row]) != delta_hash[ix]:
            raise ValueError(f"deletion request hash mismatch at index {ix}")

    # Physical exclusion: retain retrain-region rows that this request does not
    # name and that no earlier deletion named either. Only these rows are copied
    # into the trainer-visible buffers.
    retrain_rows = np.arange(retrain_lo, bounds[-1][1])
    alive = np.array(
        [commit.record_hash(salt[r], tokens[r]) not in erased
         for r in retrain_rows], dtype=bool)
    keep = retrain_rows[alive & ~np.isin(index[retrain_rows], delta_idx)]
    data = DatasetHandle(
        index[keep], salt[keep], tokens[keep],
        commit.dataset_root(index[keep], salt[keep], tokens[keep]),
        backend=backend, verify=verify, path=dev.path)
    # Slice boundaries in D_clean coordinates, so the retraining can walk the
    # shard the same way the initial training built it and certify as it goes.
    spans, prefix_roots = [], {}
    head_rows = np.arange(0, retrain_lo)
    for j in range(cp.meta["slice_idx"] + 1, len(bounds)):
        lo_row, hi_row = bounds[j]
        sel = np.nonzero((keep >= lo_row) & (keep < hi_row))[0]
        if sel.size:
            spans.append((j, int(sel[0]), int(sel[-1]) + 1))
        # What a checkpoint taken at the end of slice j will have been trained
        # on: every row up to that boundary that survives this deletion and
        # every deletion before it. The auditor recomputes this from the public
        # manifest minus the record, so it must be the retained set and not the
        # original one.
        through = np.concatenate([head_rows, keep[keep < hi_row]]).astype(int)
        prefix_roots[j] = commit.dataset_root(
            index[through], salt[through], tokens[through])
    data.slice_spans = tuple(spans)
    data.prefix_roots = prefix_roots

    del index, salt, tokens  # drop references to buffers containing Delta

    h_delta = commit.hashes_root(delta_idx,
                                 [delta_hash[int(i)] for i in delta_idx])
    inv = Invocation(base, cp, data, TeeRng(r), config, h_delta, delta_idx,
                     nonce)
    inv.device = dev
    inv.delta_hashes = tuple(delta_hash[int(i)] for i in delta_idx)
    inv.h_record = (enclave.record_head(shard_meta["shard_id"])
                    if enclave is not None
                    else dlog.head_of(shard_meta["shard_id"], record))
    return inv


def write_adapter(path, lora_params, meta) -> None:
    """Sole output path of the runtime: the new adapter and its metadata."""
    np.savez(path, **{k: np.asarray(v, dtype=np.float32)
                      for k, v in lora_params.items()})
    with open(str(path) + ".meta.json", "w") as f:
        json.dump(meta, f, sort_keys=True, indent=1)
