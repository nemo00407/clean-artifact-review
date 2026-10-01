"""End-to-end deletion-isolated retraining demo (non-TCB driver).

Picks a deletion target from a shard manifest, selects the latest clean
checkpoint (trained prefix disjoint from Delta), opens the invocation
through the gateway, runs the closed-input runtime with the enclave, and
verifies the signed ARC with the independent client-side verifier.
"""
import json
import time

from clean_runtime import attest, commit, gateway, runtime
from verifier import verify

from . import common


def pick_delta(shard_id=0, slice_idx=2, count=1):
    """Delete `count` record(s) from the given slice of a shard."""
    manifest = json.loads(
        (common.DATA_DIR / f"shard_{shard_id}.manifest.json").read_text())
    lo = slice_idx * common.RECORDS_PER_SLICE
    return [(int(ix), h) for ix, h in manifest[lo:lo + count]]


HEADS = "enclave.record_heads.json"
VERSIONS = "enclave.versions.json"


def record_path(shard_id):
    """The deletion record body, host-held and therefore untrusted."""
    return common.DATA_DIR / f"dlog_{shard_id}.json"


def load_record(shard_id):
    p = record_path(shard_id)
    if not p.exists():
        return []
    return [[bytes.fromhex(h) for h in entry]
            for entry in json.loads(p.read_text())]


def save_record(shard_id, entries):
    record_path(shard_id).write_text(json.dumps(
        [[bytes(h).hex() for h in entry] for entry in entries]))


def load_enclave():
    """Reconstruct the enclave, including the record heads the platform
    persists for it alongside the sealed key."""
    sealed = (common.sealed_dir() / "enclave.sealed").read_bytes()
    hp = common.sealed_dir() / HEADS
    heads = json.loads(hp.read_text()) if hp.exists() else {}
    vp = common.sealed_dir() / VERSIONS
    versions = json.loads(vp.read_text()) if vp.exists() else {}
    return attest.Enclave(sealed, gateway.runtime_measurement(), heads,
                          versions)


def save_enclave_heads(enclave):
    (common.sealed_dir() / HEADS).write_text(json.dumps(enclave.record_heads()))
    (common.sealed_dir() / VERSIONS).write_text(
        json.dumps({str(k): v for k, v in enclave.max_versions().items()}))


def run(shard_id=0, target_slice=2, nonce_label="nonce-demo",
        out_name="adapter_unlearned.npz", clean_version=None, delta=None,
        persist=False):
    config = json.loads((common.DATA_DIR / "config.json").read_text())
    if delta is None:
        delta = pick_delta(shard_id, target_slice)
    if clean_version is None:
        clean_version = target_slice  # CP_v trained slices 0..v-1
    enclave = load_enclave()
    record = load_record(shard_id)
    nonce = common.harness_r(nonce_label)  # client-issued session challenge
    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{shard_id}_v{clean_version}.npz",
        common.DATA_DIR / f"shard_{shard_id}.npz",
        delta,
        enclave.derive_r(nonce),
        config,
        nonce=nonce, enclave=enclave, record=record)
    t0 = time.time()
    new_lora, arc, losses = runtime.unlearn(inv, enclave)
    elapsed = time.time() - t0
    if out_name:
        gateway.write_adapter(
            common.DATA_DIR / out_name, new_lora,
            {"shard_id": shard_id, "source_cp_version": clean_version})
    if persist:
        # A completed deletion joins the record and supersedes the checkpoint
        # ladder above it. Measurement runs leave both untouched.
        save_record(shard_id, record + [list(inv.delta_hashes)])
        save_enclave_heads(enclave)
        for cp in inv.issued:
            meta = {k: v for k, v in cp.items() if k != "lora"}
            meta["h_config_train"] = commit.config_hash(config).hex()
            gateway.write_adapter(
                common.DATA_DIR
                / f"cp_{shard_id}_v{cp['version']}.npz", cp["lora"], meta)
    return new_lora, arc, losses, elapsed


def verify_run(arc, shard_id=0, nonce_label="nonce-demo", delta=None,
               target_slice=2, clean_version=None, record=None,
               known_head=None, unsafe_skip=frozenset()):
    """Client-side verification against local expectations."""
    dep = json.loads((common.DATA_DIR / "deployment.json").read_text())
    manifest = json.loads(
        (common.DATA_DIR / f"shard_{shard_id}.manifest.json").read_text())
    bounds = json.loads(
        (common.DATA_DIR / f"shard_{shard_id}.npz.meta.json").read_text()
    )["slice_bounds"]
    if delta is None:
        delta = pick_delta(shard_id, target_slice)
    if clean_version is None:
        clean_version = target_slice
    cp_meta = json.loads(
        (common.DATA_DIR / f"cp_{shard_id}_v{clean_version}.npz.meta.json"
         ).read_text())
    t0 = time.time()
    if record is None:
        record = load_record(shard_id)
    accept, reason = verify.verify_arc(
        bytes.fromhex(dep["pk"]), arc, record=record,
        known_head=known_head, unsafe_skip=unsafe_skip,
        nonce=common.harness_r(nonce_label), delta=delta, manifest=manifest,
        slice_bounds=bounds, cp_meta=cp_meta,
        cp_cert=bytes.fromhex(cp_meta["cert"]),
        c_base=bytes.fromhex(dep["c_base"]),
        h_config=bytes.fromhex(dep["h_config"]),
        mrenclave=bytes.fromhex(dep["mrenclave"]))
    return accept, reason, time.time() - t0


def main():
    new_lora, arc, losses, elapsed = run()
    print(f"retrained in {elapsed:.1f}s, "
          f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
    for k, v in arc.items():
        print(f"  {k}: {v.hex() if isinstance(v, bytes) else v}")
    accept, reason, vt = verify_run(arc)
    print(f"verify: accept={accept} ({reason}) in {vt * 1000:.2f}ms")


if __name__ == "__main__":
    main()
