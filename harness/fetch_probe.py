"""E9 - Where the post-admission substitutions are refused (non-TCB harness).

Assumption 1 closes the input domain in time as well as in origin. The
mechanism is a commitment recorded at admission and re-derived on every
fetch, so the guarantee is not "the substitution is eventually noticed" but
"the invocation stops on the fetch that carries it".

This probe makes that testable. The host is allowed to substitute a block
starting at fetch k, for k across the length of a run, on both surfaces the
threat model gives it: the storage backend (Mid-Invocation Rollback) and the
transfer path (Transfer Tampering). We record the fetch on which the run
aborts, and whether any certificate was produced. The property under test is
that the abort lands on the fetch carrying the substitution, so that no
altered block is ever consumed -- not that it lands on a fixed index.
"""
import json

import numpy as np

from clean_runtime import device, gateway, runtime, store

from . import attacks, common, run_unlearn, untrusted_store


class _CountingBackend(untrusted_store.HonestBackend):
    """Honest storage that reports how many fetches a run performs."""

    def __init__(self):
        super().__init__()
        self.fetches = 0

    def get(self, key):
        self.fetches += 1
        return super().get(key)


def _context():
    index, _, tokens = attacks._shard()
    bounds = json.loads(
        (common.DATA_DIR / f"shard_{attacks.SHARD}.npz.meta.json").read_text()
    )["slice_bounds"]
    cp_meta = json.loads(
        (common.DATA_DIR
         / f"cp_{attacks.SHARD}_v{attacks.CPV}.npz.meta.json").read_text())
    region = np.arange(bounds[cp_meta["slice_idx"] + 1][0], bounds[-1][1])
    stale = store.encode_block(tokens[region][:store.BLOCK_ROWS])
    return stale


def _invoke(delta, backend=None, path=None):
    """One invocation over the host-controlled surfaces given."""
    enclave = run_unlearn.load_enclave()
    nonce = common.harness_r(attacks.NONCE_LABEL)
    dev = device.DeviceContext(path=path) if path is not None else None
    inv = gateway.open_invocation(
        common.DATA_DIR / "base.npz",
        common.DATA_DIR / f"cp_{attacks.SHARD}_v{attacks.CPV}.npz",
        common.DATA_DIR / f"shard_{attacks.SHARD}.npz",
        delta, enclave.derive_r(nonce), attacks._config(), nonce=nonce,
        backend=backend, device=dev)
    _, arc, _ = runtime.unlearn(inv, enclave)
    return arc


def probe_storage(delta, stale, k):
    """Serve a stale block from fetch k+1 onward, on the one block key the
    host targets. Returns (first substituted fetch, abort fetch, consumed),
    where `consumed` counts substituted blocks the trainer actually read."""
    backend = untrusted_store.RollbackBackend(
        "dclean/tokens/0", stale, trigger_after=k)
    try:
        _invoke(delta, backend=backend)
        return backend.first_substitution_at, None, backend.substitutions
    except store.BlockIntegrityError:
        return (backend.first_substitution_at, backend.fetches,
                backend.substitutions - 1)


def probe_transfer(delta, k):
    """Corrupt bytes in flight from transfer k+1 onward."""
    path = untrusted_store.TamperingPath(trigger_after=k)
    try:
        _invoke(delta, path=path)
        return path.first_tamper_at, None, path.tampered
    except store.BlockIntegrityError:
        return path.first_tamper_at, path.transfers, path.tampered - 1


def main(k_max=12):
    delta = run_unlearn.pick_delta(attacks.SHARD, attacks.SLICE)
    stale = _context()

    counting = _CountingBackend()
    _invoke(delta, backend=counting)
    total = counting.fetches
    print(f"an unattacked invocation performs {total} block fetches")

    ks = list(range(1, min(k_max, total - 1) + 1))
    rows = []
    for k in ks:
        s_first, s_abort, s_used = probe_storage(delta, stale, k)
        t_first, t_abort, t_used = probe_transfer(delta, k)
        rows.append((k, s_first, s_abort, s_used, t_first, t_abort, t_used))
        print(f"k={k:3d}  storage: first substituted fetch {s_first}, "
              f"abort {s_abort}, consumed {s_used}   |   "
              f"transfer: first {t_first}, abort {t_abort}, "
              f"consumed {t_used}")
        assert s_abort is not None, "storage substitution went undetected"
        assert t_abort is not None, "transfer substitution went undetected"

    # The property is not that the run stops at a fixed index, but that it
    # stops on the fetch that carries the substitution, so nothing altered is
    # ever consumed.
    on_carrying = all(sa == sf and ta == tf
                      for _, sf, sa, _, tf, ta, _ in rows)
    none_consumed = all(su == 0 and tu == 0
                        for _, _, _, su, _, _, tu in rows)

    out = common.DATA_DIR.parent / "results"
    out.mkdir(exist_ok=True)
    lines = ["# E9 - Refusal point of post-admission substitution", "",
             f"An unattacked invocation performs **{total}** block fetches. "
             f"The host substitutes from fetch *k*+1 onward; the table gives "
             f"the fetch on which the invocation aborted. No certificate was "
             f"issued in any row.", "",
             "| k | storage: substituted at | aborted at | "
             "transfer: substituted at | aborted at |",
             "|---:|---:|---:|---:|---:|"]
    for k, sf, sa, _su, tf, ta, _tu in rows:
        lines.append(f"| {k} | {sf} | {sa} | {tf} | {ta} |")
    lines += ["",
              f"Aborted on the fetch carrying the substitution in every "
              f"case: {on_carrying}. Substituted blocks consumed by the "
              f"trainer: {0 if none_consumed else 'nonzero'}.",
              "",
              "The commitment is re-derived per fetch, so refusal lands on "
              "the carrying fetch regardless of how far into the run the "
              "substitution happens. The storage adversary targets one block "
              "key, so its substitution first lands on the next fetch of "
              "that block rather than on fetch k+1; the transfer adversary "
              "corrupts whatever crosses next.", ""]
    (out / "abort_point.md").write_text("\n".join(lines))

    dat = ["k s_first s_abort t_first t_abort"]
    for k, sf, sa, _su, tf, ta, _tu in rows:
        dat.append(f"{k} {sf} {sa} {tf} {ta}")
    (out / "abort_point.dat").write_text("\n".join(dat) + "\n")
    (out / "abort_point.meta.json").write_text(
        json.dumps({"total_fetches": total,
                    "aborted_on_carrying_fetch": on_carrying,
                    "substituted_blocks_consumed": 0 if none_consumed else -1}))
    print(f"-> {out / 'abort_point.md'} (carrying-fetch abort: {on_carrying})")


if __name__ == "__main__":
    main()
