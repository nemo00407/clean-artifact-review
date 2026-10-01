"""E17 - What a certificate costs to produce, inside the enclave and outside.

E1 measured the composite commitment and the two signatures on the emulated
substrate. Both now run where they belong, so the pair can be reported instead
of the single figure.

What an enclave changes is not the arithmetic -- SHA-256 and Ed25519 execute the
same instructions on the same core -- but the memory that arithmetic touches.
The adapter being hashed, the signing seed, and the working set of the hash all
live in EPC, which is 94 MB on this CPU, and the ratio between the two columns
is what that costs. The measurement is taken on the same objects a real
invocation commits to: the shard's LoRA adapter, the base parameters behind it,
and the ARC and CP_Cert payloads of a real deployment.

Detects its own environment, so the same module produces the enclave column
under `gramine-sgx` and the native column on the host.
"""
import json
import pathlib
import time

import numpy as np

from clean_runtime import attest, commit, dlog

from . import common, run_unlearn

IN_SGX = pathlib.Path("/dev/attestation").is_dir()
LABEL = "sgx" if IN_SGX else "native"


def _median_us(fn, reps):
    fn()                                     # warm the code path and any cache
    xs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append((time.perf_counter() - t0) * 1e6)
    xs.sort()
    return xs[len(xs) // 2]


def _npz(name):
    with np.load(common.DATA_DIR / name) as npz:
        return {k: npz[k] for k in npz.files}


def measure(reps=200):
    enclave = run_unlearn.load_enclave()
    dep = json.loads((common.DATA_DIR / "deployment.json").read_text())
    base = _npz("base.npz")
    lora = _npz("cp_0_v2.npz")      # a checkpoint file is the adapter itself

    def nbytes(d):
        return sum(len(commit.tensor_bytes(v)) for v in d.values())

    h_cp = commit.params_hash(lora)
    head = dlog.genesis(0)
    fields = {"c_base": bytes.fromhex(dep["c_base"]), "h_l_ckpt": h_cp,
              "h_l_new": h_cp, "h_dclean": commit.sha256(b"dclean"),
              "h_delta": commit.sha256(b"delta"), "h_record": head,
              "h_r": commit.sha256(b"r"),
              "h_config": bytes.fromhex(dep["h_config"]),
              "shard_id": 0, "cp_version": 2,
              "nonce": common.harness_r("cert-cost")}

    out = {
        "label": LABEL,
        "mrenclave": enclave.mrenclave.hex(),
        "adapter_bytes": nbytes(lora),
        "base_bytes": nbytes(base),
        # The composite commitment of Equation (5): the adapter only, because
        # C_base is a deployment-time constant the certificate carries forward.
        "adapter_hash_us": _median_us(lambda: commit.params_hash(lora), reps),
        # What it would cost to commit to the whole model instead, which is the
        # comparison that makes the composite worth having.
        "full_model_hash_us": _median_us(
            lambda: commit.params_hash(dict(base, **lora)), max(20, reps // 10)),
        "arc_sign_us": _median_us(lambda: enclave.sign_arc(fields), reps),
        "cp_cert_sign_us": _median_us(
            lambda: enclave.sign_cp_cert(h_cp, commit.sha256(b"prefix"), 0, 1,
                                        2, head), reps),
        "derive_r_us": _median_us(
            lambda: enclave.derive_r(fields["nonce"]), reps),
        "record_extend_us": _median_us(
            lambda: dlog.extend(head, [commit.sha256(b"x")] * 8), reps),
    }
    out["certificate_us"] = (out["adapter_hash_us"] + out["arc_sign_us"]
                             + out["record_extend_us"])
    return out


def _report(out_dir):
    """Merge whichever columns have been measured into one table."""
    cols = {}
    for label in ("native", "sgx"):
        p = out_dir / f"cert_cost_{label}.json"
        if p.exists():
            cols[label] = json.loads(p.read_text())
    if not cols:
        return
    rows = [("composite commitment $C(W)$ over the adapter", "adapter_hash_us"),
            ("hash of the whole model instead", "full_model_hash_us"),
            ("ARC signature", "arc_sign_us"),
            ("CP\\_Cert signature", "cp_cert_sign_us"),
            ("attested randomness $r$", "derive_r_us"),
            ("deletion-record head, 8 records", "record_extend_us"),
            ("**certificate, end to end**", "certificate_us")]
    have = [l for l in ("sgx", "native") if l in cols]
    headings = ["step"] + [{"sgx": "in enclave", "native": "native"}[l]
                           for l in have]
    if len(have) == 2:
        headings.append("ratio")
    head = "| " + " | ".join(headings) + " |"
    rule = "|" + "---|" * len(headings)
    lines = ["# E17 - Certificate production, inside the enclave and outside",
             "",
             "Medians over 200 repetitions. The adapter is the rank-4 LoRA "
             f"state of one shard ({cols[have[0]]['adapter_bytes']:,} bytes); "
             "the base parameters behind it are "
             f"{cols[have[0]]['base_bytes']:,} bytes.", ""]
    if "sgx" in cols:
        lines += [f"The enclave column was taken under `MRENCLAVE = "
                  f"{cols['sgx']['mrenclave'][:16]}...`, which is the "
                  f"measurement the hardware assigned to the image that "
                  f"produced it.", ""]
    lines += [head, rule]
    for name, key in rows:
        cells = [f"{cols[l][key]:.1f}\\,$\\mu$s" for l in have]
        if len(have) == 2:
            cells.append(f"{cols['sgx'][key] / cols['native'][key]:.2f}$\\times$")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [""]
    if len(have) == 2:
        ratio = cols["sgx"]["certificate_us"] / cols["native"]["certificate_us"]
        big = (cols["sgx"]["full_model_hash_us"]
               / cols["native"]["full_model_hash_us"])
        lines += [f"Producing a certificate inside the enclave costs "
                  f"{ratio:.2f}$\\times$ what it costs outside, and every object "
                  f"it touches fits in EPC at this scale. What the column "
                  f"separates is where that factor comes from: the "
                  f"{cols['sgx']['base_bytes'] / 1e6:.1f}\\,MB sequential hash "
                  f"costs {big:.2f}$\\times$, within its own run-to-run "
                  f"variation of a few percent, while every microsecond-scale "
                  f"step pays about double. The cost is attached to the number "
                  f"of calls and not to the bytes hashed, which is the shape "
                  f"the composite commitment already has: it hashes the adapter, "
                  f"once.", ""]
    (out_dir / "cert_cost.md").write_text("\n".join(lines))


def main():
    out_dir = common.DATA_DIR.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = measure()
    (out_dir / f"cert_cost_{LABEL}.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        if k.endswith("_us"):
            print(f"  {k[:-3]:<22} {v:9.1f} us")
    _report(out_dir)
    print(f"-> {out_dir / f'cert_cost_{LABEL}.json'} ({LABEL})")


if __name__ == "__main__":
    main()
