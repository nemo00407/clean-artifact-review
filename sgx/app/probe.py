"""Smallest thing that shows the enclave boundary is real.

Reports the measurement the hardware assigns to this image, seals a value
under a key only this measurement can derive, and asks for a quote. Each is
something the emulated attest.py can only assert.
"""
import hashlib, json, os, pathlib, sys, time

ATT = pathlib.Path("/dev/attestation")


def measurement():
    """MRENCLAVE as the hardware reports it, not as a hash of the source."""
    if not (ATT / "report").exists():
        return None, "no /dev/attestation (running outside SGX?)"
    # Writing target_info is what makes the report readable; for a local
    # report about ourselves an all-zero target is enough.
    (ATT / "target_info").write_bytes((ATT / "my_target_info").read_bytes())
    rep = (ATT / "report").read_bytes()
    # sgx_report_body_t: mr_enclave at offset 64, mr_signer at 128
    return {"mr_enclave": rep[64:96].hex(), "mr_signer": rep[128:160].hex(),
            "report_bytes": len(rep)}, None


def sealing(path=pathlib.Path("/sealed/state.json"), reps=200):
    """Two questions about the sealed state of Assumption 2.

    First, can this enclave open what the *previous* one left behind? The key
    is derived from the measurement, so the answer is yes only if the image is
    byte-identical. That is the property the paper assumes and never showed.

    Second, what do a seal and an unseal cost? Measured on a blob the size of
    the state CLEAN actually persists: the signing seed and one record head
    per shard.
    """
    blob = json.dumps({"sk_seed": "00" * 32,
                       "heads": {str(i): "11" * 32 for i in range(4)}}).encode()

    inherited, inherit_err = None, None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            inherited = path.read_bytes() == blob
    except Exception as e:                                      # noqa: BLE001
        # A file sealed under a different measurement. Gramine will not open
        # it and will not let this enclave replace it either, which is the
        # point: the previous state is not merely unreadable, it is out of
        # reach. Even stat() is refused, so this catch has to cover the whole
        # approach to the file. The runner clears it between steps.
        return {"inherited_previous": False,
                "inherit_error": f"{type(e).__name__}: {e}"}

    t0 = time.perf_counter()
    for _ in range(reps):
        path.write_bytes(blob)
    seal = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        back = path.read_bytes()
    unseal = (time.perf_counter() - t0) / reps
    assert back == blob, "sealed state did not round-trip"
    return {"bytes": len(blob), "seal_s": seal, "unseal_s": unseal,
            "inherited_previous": inherited,
            "inherit_error": inherit_err}


def quote(reps=5):
    if not (ATT / "quote").exists():
        return None, "no /dev/attestation/quote"
    try:
        (ATT / "target_info").write_bytes((ATT / "my_target_info").read_bytes())
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            q = (ATT / "quote").read_bytes()
            ts.append(time.perf_counter() - t0)
        ts.sort()
        return {"bytes": len(q), "median_s": ts[len(ts) // 2]}, None
    except Exception as e:                                      # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def main():
    out = {"in_sgx": ATT.exists()}
    m, err = measurement()
    out["measurement"], out["measurement_error"] = m, err
    try:
        out["sealing"] = sealing()
    except Exception as e:                                      # noqa: BLE001
        out["sealing"], out["sealing_error"] = None, f"{type(e).__name__}: {e}"
    q, qerr = quote()
    out["quote"], out["quote_error"] = q, qerr
    try:
        import numpy, cryptography
        out["deps"] = {"numpy": numpy.__version__,
                       "cryptography": cryptography.__version__}
    except ImportError as e:
        out["deps_error"] = str(e)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
