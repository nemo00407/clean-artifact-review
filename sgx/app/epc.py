"""How the retained commitment set behaves against a 94 MB EPC.

CLEAN keeps one commitment per admitted block inside the enclave while the
block bodies live in host storage. Section 2.1 motivates the whole design from
secure memory being too small to hold a run's working set; this measures what
happens when the thing the enclave *does* hold -- 32 bytes per block -- grows
past the EPC anyway.

The sweep allocates n commitments, then walks them re-deriving each from its
block, which is what the read path does on every fetch. Below the EPC the pages
are resident; above it the hardware pages them, and the cost of a pass is the
measurement.
"""
import hashlib, json, os, sys, time

SHA = hashlib.sha256
BLOCK = b"\x5a" * 2048          # a block body, as store.py would hand it over


def build(n):
    """n commitments, held as one contiguous buffer, 32 bytes each."""
    buf = bytearray(n * 32)
    h = SHA(BLOCK).digest()
    for i in range(n):
        buf[i * 32:(i + 1) * 32] = h
    return buf


def pass_over(buf, n, sample):
    """Re-derive and compare `sample` commitments spread across the buffer.

    Spreading rather than scanning is deliberate: the read path touches blocks
    in whatever order training asks for, so a sequential walk would flatter the
    page cache.
    """
    stride = max(1, n // sample)
    t0 = time.perf_counter()
    bad = 0
    for i in range(0, n, stride):
        if SHA(BLOCK).digest() != bytes(buf[i * 32:(i + 1) * 32]):
            bad += 1
    dt = time.perf_counter() - t0
    touched = len(range(0, n, stride))
    return {"touched": touched, "seconds": dt,
            "per_touch_us": dt / touched * 1e6, "mismatches": bad}


def main():
    sizes = [int(x) for x in (sys.argv[1:] or
                              ["1000", "10000", "100000", "1000000",
                               "3000000", "6000000", "10000000"])]
    out = {"in_sgx": os.path.exists("/dev/attestation"),
           "epc_bytes": 94 * 1024 * 1024, "rows": []}
    for n in sizes:
        held = n * 32
        try:
            t0 = time.perf_counter()
            buf = build(n)
            build_s = time.perf_counter() - t0
            r = pass_over(buf, n, sample=20000)
            r.update({"n": n, "held_bytes": held, "build_s": build_s,
                      "over_epc": held > out["epc_bytes"]})
            out["rows"].append(r)
            print(f"n={n:>10,}  held={held/1e6:7.1f} MB  "
                  f"{'OVER ' if r['over_epc'] else 'under'} EPC  "
                  f"build {build_s:6.2f} s  "
                  f"{r['per_touch_us']:7.2f} us/commitment", flush=True)
            del buf
        except MemoryError:
            print(f"n={n:>10,}  MemoryError", flush=True)
            out["rows"].append({"n": n, "held_bytes": held,
                                "error": "MemoryError"})
            break
    print("JSON " + json.dumps(out))


if __name__ == "__main__":
    main()
