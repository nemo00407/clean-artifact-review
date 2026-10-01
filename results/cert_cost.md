# E17 - Certificate production, inside the enclave and outside

Medians over 200 repetitions. The adapter is the rank-4 LoRA state of one shard (16,480 bytes); the base parameters behind it are 1,351,972 bytes.

The enclave column was taken under `MRENCLAVE = 428e366f2cc974a8...`, which is the measurement the hardware assigned to the image that produced it.

| step | in enclave | native | ratio |
|---|---|---|---|
| composite commitment $C(W)$ over the adapter | 73.0\,$\mu$s | 44.2\,$\mu$s | 1.65$\times$ |
| hash of the whole model instead | 2582.1\,$\mu$s | 2537.5\,$\mu$s | 1.02$\times$ |
| ARC signature | 73.0\,$\mu$s | 34.2\,$\mu$s | 2.13$\times$ |
| CP\_Cert signature | 73.9\,$\mu$s | 34.1\,$\mu$s | 2.17$\times$ |
| attested randomness $r$ | 7.9\,$\mu$s | 0.9\,$\mu$s | 8.41$\times$ |
| deletion-record head, 8 records | 29.1\,$\mu$s | 8.7\,$\mu$s | 3.34$\times$ |
| **certificate, end to end** | 175.0\,$\mu$s | 87.1\,$\mu$s | 2.01$\times$ |

Producing a certificate inside the enclave costs 2.01$\times$ what it costs outside, and every object it touches fits in EPC at this scale. What the column separates is where that factor comes from: the 1.4\,MB sequential hash costs 1.02$\times$, within its own run-to-run variation of a few percent, while every microsecond-scale step pays about double. The cost is attached to the number of calls and not to the bytes hashed, which is the shape the composite commitment already has: it hashes the adapter, once.
