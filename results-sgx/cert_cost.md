# E17 - Certificate production, inside the enclave and outside

Medians over 200 repetitions. The adapter is the rank-4 LoRA state of one shard (16,480 bytes); the base parameters behind it are 1,351,972 bytes.

The enclave column was taken under `MRENCLAVE = 428e366f2cc974a8...`, which is the measurement the hardware assigned to the image that produced it.

| step | in enclave |
|---|---|
| composite commitment $C(W)$ over the adapter | 73.0\,$\mu$s |
| hash of the whole model instead | 2582.1\,$\mu$s |
| ARC signature | 73.0\,$\mu$s |
| CP\_Cert signature | 73.9\,$\mu$s |
| attested randomness $r$ | 7.9\,$\mu$s |
| deletion-record head, 8 records | 29.1\,$\mu$s |
| **certificate, end to end** | 175.0\,$\mu$s |
