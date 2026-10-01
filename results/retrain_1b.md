# E18 - Retraining cost against deletion position at 1B

`TinyLlama/TinyLlama-1.1B-Chat-v1.0`, LoRA rank 4 on `q_proj, v_proj`, 2 shards $\times$ 4 slices $\times$ 8 records. Each row is 6 invocations per slice position, over all shards. A deletion in slice 0 resumes from the base model, so that row is the cost of rebuilding the shard outright.

| deletion slice | slices retrained | median (s) | min | max |
|---:|---:|---:|---:|---:|
| 0 | 4 | 2.71 | 2.70 | 2.72 |
| 1 | 3 | 2.03 | 2.02 | 2.04 |
| 2 | 2 | 1.35 | 1.34 | 1.35 |
| 3 | 1 | 0.66 | 0.66 | 0.66 |

Retraining falls by 0.68\,s per slice the deletion spares, within 0.00\,s of a straight line across all 4 positions, and the line has no fixed component this measurement can separate from zero (intercept -0.02\,s). The relationship is SISA's and it is the payload-independent one the 0.9\,M model also shows; the two sets of absolute times are not comparable, because a slice here holds 8 records against 32 there.
