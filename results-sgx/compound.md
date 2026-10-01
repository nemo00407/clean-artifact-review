# E13 - Compound erasure across invocations

Two deletions on the same shard, in sequence, each persisted to the shard's deletion record.

| step | resumed from | slices retrained | verifies |
|---|---|---:|---|
| delete from slice 2 | v2 (slice 1) | 2 | yes |
| delete from slice 3 | v5 (slice 2) | 1 | yes |

The second deletion resumes from a checkpoint the first one issued, so it retrains 1 slice rather than falling back to a pre-deletion state.

Restoring the first deletion, four ways:

- resuming from a checkpoint whose prefix holds it: refused by the gateway: **True**
- serving a record with that entry removed: refused against the retained head: **True**
- an ARC from a run that bypassed the gateway: rejected by the full verifier: **True**; accepted with the compound-erasure check ablated: **True**
- restoring the sealed state from before the first deletion and certifying in the world that comes back: the run resumes from v3 (slice 2), whose trained prefix carries record 232. An auditor holding no earlier head accepts it: **True**. An auditor carrying the head it saw when that deletion was certified rejects it: **True**; with the freshness check ablated it accepts again: **True**
