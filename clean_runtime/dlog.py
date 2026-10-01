"""Append-only record of the deletions an enclave has performed (spec: compound
erasure).

A single invocation's guarantee is that the records it was asked to delete did
not reach the parameters it produced. Carrying that across invocations needs one
more fact: the state an invocation resumes from must be free of everything
deleted before it, not only of the current request. Otherwise a later deletion
can resume from a checkpoint that predates an earlier one and restore it.

The record is what makes that checkable. Entries are keyed by *record hash* and
never by row index, because the compaction daemon renumbers rows; a hash still
identifies the same record after compaction. The body may live in host storage:
only the chain head is enclave-resident, and a body that does not hash to the
retained head is refused.
"""
from . import commit

TAG = b"CLEAN-DLOG-v1"


def genesis(shard_id: int) -> bytes:
    """Head of an empty record."""
    return commit.sha256(TAG + b"genesis" + int(shard_id).to_bytes(8, "big"))


def entry_root(record_hashes) -> bytes:
    """Commitment to the set of records removed by one invocation."""
    leaves = sorted(bytes(h) for h in record_hashes)
    return commit.merkle_root(leaves)


def extend(head: bytes, record_hashes) -> bytes:
    """Head after appending one entry."""
    return commit.sha256(TAG + bytes(head) + entry_root(record_hashes))


def head_of(shard_id: int, entries) -> bytes:
    """Fold a record body into the head it must match."""
    head = genesis(shard_id)
    for entry in entries:
        head = extend(head, entry)
    return head


def deleted_hashes(entries) -> set:
    """Every record this shard has been asked to forget so far."""
    out = set()
    for entry in entries:
        out.update(bytes(h) for h in entry)
    return out


def entries_after(shard_id, entries, head) -> list:
    """The entries appended after a certificate bound `head`.

    A checkpoint is trained on the data that survived the deletions recorded
    when it was issued, so the ones it can still be carrying are exactly the
    ones appended later. Returns None when `head` is not a prefix of this
    record, which is what a truncated or forked record looks like.
    """
    running = genesis(shard_id)
    if running == bytes(head):
        return list(entries)
    for i, entry in enumerate(entries):
        running = extend(running, entry)
        if running == bytes(head):
            return list(entries[i + 1:])
    return None
