import numpy as np

from clean_runtime import commit


def test_tensor_bytes_header_distinguishes_shape():
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.zeros((3, 2), dtype=np.float32)
    assert commit.tensor_bytes(a) != commit.tensor_bytes(b)
    assert commit.sha256(commit.tensor_bytes(a)) == commit.sha256(commit.tensor_bytes(a.copy()))


def test_params_hash_key_order_invariant():
    p1 = {"b": np.ones(2, dtype=np.float32), "a": np.zeros(3, dtype=np.float32)}
    p2 = dict(reversed(list(p1.items())))
    assert commit.params_hash(p1) == commit.params_hash(p2)


def test_merkle_root_cases():
    h = commit.sha256
    assert commit.merkle_root([]) == h(b"")
    leaf = h(b"x")
    assert commit.merkle_root([leaf]) == leaf
    l3 = [h(b"a"), h(b"b"), h(b"c")]
    # odd node pairs with itself
    expected = h(h(l3[0] + l3[1]) + h(l3[2] + l3[2]))
    assert commit.merkle_root(l3) == expected


def test_dataset_root_row_order_invariant():
    idx = np.array([5, 1, 9], dtype=np.uint32)
    salts = [bytes([i]) * 16 for i in range(3)]
    payloads = [bytes([i]) * 8 for i in range(3)]
    r1 = commit.dataset_root(idx, salts, payloads)
    perm = [2, 0, 1]
    r2 = commit.dataset_root(idx[perm], [salts[i] for i in perm],
                             [payloads[i] for i in perm])
    assert r1 == r2


def test_encode_signed_message_unambiguous():
    m1 = commit.encode_signed_message(b"T", [b"ab", b"c"])
    m2 = commit.encode_signed_message(b"T", [b"a", b"bc"])
    assert m1 != m2
