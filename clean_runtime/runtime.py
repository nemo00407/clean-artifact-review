"""Closed-input unlearning state transition (spec sections 1-2).

    W_new = R(CP_clean, D_clean, r, c)

The runtime receives a frozen Invocation from the gateway, retrains the
LoRA adapter from the clean checkpoint over the physically excluded
dataset, and assembles the ARC binding fields. Certificate signing is
performed by the attest module strictly after the state transition
completes (Stage T3); until then `signature` is None.
"""
from . import commit, kernel, loader, store


def arc_fields(inv, c_base, h_l_ckpt, h_l_new):
    """Binding fields of the Attested Retraining Certificate, in the
    canonical field order of spec section 4.4."""
    return {
        "c_base": c_base,
        "h_l_ckpt": h_l_ckpt,
        "h_l_new": h_l_new,
        "h_dclean": inv.data.h_dclean,
        "h_delta": inv.h_delta,
        "h_record": inv.h_record,
        "h_r": commit.sha256(inv.rng.r),
        "h_config": inv.h_config,
        "shard_id": int(inv.cp.meta["shard_id"]),
        "cp_version": int(inv.cp.meta["version"]),
        "nonce": inv.nonce,
    }


def _certify(inv, enclave, lora, slice_idx, version):
    """Issue a checkpoint certificate for the state at a slice boundary."""
    h_cp = commit.params_hash(lora)
    h_prefix = inv.data.prefix_roots[slice_idx]
    cert = enclave.sign_cp_cert(h_cp, h_prefix, inv.cp.meta["shard_id"],
                                slice_idx, version, inv.h_record)
    return {"shard_id": int(inv.cp.meta["shard_id"]),
            "slice_idx": int(slice_idx), "version": int(version),
            "h_d_prefix": h_prefix.hex(), "h_record": inv.h_record.hex(),
            "cert": cert.hex(), "lora": lora}


def unlearn(inv, enclave=None):
    """Execute the deletion-isolated retraining state transition.

    Returns (new_lora, arc, losses). The ARC signature is issued by the
    enclave strictly after the state transition completes; without an
    enclave (unit tests) the unsigned binding fields are returned."""
    base = loader.base_params(inv.base)
    lora = loader.checkpoint_lora(inv.cp)
    tokens = loader.training_tokens(inv.data)
    config = dict(inv.config)

    h_l_ckpt = commit.params_hash(lora)

    # The deletion joins the shard's record before anything is certified, so
    # every certificate this invocation issues names the record it enforced.
    if enclave is not None:
        inv.h_record = enclave.append_deletion(
            inv.cp.meta["shard_id"], inv.delta_hashes)

    spans = inv.data.slice_spans
    if spans:
        # Retrain the way the shard was built: slice by slice, certifying the
        # state at each boundary, so a later deletion can resume from a recent
        # checkpoint whose prefix is already free of this one.
        losses = []
        for j, lo, hi in spans:
            lora, ls = kernel.train(
                base, lora, store.RowWindow(tokens, lo, hi), inv.rng, config)
            losses.extend(ls)
            if enclave is not None:
                version = enclave.next_version(inv.cp.meta["shard_id"])
                inv.issued.append(_certify(inv, enclave, lora, j, version))
        new_lora = lora
    else:
        new_lora, losses = kernel.train(base, lora, tokens, inv.rng, config)

    c_base = commit.params_hash(base)
    arc = arc_fields(inv, c_base, h_l_ckpt, commit.params_hash(new_lora))
    if enclave is not None:
        arc["mrenclave"] = enclave.mrenclave
        arc["signature"] = enclave.sign_arc(arc)
    else:
        arc["mrenclave"] = None
        arc["signature"] = None
    return new_lora, arc, losses
