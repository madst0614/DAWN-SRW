import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P


def padded_vocab_size(vocab_size, mesh_model):
    vocab_size = int(vocab_size)
    mesh_model = int(mesh_model)
    if mesh_model <= 0:
        raise ValueError(f"mesh_model must be > 0, got {mesh_model}")
    return int(math.ceil(vocab_size / mesh_model) * mesh_model)


def make_vocab_parallel_embedding(mesh, vocab_size, padded_vocab_size):
    mesh_model = int(mesh.shape["model"])
    vocab_size = int(vocab_size)
    padded_vocab_size = int(padded_vocab_size)
    if padded_vocab_size % mesh_model != 0:
        raise ValueError(
            f"padded_vocab_size={padded_vocab_size} must be divisible by "
            f"mesh_model={mesh_model}")
    vocab_per_shard = padded_vocab_size // mesh_model

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("data", None),   # input_ids [B, S]
            P("model", None),  # embedding local [V_local, D]
        ),
        out_specs=P("data", None, None),
        check_rep=False,
    )
    def vocab_parallel_embedding(input_ids, embedding_local):
        axis_idx = jax.lax.axis_index("model")
        vocab_start = axis_idx * vocab_per_shard
        vocab_end = vocab_start + vocab_per_shard

        ids = input_ids.astype(jnp.int32)
        in_local = (
            (ids >= vocab_start) & (ids < vocab_end) & (ids < vocab_size))
        local_ids = jnp.clip(ids - vocab_start, 0, vocab_per_shard - 1)

        local_emb = embedding_local[local_ids]
        local_emb = jnp.where(in_local[..., None], local_emb, 0.0)
        return jax.lax.psum(local_emb, "model")

    return vocab_parallel_embedding


def make_vocab_parallel_cross_entropy(mesh, vocab_size, padded_vocab_size):
    mesh_model = int(mesh.shape["model"])
    vocab_size = int(vocab_size)
    padded_vocab_size = int(padded_vocab_size)
    if padded_vocab_size % mesh_model != 0:
        raise ValueError(
            f"padded_vocab_size={padded_vocab_size} must be divisible by "
            f"mesh_model={mesh_model}")
    vocab_per_shard = padded_vocab_size // mesh_model

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("data", None, None),  # hidden [B, T, D]
            P("model", None),       # embedding local [V_local, D]
            P("data", None),        # labels [B, T]
            P("data", None),        # valid_mask [B, T]
        ),
        out_specs=(P(), P(), P()),
        check_rep=False,
    )
    def vocab_parallel_ce(hidden, embedding_local, labels, valid_mask):
        axis_idx = jax.lax.axis_index("model")
        vocab_start = axis_idx * vocab_per_shard
        vocab_ids = vocab_start + jnp.arange(vocab_per_shard, dtype=jnp.int32)
        vocab_valid = vocab_ids < vocab_size

        h = hidden.astype(jnp.bfloat16)
        w = embedding_local.astype(jnp.bfloat16)
        logits_local = jnp.einsum("btd,vd->btv", h, w).astype(jnp.float32)

        neg_inf = jnp.finfo(jnp.float32).min
        logits_local = jnp.where(
            vocab_valid[None, None, :], logits_local, neg_inf)

        local_max = jnp.max(logits_local, axis=-1)
        global_max = jax.lax.pmax(jax.lax.stop_gradient(local_max), "model")

        exp_local = jnp.exp(logits_local - global_max[..., None])
        exp_local = jnp.where(vocab_valid[None, None, :], exp_local, 0.0)
        local_denom = jnp.sum(exp_local, axis=-1)
        global_denom = jax.lax.psum(local_denom, "model")
        log_z = jnp.log(global_denom + 1.0e-30) + global_max

        labels = labels.astype(jnp.int32)
        valid = valid_mask.astype(jnp.bool_)
        in_local = (
            (labels >= vocab_start)
            & (labels < vocab_start + vocab_per_shard)
            & (labels < vocab_size)
            & valid)
        local_label = jnp.clip(labels - vocab_start, 0, vocab_per_shard - 1)
        target_local = jnp.take_along_axis(
            logits_local, local_label[..., None], axis=-1).squeeze(-1)
        target_local = jnp.where(in_local, target_local, 0.0)
        target_logit = jax.lax.psum(target_local, "model")

        token_loss = log_z - target_logit
        valid_f = valid.astype(jnp.float32)
        loss_sum_local_data = jnp.sum(token_loss * valid_f)
        valid_count_local_data = jnp.sum(valid.astype(jnp.int32))

        local_pred_val = jnp.max(logits_local, axis=-1)
        global_pred_val = jax.lax.pmax(jax.lax.stop_gradient(local_pred_val), "model")
        correct_local_data = jnp.sum(
            ((target_logit == global_pred_val) & valid).astype(jnp.int32))

        loss_sum = jax.lax.psum(loss_sum_local_data, "data")
        correct = jax.lax.psum(correct_local_data, "data")
        valid_count = jax.lax.psum(valid_count_local_data, "data")

        loss = loss_sum / (valid_count.astype(jnp.float32) + 1.0e-8)
        return loss, correct, valid_count

    return vocab_parallel_ce
