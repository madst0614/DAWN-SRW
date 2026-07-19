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
        global_max = jax.lax.stop_gradient(
            jax.lax.pmax(jax.lax.stop_gradient(local_max), "model"))

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
        global_pred_val = jax.lax.stop_gradient(
            jax.lax.pmax(jax.lax.stop_gradient(local_pred_val), "model"))
        correct_local_data = jnp.sum(
            ((target_logit == global_pred_val) & valid).astype(jnp.int32))

        loss_sum = jax.lax.psum(loss_sum_local_data, "data")
        correct = jax.lax.psum(correct_local_data, "data")
        valid_count = jax.lax.psum(valid_count_local_data, "data")

        loss = loss_sum / (valid_count.astype(jnp.float32) + 1.0e-8)
        return loss, correct, valid_count

    return vocab_parallel_ce


def make_vocab_parallel_ce(
    mesh,
    logical_vocab_size: int,
    vocab_size_padded: int,
    token_chunk_size: int = 32768,
    compute_accuracy: bool = True,
    compute_logit_stats: bool = True,
):
    """Exact vocab-parallel CE over a row-sharded tied embedding table.

    Returns loss/per-token CE/correct/count plus logit diagnostics.  Padded
    vocabulary rows are excluded from the loss, argmax, and diagnostics.
    """
    mesh_model = int(mesh.shape["model"])
    logical_vocab_size = int(logical_vocab_size)
    vocab_size_padded = int(vocab_size_padded)
    token_chunk_size = int(token_chunk_size)
    compute_accuracy = bool(compute_accuracy)
    compute_logit_stats = bool(compute_logit_stats)
    if logical_vocab_size <= 0:
        raise ValueError(
            f"logical_vocab_size must be > 0, got {logical_vocab_size}")
    if token_chunk_size <= 0:
        raise ValueError(
            f"token_chunk_size must be > 0, got {token_chunk_size}")
    if vocab_size_padded < logical_vocab_size:
        raise ValueError(
            f"vocab_size_padded={vocab_size_padded} must be >= "
            f"logical_vocab_size={logical_vocab_size}")
    if vocab_size_padded % mesh_model != 0:
        raise ValueError(
            f"vocab_size_padded={vocab_size_padded} must be divisible by "
            f"mesh_model={mesh_model}")
    vocab_per_shard = vocab_size_padded // mesh_model

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("data", None, None),  # shift_x [B, T, D]
            P("model", None),       # embedding local [V_local, D]
            P("data", None),        # labels [B, T]
            P("data", None),        # valid_mask [B, T]
        ),
        out_specs=(P(), P("data", None), P(), P(), P(), P(), P(), P()),
        check_rep=False,
    )
    def vocab_parallel_ce(shift_x, embedding_local, shift_labels, valid_mask):
        model_idx = jax.lax.axis_index("model")
        vocab_start = model_idx * vocab_per_shard
        local_vocab_ids = (
            vocab_start
            + jnp.arange(vocab_per_shard, dtype=jnp.int32))
        valid_vocab = local_vocab_ids < logical_vocab_size
        neg_inf = jnp.finfo(jnp.float32).min

        B, T, D = shift_x.shape
        flat_x = shift_x.reshape(B * T, D)
        flat_labels = shift_labels.reshape(B * T).astype(jnp.int32)
        flat_valid = valid_mask.reshape(B * T).astype(jnp.bool_)

        n_tokens = flat_x.shape[0]
        pad = (-n_tokens) % token_chunk_size
        flat_x = jnp.pad(flat_x, ((0, pad), (0, 0)))
        flat_labels = jnp.pad(flat_labels, ((0, pad),), constant_values=0)
        flat_valid = jnp.pad(
            flat_valid, ((0, pad),), constant_values=False)

        flat_x = flat_x.reshape(-1, token_chunk_size, D)
        flat_labels = flat_labels.reshape(-1, token_chunk_size)
        flat_valid = flat_valid.reshape(-1, token_chunk_size)

        def chunk_step(carry, xs):
            del carry
            x_c, labels_c, valid_c = xs
            local_logits = (x_c @ embedding_local.T).astype(jnp.float32)
            local_logits = jnp.where(
                valid_vocab[None, :], local_logits, neg_inf)

            local_max = jnp.max(local_logits, axis=-1)
            global_max = jax.lax.stop_gradient(
                jax.lax.pmax(jax.lax.stop_gradient(local_max), "model"))
            local_exp_sum = jnp.sum(
                jnp.exp(local_logits - global_max[:, None]), axis=-1)
            global_exp_sum = jax.lax.psum(local_exp_sum, "model")
            log_z = global_max + jnp.log(global_exp_sum + 1.0e-30)

            safe_labels = jnp.where(valid_c, labels_c, 0)
            in_local = (
                (safe_labels >= vocab_start)
                & (safe_labels < vocab_start + vocab_per_shard)
                & (safe_labels < logical_vocab_size)
                & valid_c)
            local_idx = safe_labels - vocab_start
            local_idx_safe = jnp.clip(local_idx, 0, vocab_per_shard - 1)
            local_target = jnp.take_along_axis(
                local_logits, local_idx_safe[:, None], axis=-1).squeeze(-1)
            local_target = jnp.where(in_local, local_target, 0.0)
            target_logit = jax.lax.psum(local_target, "model")

            token_ce = (log_z - target_logit).astype(jnp.float32)
            token_ce = jnp.where(valid_c, token_ce, 0.0)

            if compute_accuracy:
                local_best_score = jnp.max(local_logits, axis=-1)
                local_best_idx = jnp.argmax(local_logits, axis=-1)
                local_best_global_id = vocab_start + local_best_idx
                global_best_score = jax.lax.stop_gradient(
                    jax.lax.pmax(
                        jax.lax.stop_gradient(local_best_score), "model"))
                is_winner = local_best_score == global_best_score
                candidate_id = jnp.where(
                    is_winner,
                    local_best_global_id,
                    jnp.asarray(logical_vocab_size + 1000000000,
                                dtype=jnp.int32))
                pred = jax.lax.pmin(candidate_id, "model")
                correct = jnp.sum(
                    ((pred == labels_c) & valid_c).astype(jnp.int32))
            else:
                correct = jnp.array(0, dtype=jnp.int32)

            if compute_logit_stats:
                valid_2d = valid_c[:, None] & valid_vocab[None, :]
                logits_for_sum = jnp.where(valid_2d, local_logits, 0.0)
                local_logit_sum = jnp.sum(logits_for_sum)
                local_logit_sumsq = jnp.sum(logits_for_sum * logits_for_sum)
                local_abs_max = jnp.max(jnp.where(
                    valid_2d, jnp.abs(local_logits), 0.0))
                local_token_sumsq = jnp.sum(
                    jnp.where(valid_vocab[None, :],
                              local_logits * local_logits, 0.0),
                    axis=-1)
                global_token_sumsq = jax.lax.psum(
                    local_token_sumsq, "model")
                logit_norm_sum = jnp.sum(
                    jnp.where(
                        valid_c,
                        jnp.sqrt(jnp.maximum(global_token_sumsq, 0.0)),
                        0.0))
            else:
                local_logit_sum = jnp.array(0.0, dtype=jnp.float32)
                local_logit_sumsq = jnp.array(0.0, dtype=jnp.float32)
                local_abs_max = jnp.array(0.0, dtype=jnp.float32)
                logit_norm_sum = jnp.array(0.0, dtype=jnp.float32)

            valid_count = jnp.sum(valid_c.astype(jnp.int32))
            return None, (
                token_ce,
                correct,
                valid_count,
                jnp.sum(token_ce),
                local_abs_max,
                local_logit_sum,
                local_logit_sumsq,
                logit_norm_sum,
            )

        _, ys = jax.lax.scan(
            chunk_step, None, (flat_x, flat_labels, flat_valid))
        (token_ce_chunks, correct_chunks, valid_chunks, loss_chunks,
         abs_max_chunks, logit_sum_chunks, logit_sumsq_chunks,
         logit_norm_chunks) = ys

        per_token_ce_flat = token_ce_chunks.reshape(-1)[:n_tokens]
        per_token_ce = per_token_ce_flat.reshape(B, T)

        loss_sum_local_data = jnp.sum(loss_chunks)
        valid_sum_local_data = jnp.sum(valid_chunks)
        correct_local_data = jnp.sum(correct_chunks)

        loss_sum_global = jax.lax.psum(loss_sum_local_data, "data")
        valid_count_global = jax.lax.psum(valid_sum_local_data, "data")
        correct_global = jax.lax.psum(correct_local_data, "data")

        loss = (
            loss_sum_global
            / (valid_count_global.astype(jnp.float32) + 1.0e-8))
        loss = jax.lax.pmean(loss, "model")
        correct = jax.lax.pmean(
            correct_global.astype(jnp.float32), "model").astype(jnp.int32)
        valid_count = jax.lax.pmean(
            valid_count_global.astype(jnp.float32),
            "model").astype(jnp.int32)

        local_abs_max = jnp.max(abs_max_chunks)
        logit_abs_max = jax.lax.stop_gradient(
            jax.lax.pmax(
                jax.lax.pmax(jax.lax.stop_gradient(local_abs_max), "data"),
                "model"))
        logit_sum = jax.lax.psum(
            jax.lax.psum(jnp.sum(logit_sum_chunks), "data"), "model")
        logit_sumsq = jax.lax.psum(
            jax.lax.psum(jnp.sum(logit_sumsq_chunks), "data"), "model")
        logit_norm_sum = jax.lax.psum(
            jax.lax.pmean(jnp.sum(logit_norm_chunks), "model"), "data")
        diag_count = (
            valid_count.astype(jnp.float32)
            * jnp.asarray(logical_vocab_size, dtype=jnp.float32))
        logit_mean = logit_sum / (diag_count + 1.0e-8)
        logit_var = (
            logit_sumsq / (diag_count + 1.0e-8)
            - logit_mean * logit_mean)
        logit_std = jnp.sqrt(jnp.maximum(logit_var, 0.0))
        logit_norm_mean = (
            logit_norm_sum
            / (valid_count.astype(jnp.float32) + 1.0e-8))
        if not compute_logit_stats:
            logit_abs_max = jnp.array(0.0, dtype=jnp.float32)
            logit_norm_mean = jnp.array(0.0, dtype=jnp.float32)
            logit_mean = jnp.array(0.0, dtype=jnp.float32)
            logit_std = jnp.array(0.0, dtype=jnp.float32)

        return (
            loss,
            per_token_ce,
            correct,
            valid_count,
            logit_abs_max,
            logit_norm_mean,
            logit_mean,
            logit_std,
        )

    return vocab_parallel_ce


def make_vocab_parallel_ce_loss(
    mesh,
    logical_vocab_size: int,
    vocab_size_padded: int,
    token_chunk_size: int = 32768,
):
    """Static training-fast CE profile returning only scalar loss."""
    full_ce = make_vocab_parallel_ce(
        mesh,
        logical_vocab_size=logical_vocab_size,
        vocab_size_padded=vocab_size_padded,
        token_chunk_size=token_chunk_size,
        compute_accuracy=False,
        compute_logit_stats=False,
    )

    def vocab_parallel_ce_loss(
            shift_x, embedding_local, shift_labels, valid_mask):
        return full_ce(
            shift_x, embedding_local, shift_labels, valid_mask)[0]

    return vocab_parallel_ce_loss


def make_vocab_parallel_eval_stats(
    mesh,
    logical_vocab_size: int,
    vocab_size_padded: int,
    token_chunk_size: int = 32768,
):
    """Exact per-token NLL and greedy matches for read-only evaluation.

    Unlike :func:`make_vocab_parallel_ce`, this function deliberately performs
    no data-axis reduction.  The caller needs one score and one greedy decision
    per example, so both outputs remain sharded over ``data``.  The model-axis
    collectives are exact and padded vocabulary rows can never win argmax.
    """
    mesh_model = int(mesh.shape["model"])
    logical_vocab_size = int(logical_vocab_size)
    vocab_size_padded = int(vocab_size_padded)
    token_chunk_size = int(token_chunk_size)
    if logical_vocab_size <= 0:
        raise ValueError(
            f"logical_vocab_size must be > 0, got {logical_vocab_size}")
    if token_chunk_size <= 0:
        raise ValueError(
            f"token_chunk_size must be > 0, got {token_chunk_size}")
    if vocab_size_padded < logical_vocab_size:
        raise ValueError(
            f"vocab_size_padded={vocab_size_padded} must be >= "
            f"logical_vocab_size={logical_vocab_size}")
    if vocab_size_padded % mesh_model != 0:
        raise ValueError(
            f"vocab_size_padded={vocab_size_padded} must be divisible by "
            f"mesh_model={mesh_model}")
    vocab_per_shard = vocab_size_padded // mesh_model

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("data", None, None),
            P("model", None),
            P("data", None),
            P("data", None),
        ),
        out_specs=(P("data", None), P("data", None)),
        check_rep=False,
    )
    def eval_stats(shift_x, embedding_local, shift_labels, valid_mask):
        model_idx = jax.lax.axis_index("model")
        vocab_start = model_idx * vocab_per_shard
        local_vocab_ids = (
            vocab_start
            + jnp.arange(vocab_per_shard, dtype=jnp.int32))
        valid_vocab = local_vocab_ids < logical_vocab_size
        neg_inf = jnp.finfo(jnp.float32).min

        batch, seq, width = shift_x.shape
        n_tokens = batch * seq
        flat_x = shift_x.reshape(n_tokens, width)
        flat_labels = shift_labels.reshape(n_tokens).astype(jnp.int32)
        flat_valid = valid_mask.reshape(n_tokens).astype(jnp.bool_)
        pad = (-n_tokens) % token_chunk_size
        flat_x = jnp.pad(flat_x, ((0, pad), (0, 0)))
        flat_labels = jnp.pad(flat_labels, ((0, pad),), constant_values=0)
        flat_valid = jnp.pad(
            flat_valid, ((0, pad),), constant_values=False)
        flat_x = flat_x.reshape(-1, token_chunk_size, width)
        flat_labels = flat_labels.reshape(-1, token_chunk_size)
        flat_valid = flat_valid.reshape(-1, token_chunk_size)

        def chunk_step(carry, xs):
            del carry
            x_c, labels_c, valid_c = xs
            local_logits = (x_c @ embedding_local.T).astype(jnp.float32)
            local_logits = jnp.where(
                valid_vocab[None, :], local_logits, neg_inf)

            local_max = jnp.max(local_logits, axis=-1)
            global_max = jax.lax.stop_gradient(
                jax.lax.pmax(jax.lax.stop_gradient(local_max), "model"))
            local_exp_sum = jnp.sum(
                jnp.exp(local_logits - global_max[:, None]), axis=-1)
            log_z = global_max + jnp.log(
                jax.lax.psum(local_exp_sum, "model") + 1.0e-30)

            safe_labels = jnp.where(valid_c, labels_c, 0)
            in_local = (
                (safe_labels >= vocab_start)
                & (safe_labels < vocab_start + vocab_per_shard)
                & (safe_labels < logical_vocab_size)
                & valid_c)
            local_idx = jnp.clip(
                safe_labels - vocab_start, 0, vocab_per_shard - 1)
            local_target = jnp.take_along_axis(
                local_logits, local_idx[:, None], axis=-1).squeeze(-1)
            target_logit = jax.lax.psum(
                jnp.where(in_local, local_target, 0.0), "model")
            token_ce = jnp.where(
                valid_c, (log_z - target_logit).astype(jnp.float32), 0.0)

            local_best_idx = jnp.argmax(local_logits, axis=-1)
            local_best_score = jnp.max(local_logits, axis=-1)
            global_best_score = jax.lax.stop_gradient(
                jax.lax.pmax(
                    jax.lax.stop_gradient(local_best_score), "model"))
            sentinel = jnp.asarray(
                logical_vocab_size + 1000000000, dtype=jnp.int32)
            candidate_id = jnp.where(
                local_best_score == global_best_score,
                vocab_start + local_best_idx,
                sentinel,
            )
            pred = jax.lax.pmin(candidate_id, "model")
            token_correct = (pred == labels_c) & valid_c
            return None, (token_ce, token_correct)

        _, (ce_chunks, correct_chunks) = jax.lax.scan(
            chunk_step, None, (flat_x, flat_labels, flat_valid))
        per_token_ce = ce_chunks.reshape(-1)[:n_tokens].reshape(batch, seq)
        per_token_correct = (
            correct_chunks.reshape(-1)[:n_tokens].reshape(batch, seq))
        return per_token_ce, per_token_correct

    return eval_stats


def make_vocab_parallel_argmax(
    mesh,
    logical_vocab_size: int,
    vocab_size_padded: int,
    token_chunk_size: int = 32768,
):
    """Return exact global-vocabulary argmax ids without gathering weights."""
    mesh_model = int(mesh.shape["model"])
    logical_vocab_size = int(logical_vocab_size)
    vocab_size_padded = int(vocab_size_padded)
    token_chunk_size = int(token_chunk_size)
    if logical_vocab_size <= 0 or token_chunk_size <= 0:
        raise ValueError("logical_vocab_size and token_chunk_size must be > 0")
    if vocab_size_padded < logical_vocab_size:
        raise ValueError("vocab_size_padded is smaller than logical_vocab_size")
    if vocab_size_padded % mesh_model != 0:
        raise ValueError("vocab_size_padded must be divisible by mesh_model")
    vocab_per_shard = vocab_size_padded // mesh_model

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(P("data", None, None), P("model", None)),
        out_specs=P("data", None),
        check_rep=False,
    )
    def vocab_argmax(hidden, embedding_local):
        model_idx = jax.lax.axis_index("model")
        vocab_start = model_idx * vocab_per_shard
        vocab_ids = vocab_start + jnp.arange(
            vocab_per_shard, dtype=jnp.int32)
        valid_vocab = vocab_ids < logical_vocab_size
        neg_inf = jnp.finfo(jnp.float32).min
        batch, seq, width = hidden.shape
        n_tokens = batch * seq
        flat = hidden.reshape(n_tokens, width)
        pad = (-n_tokens) % token_chunk_size
        flat = jnp.pad(flat, ((0, pad), (0, 0)))
        flat = flat.reshape(-1, token_chunk_size, width)

        def chunk_step(carry, x_c):
            del carry
            logits = (x_c @ embedding_local.T).astype(jnp.float32)
            logits = jnp.where(valid_vocab[None, :], logits, neg_inf)
            local_idx = jnp.argmax(logits, axis=-1)
            local_score = jnp.max(logits, axis=-1)
            global_score = jax.lax.stop_gradient(
                jax.lax.pmax(jax.lax.stop_gradient(local_score), "model"))
            sentinel = jnp.asarray(
                logical_vocab_size + 1000000000, dtype=jnp.int32)
            candidate = jnp.where(
                local_score == global_score,
                vocab_start + local_idx,
                sentinel,
            )
            return None, jax.lax.pmin(candidate, "model")

        _, chunks = jax.lax.scan(chunk_step, None, flat)
        return chunks.reshape(-1)[:n_tokens].reshape(batch, seq)

    return vocab_argmax


def make_unsharded_eval_stats(
    logical_vocab_size: int,
    token_chunk_size: int = 32768,
):
    """Single-model-axis counterpart of ``make_vocab_parallel_eval_stats``."""
    logical_vocab_size = int(logical_vocab_size)
    token_chunk_size = int(token_chunk_size)
    if logical_vocab_size <= 0 or token_chunk_size <= 0:
        raise ValueError("logical_vocab_size and token_chunk_size must be > 0")

    def eval_stats(shift_x, embedding, shift_labels, valid_mask):
        batch, seq, width = shift_x.shape
        n_tokens = batch * seq
        flat_x = shift_x.reshape(n_tokens, width)
        flat_labels = shift_labels.reshape(n_tokens).astype(jnp.int32)
        flat_valid = valid_mask.reshape(n_tokens).astype(jnp.bool_)
        pad = (-n_tokens) % token_chunk_size
        flat_x = jnp.pad(flat_x, ((0, pad), (0, 0)))
        flat_labels = jnp.pad(flat_labels, ((0, pad),), constant_values=0)
        flat_valid = jnp.pad(
            flat_valid, ((0, pad),), constant_values=False)
        flat_x = flat_x.reshape(-1, token_chunk_size, width)
        flat_labels = flat_labels.reshape(-1, token_chunk_size)
        flat_valid = flat_valid.reshape(-1, token_chunk_size)
        vocab_ids = jnp.arange(embedding.shape[0], dtype=jnp.int32)
        valid_vocab = vocab_ids < logical_vocab_size
        neg_inf = jnp.finfo(jnp.float32).min

        def chunk_step(carry, xs):
            del carry
            x_c, labels_c, valid_c = xs
            logits = (x_c @ embedding.T).astype(jnp.float32)
            logits = jnp.where(valid_vocab[None, :], logits, neg_inf)
            safe = jnp.where(valid_c, labels_c, 0)
            target = jnp.take_along_axis(
                logits, safe[:, None], axis=-1).squeeze(-1)
            ce = jax.nn.logsumexp(logits, axis=-1) - target
            ce = jnp.where(valid_c, ce.astype(jnp.float32), 0.0)
            correct = (jnp.argmax(logits, axis=-1) == labels_c) & valid_c
            return None, (ce, correct)

        _, (ce_chunks, correct_chunks) = jax.lax.scan(
            chunk_step, None, (flat_x, flat_labels, flat_valid))
        return (
            ce_chunks.reshape(-1)[:n_tokens].reshape(batch, seq),
            correct_chunks.reshape(-1)[:n_tokens].reshape(batch, seq),
        )

    return eval_stats


def make_unsharded_argmax(
    logical_vocab_size: int,
    token_chunk_size: int = 32768,
):
    """Single-model-axis exact argmax over a tied embedding table."""
    logical_vocab_size = int(logical_vocab_size)
    token_chunk_size = int(token_chunk_size)
    if logical_vocab_size <= 0 or token_chunk_size <= 0:
        raise ValueError("logical_vocab_size and token_chunk_size must be > 0")

    def vocab_argmax(hidden, embedding):
        batch, seq, width = hidden.shape
        n_tokens = batch * seq
        flat = hidden.reshape(n_tokens, width)
        pad = (-n_tokens) % token_chunk_size
        flat = jnp.pad(flat, ((0, pad), (0, 0)))
        flat = flat.reshape(-1, token_chunk_size, width)
        vocab_ids = jnp.arange(embedding.shape[0], dtype=jnp.int32)
        valid_vocab = vocab_ids < logical_vocab_size
        neg_inf = jnp.finfo(jnp.float32).min

        def chunk_step(carry, x_c):
            del carry
            logits = (x_c @ embedding.T).astype(jnp.float32)
            logits = jnp.where(valid_vocab[None, :], logits, neg_inf)
            return None, jnp.argmax(logits, axis=-1).astype(jnp.int32)

        _, chunks = jax.lax.scan(chunk_step, None, flat)
        return chunks.reshape(-1)[:n_tokens].reshape(batch, seq)

    return vocab_argmax
