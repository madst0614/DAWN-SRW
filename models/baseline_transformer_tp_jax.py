"""Tensor-parallel vanilla Transformer baseline for JAX/Flax.

This model keeps the same public training interface and parameter structure as
``VanillaTransformer`` while letting the trainer shard attention and FFN dense
matrices over the mesh ``model`` axis.
"""

from functools import partial

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from models.baseline_transformer_jax import scaled_normal


def _check_divisible(name, value, divisor):
    if int(value) % int(divisor) != 0:
        raise ValueError(
            f"{name}={value} must be divisible by mesh_model={divisor} "
            "for baseline-tp tensor parallelism.")


def _local_causal_attention(q, k, v, d_head):
    """Naive causal attention over local heads.

    Args:
        q, k, v: [B, H_local, S, d_head]
    """
    scale = jnp.sqrt(jnp.float32(d_head))
    scores = jnp.einsum("bhsd,bhtd->bhst", q, k) / scale
    seq_len = q.shape[2]
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhst,bhtd->bhsd", attn, v)


def make_baseline_tp_attention(mesh, n_heads, d_model):
    """Create Megatron-style local-head attention over the model axis."""
    mesh_model = int(mesh.shape["model"])
    _check_divisible("n_heads", n_heads, mesh_model)
    _check_divisible("d_model", d_model, mesh_model)
    if int(d_model) % int(n_heads) != 0:
        raise ValueError(
            f"d_model={d_model} must be divisible by n_heads={n_heads}.")

    d_head = int(d_model) // int(n_heads)
    heads_per_shard = int(n_heads) // mesh_model
    d_local = heads_per_shard * d_head

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("data", None, None),  # x [B, S, D]
            P(None, "model"),       # q kernel [D, D_local]
            P("model"),             # q bias [D_local]
            P(None, "model"),
            P("model"),
            P(None, "model"),
            P("model"),
            P("model", None),       # o kernel [D_local, D]
        ),
        out_specs=P("data", None, None),
        check_rep=False,
    )
    def baseline_tp_attention(x, q_kernel, q_bias, k_kernel, k_bias,
                              v_kernel, v_bias, o_kernel):
        batch, seq_len, _ = x.shape
        q = x @ q_kernel + q_bias
        k = x @ k_kernel + k_bias
        v = x @ v_kernel + v_bias

        q = q.reshape(batch, seq_len, heads_per_shard, d_head)
        k = k.reshape(batch, seq_len, heads_per_shard, d_head)
        v = v.reshape(batch, seq_len, heads_per_shard, d_head)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = _local_causal_attention(q, k, v, d_head)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_local)
        partial_out = out @ o_kernel
        return jax.lax.psum(partial_out, "model")

    return baseline_tp_attention


def make_baseline_tp_ffn(mesh, d_ff):
    """Create column-parallel up and row-parallel down FFN."""
    mesh_model = int(mesh.shape["model"])
    _check_divisible("d_ff", d_ff, mesh_model)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("data", None, None),  # x [B, S, D]
            P(None, "model"),       # up kernel [D, d_ff_local]
            P("model"),             # up bias [d_ff_local]
            P("model", None),       # down kernel [d_ff_local, D]
            P(),                     # down bias [D], replicated
        ),
        out_specs=P("data", None, None),
        check_rep=False,
    )
    def baseline_tp_ffn(x, up_kernel, up_bias, down_kernel, down_bias):
        h = x @ up_kernel + up_bias
        h = nn.gelu(h)
        partial_out = h @ down_kernel
        out = jax.lax.psum(partial_out, "model")
        return out + down_bias

    return baseline_tp_ffn


def create_baseline_tp_sharded_fns(mesh, cfg):
    """Build sharded functions required by ``TensorParallelVanillaTransformer``."""
    model_cfg = cfg.get("model", cfg)
    training_cfg = cfg.get("training", {})
    mesh_model = int(mesh.shape["model"])
    d_model = int(model_cfg.get("d_model", 384))
    n_heads = int(model_cfg.get("n_heads", 6))
    d_ff = int(model_cfg.get("d_ff", 1536))
    dropout_rate = float(model_cfg.get("dropout", model_cfg.get(
        "dropout_rate", 0.0)))

    _check_divisible("n_heads", n_heads, mesh_model)
    _check_divisible("d_model", d_model, mesh_model)
    _check_divisible("d_ff", d_ff, mesh_model)
    if dropout_rate > 0.0 and mesh_model > 1:
        raise ValueError(
            "baseline-tp currently supports dropout=0.0 when mesh_model > 1; "
            f"got dropout={dropout_rate} and mesh_model={mesh_model}.")
    if int(training_cfg.get("mesh_model", mesh_model)) != mesh_model:
        raise ValueError(
            "baseline-tp mesh_model mismatch between config and mesh: "
            f"config={training_cfg.get('mesh_model')} mesh={mesh_model}.")

    return {
        "baseline_tp_attention": make_baseline_tp_attention(
            mesh, n_heads=n_heads, d_model=d_model),
        "baseline_tp_ffn": make_baseline_tp_ffn(mesh, d_ff=d_ff),
    }


class TensorParallelAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout_rate: float = 0.0

    def setup(self):
        self.d_head = self.d_model // self.n_heads
        self.q_proj = nn.Dense(self.d_model)
        self.k_proj = nn.Dense(self.d_model)
        self.v_proj = nn.Dense(self.d_model)
        self.o_proj = nn.Dense(self.d_model, use_bias=False)
        self.attn_dropout = nn.Dropout(self.dropout_rate)

    def _standard_attention(self, x, deterministic=False):
        batch, seq_len, dim = x.shape
        q = self.q_proj(x).reshape(
            batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(
            batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(
            batch, seq_len, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scale = jnp.sqrt(jnp.float32(self.d_head))
        scores = jnp.einsum("bhsd,bhtd->bhst", q, k) / scale
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn = jax.nn.softmax(scores, axis=-1)
        attn = self.attn_dropout(attn, deterministic=deterministic)
        out = jnp.einsum("bhst,bhtd->bhsd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, dim)
        return self.o_proj(out)

    def __call__(self, x, deterministic=False, sharded_fns=None):
        fn = None
        if isinstance(sharded_fns, dict):
            fn = sharded_fns.get("baseline_tp_attention")
        if self.is_initializing() or fn is None:
            return self._standard_attention(x, deterministic=deterministic)
        if self.dropout_rate > 0.0 and not deterministic:
            raise ValueError(
                "baseline-tp TP train path requires dropout_rate=0.0; "
                f"got dropout_rate={self.dropout_rate}.")
        params = self.variables["params"]
        return fn(
            x,
            params["q_proj"]["kernel"], params["q_proj"]["bias"],
            params["k_proj"]["kernel"], params["k_proj"]["bias"],
            params["v_proj"]["kernel"], params["v_proj"]["bias"],
            params["o_proj"]["kernel"],
        )


class TensorParallelFFN(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float = 0.0

    def setup(self):
        self.Dense_0 = nn.Dense(self.d_ff)
        self.Dense_1 = nn.Dense(self.d_model)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _standard_ffn(self, x, deterministic=False):
        h = self.Dense_0(x)
        h = nn.gelu(h)
        h = self.dropout(h, deterministic=deterministic)
        return self.Dense_1(h)

    def __call__(self, x, deterministic=False, sharded_fns=None):
        fn = None
        if isinstance(sharded_fns, dict):
            fn = sharded_fns.get("baseline_tp_ffn")
        if self.is_initializing() or fn is None:
            return self._standard_ffn(x, deterministic=deterministic)
        if self.dropout_rate > 0.0 and not deterministic:
            raise ValueError(
                "baseline-tp TP train path requires dropout_rate=0.0; "
                f"got dropout_rate={self.dropout_rate}.")
        params = self.variables["params"]
        return fn(
            x,
            params["Dense_0"]["kernel"], params["Dense_0"]["bias"],
            params["Dense_1"]["kernel"], params["Dense_1"]["bias"],
        )


class TensorParallelTransformerLayer(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.0

    def setup(self):
        self.attn = TensorParallelAttention(
            self.d_model, self.n_heads, self.dropout_rate)
        self.ffn = TensorParallelFFN(
            self.d_model, self.d_ff, self.dropout_rate)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.attn_resid_dropout = nn.Dropout(self.dropout_rate)
        self.ffn_resid_dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic=False, sharded_fns=None):
        normed = self.norm1(x)
        attn_out = self.attn(
            normed, deterministic=deterministic, sharded_fns=sharded_fns)
        attn_out = self.attn_resid_dropout(
            attn_out, deterministic=deterministic)
        x = x + attn_out

        normed = self.norm2(x)
        ffn_out = self.ffn(
            normed, deterministic=deterministic, sharded_fns=sharded_fns)
        ffn_out = self.ffn_resid_dropout(
            ffn_out, deterministic=deterministic)
        return x + ffn_out


class TensorParallelVanillaTransformer(nn.Module):
    """Vanilla Transformer baseline with model-axis tensor parallel kernels."""

    __version__ = "baseline-tp"

    vocab_size: int = 30522
    d_model: int = 432
    d_ff: int = 1728
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.0
    gradient_checkpointing: bool = False

    def setup(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads}).")
        self.token_emb = nn.Embed(
            self.vocab_size, self.d_model, embedding_init=scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model, embedding_init=scaled_normal(0.02))
        layer_cls = TensorParallelTransformerLayer
        if self.gradient_checkpointing:
            layer_cls = nn.remat(
                TensorParallelTransformerLayer, static_argnums=(1, 2))
        self.layers = [
            layer_cls(self.d_model, self.n_heads, self.d_ff,
                      self.dropout_rate, name=f"layer_{i}")
            for i in range(self.n_layers)
        ]
        self.norm = nn.LayerNorm()
        self.emb_dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False, sharded_fns=None):
        del attention_mask
        batch, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len")
        positions = jnp.arange(seq_len)[jnp.newaxis, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x, deterministic=deterministic)

        for layer in self.layers:
            x = layer(x, deterministic, sharded_fns)

        x = self.norm(x)
        result = {"aux_loss": jnp.float32(0.0)}

        if labels is not None:
            embedding_matrix = self.token_emb.embedding
            shift_x = x[:, :-1, :]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            valid_mask = shift_labels != -100

            @jax.checkpoint
            def compute_loss_and_acc(x_chunk, emb, labs, vmask):
                logits = x_chunk @ emb.T
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                safe = jnp.where(vmask, labs, 0)
                token_loss = -jnp.take_along_axis(
                    log_probs, safe[..., jnp.newaxis], axis=-1).squeeze(-1)
                loss = (token_loss * vmask).sum() / (vmask.sum() + 1e-8)
                preds = jnp.argmax(logits, axis=-1)
                correct = jnp.sum((preds == labs) & vmask)
                valid_count = jnp.sum(vmask)
                return loss, correct, valid_count

            loss, correct, valid_count = compute_loss_and_acc(
                shift_x, embedding_matrix, shift_labels, valid_mask)
            result.update({
                "loss": loss,
                "correct": correct,
                "valid_count": valid_count,
            })
        else:
            result["logits"] = self.token_emb.attend(x)

        return result

    def orthogonality_loss(self):
        return jnp.float32(0.0)

    def knowledge_diversity_loss(self):
        return jnp.float32(0.0)

    def get_config(self):
        return {
            "model_version": self.__version__,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "max_seq_len": self.max_seq_len,
        }

    def get_model_info(self):
        ffn_ratio = self.d_ff / self.d_model
        return [
            "  Model: TensorParallelVanillaTransformer (baseline-tp)",
            (f"  d_model={self.d_model}, d_ff={self.d_ff}, "
             f"n_layers={self.n_layers}, n_heads={self.n_heads}"),
            f"  FFN ratio={ffn_ratio:.2f}, dropout={self.dropout_rate}",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
        ]
