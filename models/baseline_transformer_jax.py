"""Vanilla Transformer baseline for JAX/Flax."""
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

from models.vocab_parallel import (
    make_vocab_parallel_cross_entropy,
    make_vocab_parallel_embedding,
    padded_vocab_size,
)


def scaled_normal(scale=0.02):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * scale
    return init


def _check_divisible(name, value, divisor):
    if int(value) % int(divisor) != 0:
        raise ValueError(
            f"{name}={value} must be divisible by mesh_model={divisor} "
            "for baseline tensor parallelism.")


def _local_causal_attention(q, k, v, d_head):
    scale = jnp.sqrt(jnp.float32(d_head))
    scores = jnp.einsum("bhsd,bhtd->bhst", q, k) / scale
    seq_len = q.shape[2]
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
    attn = jax.nn.softmax(scores, axis=-1)
    return jnp.einsum("bhst,bhtd->bhsd", attn, v)


def make_baseline_model_parallel_attention(mesh, n_heads, d_model):
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
            P("data", None, None),
            P(None, "model"),
            P("model"),
            P(None, "model"),
            P("model"),
            P(None, "model"),
            P("model"),
            P("model", None),
        ),
        out_specs=P("data", None, None),
        check_rep=False,
    )
    def baseline_attention(x, q_kernel, q_bias, k_kernel, k_bias,
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

    return baseline_attention


def make_baseline_model_parallel_ffn(mesh, d_ff):
    mesh_model = int(mesh.shape["model"])
    _check_divisible("d_ff", d_ff, mesh_model)

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(
            P("data", None, None),
            P(None, "model"),
            P("model"),
            P("model", None),
            P(),
        ),
        out_specs=P("data", None, None),
        check_rep=False,
    )
    def baseline_ffn(x, up_kernel, up_bias, down_kernel, down_bias):
        h = x @ up_kernel + up_bias
        h = nn.gelu(h)
        partial_out = h @ down_kernel
        out = jax.lax.psum(partial_out, "model")
        return out + down_bias

    return baseline_ffn


def create_baseline_sharded_fns(mesh, cfg):
    """Build optional tensor/vocab-parallel helpers for VanillaTransformer."""
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
            "baseline tensor-parallel train path requires dropout=0.0 "
            f"when mesh_model > 1; got dropout={dropout_rate}.")
    if int(training_cfg.get("mesh_model", mesh_model)) != mesh_model:
        raise ValueError(
            "baseline mesh_model mismatch between config and mesh: "
            f"config={training_cfg.get('mesh_model')} mesh={mesh_model}.")

    logical_vocab_size = int(model_cfg.get(
        "logical_vocab_size", model_cfg.get("vocab_size", 30522)))
    padded_vocab = int(model_cfg.get(
        "vocab_size_padded",
        padded_vocab_size(logical_vocab_size, mesh_model)))

    return {
        "baseline_attention": make_baseline_model_parallel_attention(
            mesh, n_heads=n_heads, d_model=d_model),
        "baseline_ffn": make_baseline_model_parallel_ffn(mesh, d_ff=d_ff),
        "vocab_parallel_embedding": make_vocab_parallel_embedding(
            mesh, logical_vocab_size, padded_vocab),
        "vocab_parallel_ce": make_vocab_parallel_cross_entropy(
            mesh, logical_vocab_size, padded_vocab),
    }


class StandardAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.d_head = self.d_model // self.n_heads
        self.q_proj = nn.Dense(self.d_model)
        self.k_proj = nn.Dense(self.d_model)
        self.v_proj = nn.Dense(self.d_model)
        self.o_proj = nn.Dense(self.d_model, use_bias=False)
        self.attn_dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic=False):
        B, S, D = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scale = jnp.sqrt(jnp.float32(self.d_head))
        scores = jnp.einsum('bhsd,bhtd->bhst', q, k) / scale
        causal_mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal_mask, scores, jnp.finfo(scores.dtype).min)
        attn_weights = jax.nn.softmax(scores, axis=-1)

        attn_weights = self.attn_dropout(attn_weights, deterministic=deterministic)

        out = jnp.einsum('bhst,bhtd->bhsd', attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        return self.o_proj(out)


class StandardFFN(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=False):
        h = nn.Dense(self.d_ff)(x)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic)
        h = nn.Dense(self.d_model)(h)
        return h


class TransformerLayer(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    def setup(self):
        self.attn = StandardAttention(self.d_model, self.n_heads, self.dropout_rate)
        self.ffn = StandardFFN(self.d_model, self.d_ff, self.dropout_rate)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.attn_resid_dropout = nn.Dropout(self.dropout_rate)
        self.ffn_resid_dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic=False):
        normed = self.norm1(x)
        attn_out = self.attn(normed, deterministic=deterministic)
        attn_out = self.attn_resid_dropout(
            attn_out, deterministic=deterministic)
        x = x + attn_out

        normed = self.norm2(x)
        ffn_out = self.ffn(normed, deterministic=deterministic)
        ffn_out = self.ffn_resid_dropout(ffn_out, deterministic=deterministic)
        x = x + ffn_out
        return x


class VanillaTransformer(nn.Module):
    """Vanilla Transformer with optional tensor/vocab-parallel execution."""
    __version__ = "baseline-JAX"

    vocab_size: int = 30522
    d_model: int = 432
    d_ff: int = 1728
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.0
    gradient_checkpointing: bool = False
    logical_vocab_size: Optional[int] = None
    vocab_size_padded: Optional[int] = None

    def _vocab_sizes(self):
        logical = (
            int(self.logical_vocab_size)
            if self.logical_vocab_size is not None
            else int(self.vocab_size))
        embedding = (
            int(self.vocab_size_padded)
            if self.vocab_size_padded is not None
            else logical)
        if embedding < logical:
            raise ValueError(
                f"vocab_size_padded={embedding} must be >= "
                f"logical_vocab_size={logical}")
        return logical, embedding

    def setup(self):
        _, embedding_vocab_size = self._vocab_sizes()
        self.token_emb = nn.Embed(embedding_vocab_size, self.d_model,
                                  embedding_init=scaled_normal(0.02))
        self.pos_emb = nn.Embed(self.max_seq_len, self.d_model,
                                embedding_init=scaled_normal(0.02))
        LayerCls = (
            nn.remat(TransformerLayer, static_argnums=(1,))
            if self.gradient_checkpointing else TransformerLayer)
        self.layers = [
            LayerCls(self.d_model, self.n_heads, self.d_ff,
                     self.dropout_rate, name=f'layer_{i}')
            for i in range(self.n_layers)
        ]
        self.norm = nn.LayerNorm()
        self.emb_dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False, sharded_fns=None):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]

        vp_embed = (
            sharded_fns.get("vocab_parallel_embedding")
            if isinstance(sharded_fns, dict) else None)
        if vp_embed is not None:
            x = vp_embed(input_ids, self.token_emb.embedding)
        else:
            x = self.token_emb(input_ids)
        x = x + self.pos_emb(positions)

        x = self.emb_dropout(x, deterministic=deterministic)

        attn_fn = None
        ffn_fn = None
        if isinstance(sharded_fns, dict):
            attn_fn = sharded_fns.get("baseline_attention")
            ffn_fn = sharded_fns.get("baseline_ffn")

        use_tp_path = (
            not self.is_initializing()
            and attn_fn is not None
            and ffn_fn is not None)
        if use_tp_path:
            if self.dropout_rate > 0.0 and not deterministic:
                raise ValueError(
                    "baseline tensor-parallel train path requires "
                    f"dropout_rate=0.0; got dropout_rate={self.dropout_rate}.")

            def layer_forward(x_in, layer_params):
                normed = _layer_norm(
                    x_in,
                    layer_params["norm1"]["scale"],
                    layer_params["norm1"]["bias"])
                attn_params = layer_params["attn"]
                attn_out = attn_fn(
                    normed,
                    attn_params["q_proj"]["kernel"],
                    attn_params["q_proj"]["bias"],
                    attn_params["k_proj"]["kernel"],
                    attn_params["k_proj"]["bias"],
                    attn_params["v_proj"]["kernel"],
                    attn_params["v_proj"]["bias"],
                    attn_params["o_proj"]["kernel"],
                )
                x_mid = x_in + attn_out

                normed = _layer_norm(
                    x_mid,
                    layer_params["norm2"]["scale"],
                    layer_params["norm2"]["bias"])
                ffn_params = layer_params["ffn"]
                ffn_out = ffn_fn(
                    normed,
                    ffn_params["Dense_0"]["kernel"],
                    ffn_params["Dense_0"]["bias"],
                    ffn_params["Dense_1"]["kernel"],
                    ffn_params["Dense_1"]["bias"],
                )
                return x_mid + ffn_out

            if self.gradient_checkpointing:
                layer_forward = jax.checkpoint(layer_forward)

            params = self.variables["params"]
            for i in range(self.n_layers):
                x = layer_forward(x, params[f"layer_{i}"])
        else:
            for layer in self.layers:
                x = layer(x, deterministic)

        x = self.norm(x)

        result = {
            'aux_loss': jnp.float32(0.0),
        }

        if labels is not None:
            embedding_matrix = self.token_emb.embedding
            shift_x = x[:, :-1, :]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            valid_mask = (shift_labels != -100)

            vp_ce = (
                sharded_fns.get("vocab_parallel_ce")
                if isinstance(sharded_fns, dict) else None)
            if vp_ce is not None:
                loss, correct, valid_count = vp_ce(
                    shift_x, embedding_matrix, shift_labels, valid_mask)
            else:
                logical_vocab_size, _ = self._vocab_sizes()

                @jax.checkpoint
                def compute_loss_and_acc(x_chunk, emb, labs, vmask):
                    logits = x_chunk @ emb.T
                    if emb.shape[0] != logical_vocab_size:
                        vocab_ids = jnp.arange(emb.shape[0])
                        logits = jnp.where(
                            vocab_ids[None, None, :] < logical_vocab_size,
                            logits,
                            jnp.finfo(logits.dtype).min)
                    log_probs = jax.nn.log_softmax(logits, axis=-1)
                    safe = jnp.where(vmask, labs, 0)
                    tl = -jnp.take_along_axis(
                        log_probs, safe[..., jnp.newaxis], axis=-1).squeeze(-1)
                    loss = (tl * vmask).sum() / (vmask.sum() + 1e-8)
                    preds = jnp.argmax(logits, axis=-1)
                    correct = jnp.sum((preds == labs) & vmask)
                    valid_count = jnp.sum(vmask)
                    return loss, correct, valid_count

                loss, correct, valid_count = compute_loss_and_acc(
                    shift_x, embedding_matrix, shift_labels, valid_mask)
            result['loss'] = loss
            result['correct'] = correct
            result['valid_count'] = valid_count
        else:
            if vp_embed is not None:
                raise NotImplementedError(
                    "Full logits are disabled on the vocab-parallel baseline "
                    "training path. Pass labels or run without sharded_fns.")
            logits = self.token_emb.attend(x)
            result['logits'] = logits

        return result

    def orthogonality_loss(self):
        return jnp.float32(0.0)

    def knowledge_diversity_loss(self):
        return jnp.float32(0.0)

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size,
            'logical_vocab_size': self.logical_vocab_size,
            'vocab_size_padded': self.vocab_size_padded,
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
        }

    def get_model_info(self):
        ffn_ratio = self.d_ff / self.d_model
        logical_vocab_size, embedding_vocab_size = self._vocab_sizes()
        return [
            "  Model: VanillaTransformer (baseline-JAX, optional TP)",
            f"  d_model={self.d_model}, d_ff={self.d_ff}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  vocab logical/padded={logical_vocab_size}/{embedding_vocab_size}",
            f"  FFN ratio={ffn_ratio:.2f}, dropout={self.dropout_rate}",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
        ]


# ================================================================
# KV-Cached Inference  (for fast autoregressive generation)
# ================================================================

def _layer_norm(x, scale, bias, eps=1e-6):
    """Pure functional LayerNorm."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def _vanilla_attention_cached(x, attn_params, n_heads, d_model,
                               kv_cache_k, kv_cache_v, cache_index):
    """Standard attention with KV cache.

    Args:
        x:           [B, S, D]
        attn_params: dict with q_proj, k_proj, v_proj, o_proj
        kv_cache_k:  [B, H, max_len, d_head]
        kv_cache_v:  [B, H, max_len, d_head]
        cache_index: scalar int

    Returns:
        (output [B,S,D], updated_kv_cache_k, updated_kv_cache_v)
    """
    B, S, D = x.shape
    d_head = d_model // n_heads

    Q     = x @ attn_params['q_proj']['kernel'] + attn_params['q_proj']['bias']
    K_new = x @ attn_params['k_proj']['kernel'] + attn_params['k_proj']['bias']
    V_new = x @ attn_params['v_proj']['kernel'] + attn_params['v_proj']['bias']

    Q     = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K_new = K_new.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V_new = V_new.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    kv_cache_k = jax.lax.dynamic_update_slice(
        kv_cache_k, K_new, (0, 0, cache_index, 0))
    kv_cache_v = jax.lax.dynamic_update_slice(
        kv_cache_v, V_new, (0, 0, cache_index, 0))

    scale = jnp.sqrt(jnp.float32(d_head))
    scores = jnp.einsum('bhsd,bhtd->bhst', Q, kv_cache_k) / scale

    max_len = kv_cache_k.shape[2]
    q_positions = cache_index + jnp.arange(S)
    cache_positions = jnp.arange(max_len)
    causal = cache_positions[None, :] <= q_positions[:, None]
    scores = jnp.where(causal[None, None, :, :], scores,
                        jnp.finfo(scores.dtype).min)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_weights, kv_cache_v)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)

    output = attn_out @ attn_params['o_proj']['kernel']
    return output, kv_cache_k, kv_cache_v


def vanilla_init_kv_cache(config, batch_size=1):
    """Create zero-initialised KV caches for all layers.

    Returns:
        (all_k, all_v)  each  [n_layers, B, H, max_seq_len, d_head]
    """
    n_layers = config['n_layers']
    n_heads  = config.get('n_heads', 6)
    d_model  = config.get('d_model', 384)
    max_len  = config.get('max_seq_len', 512)
    d_head   = d_model // n_heads
    shape = (n_layers, batch_size, n_heads, max_len, d_head)
    return jnp.zeros(shape), jnp.zeros(shape)


def vanilla_cached_forward(params, config, input_ids,
                            kv_caches_k, kv_caches_v, cache_index):
    """Full forward pass with KV cache for VanillaTransformer.

    Pure function suitable for ``jax.jit``.

    Args:
        params:       param dict (may have outer 'params' key)
        config:       model config dict
        input_ids:    [B, S]
        kv_caches_k:  [n_layers, B, H, max_len, d_head]
        kv_caches_v:  [n_layers, B, H, max_len, d_head]
        cache_index:  scalar int

    Returns:
        (logits [B,S,V], updated_kv_caches_k, updated_kv_caches_v)
    """
    n_layers = config['n_layers']
    d_model  = config.get('d_model', 384)
    n_heads  = config.get('n_heads', 6)

    p = params
    if hasattr(p, 'get') and 'params' in p:
        p = p['params']

    token_emb   = p['token_emb']['embedding']
    pos_emb     = p['pos_emb']['embedding']
    norm_params = p['norm']

    B, S = input_ids.shape
    x = token_emb[input_ids]
    positions = jnp.arange(S) + cache_index
    x = x + pos_emb[positions][None, :]

    # Stack per-layer params
    layer_params_list = [p[f'layer_{i}'] for i in range(n_layers)]
    stacked_lp = jax.tree.map(lambda *a: jnp.stack(a), *layer_params_list)

    def scan_body(carry, xs):
        x = carry
        lp   = xs['params']
        kv_k = xs['kv_k']
        kv_v = xs['kv_v']

        # Attention
        normed = _layer_norm(x, lp['norm1']['scale'], lp['norm1']['bias'])
        attn_out, kv_k, kv_v = _vanilla_attention_cached(
            normed, lp['attn'], n_heads, d_model,
            kv_k, kv_v, cache_index)
        x = x + attn_out

        # FFN
        normed = _layer_norm(x, lp['norm2']['scale'], lp['norm2']['bias'])
        h = normed @ lp['ffn']['Dense_0']['kernel'] + lp['ffn']['Dense_0']['bias']
        h = jax.nn.gelu(h)
        ffn_out = h @ lp['ffn']['Dense_1']['kernel'] + lp['ffn']['Dense_1']['bias']
        x = x + ffn_out

        return x, {'kv_k': kv_k, 'kv_v': kv_v}

    xs = {
        'params': stacked_lp,
        'kv_k':   kv_caches_k,
        'kv_v':   kv_caches_v,
    }
    x, outputs = jax.lax.scan(scan_body, x, xs)

    x = _layer_norm(x, norm_params['scale'], norm_params['bias'])
    logits = x @ token_emb.T

    return logits, outputs['kv_k'], outputs['kv_v']
