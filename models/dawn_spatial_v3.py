"""
DAWN-Spatial v3.6: Threshold Gate + F-R Basis Dynamic Synthesis (JAX/Flax)

Changelog:
  spatial-r1-v3.6.0 (2026-04-01):
    - Emit replaced with F-R basis dynamic synthesis
    - 21000 sense neurons (emb[64]) produce gate[B,S,N]
    - gate grouped into M basis weights via reshape+sum
    - feature_fn/restore_fn on M basis neurons (v17.1 pattern)
    - Neuron activation pattern determines which F-R basis to use
    - per_slice_orthogonal init for F-R basis

  spatial-r1-v3.5.0 (2026-04-01):
    - Threshold-only gating, dense bottleneck emit. 4s/step.

Architecture:
  NeuronPool    -- emb[N,d_bn] (sense) + f_basis[M,D,R] + r_basis[M,R,D]
  Router        -- proj + tau. Uses pool emb for routing.
  threshold_gate -- element-wise threshold, no sort/top_k (v3.5.0)
  group_gate    -- gate[B,S,N] -> gate_grouped[B,S,M] via reshape+sum
  feature_fn    -- @jax.checkpoint: gate-weighted basis projections
  restore_fn    -- @jax.checkpoint: gate-weighted basis restores
  _attn_forward -- threshold gate -> group -> F-R emit -> self-attn
  _know_forward -- threshold gate -> group -> F-R emit
  DAWN          -- embedding + jax.lax.scan + weight-tied lm_head
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict


# ================================================================
# 1. Helpers
# ================================================================

def safe_dropout(x, rate, deterministic, rng):
    if rate == 0.0:
        return x
    keep_rate = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep_rate, x.shape)
    mask = jnp.where(deterministic, jnp.ones_like(mask), mask)
    return jnp.where(mask, x / keep_rate, 0.0)


def _layer_norm(x, scale, bias, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def scaled_normal(scale=0.02):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * scale
    return init


def unit_norm_init(scale=1.0):
    def init(key, shape, dtype=jnp.float32):
        x = jax.random.normal(key, shape, dtype)
        norms = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        return x / norms * scale
    return init


def per_slice_orthogonal():
    """Orthogonal init per basis slice. For [M, D, R] or [M, R, D]."""
    base = nn.initializers.orthogonal()
    def init(key, shape, dtype=jnp.float32):
        keys = jax.random.split(key, shape[0])
        return jax.vmap(lambda k: base(k, shape[1:], dtype))(keys)
    return init


# ================================================================
# 2. Threshold gate (v3.5.0, unchanged)
# ================================================================

def threshold_gate(scores, tau):
    """Element-wise threshold gating. No sort, no top_k."""
    raw = scores - tau
    gate = jnp.where(raw > 0, raw, 0.0)
    gate_sum = gate.sum(axis=-1, keepdims=True) + 1e-8
    return gate / gate_sum


# ================================================================
# 3. Gate grouping: N neurons -> M basis weights
# ================================================================

def group_gate(gate, n_basis):
    """Group gate[B,S,N] into gate_grouped[B,S,M] via reshape+sum.

    Each basis covers N//M consecutive neurons. Gate values summed per group.
    Remainder neurons (N % M) assigned to last group.
    """
    B, S, N = gate.shape
    group_size = N // n_basis

    # Trim to exact multiple, reshape, sum
    n_exact = n_basis * group_size
    gate_main = gate[:, :, :n_exact].reshape(B, S, n_basis, group_size)
    gate_grouped = gate_main.sum(axis=-1)  # [B, S, M]

    # Add remainder to last group
    if N > n_exact:
        gate_grouped = gate_grouped.at[:, :, -1].add(
            gate[:, :, n_exact:].sum(axis=-1))

    # Re-normalize
    gs = gate_grouped.sum(axis=-1, keepdims=True) + 1e-8
    return gate_grouped / gs


# ================================================================
# 4. Feature-Restore (v17.1 pattern)
# ================================================================

@jax.checkpoint
def feature_fn(x, neurons, weights):
    """Feature: x[B,S,D] -> h[B,S,R].
    neurons: [M, D, R], weights: [B, S, M] (grouped gate).
    Intermediate [B,S,M,R] not saved (checkpoint).
    """
    all_h = jnp.einsum('bsd,ndr->bsnr', x, neurons)
    return jnp.einsum('bsnr,bsn->bsr', all_h, weights)


@jax.checkpoint
def restore_fn(h, neurons, weights):
    """Restore: h[B,S,R] -> out[B,S,D].
    neurons: [M, R, D], weights: [B, S, M] (grouped gate).
    Intermediate [B,S,M,R] not saved (checkpoint).
    """
    pre = jnp.einsum('bsr,bsn->bsnr', h, weights)
    return jnp.einsum('bsnr,nrd->bsd', pre, neurons)


# ================================================================
# 5. NeuronPool -- emb[N,d_bn] (sense) + f/r basis[M,D,R] (F-R emit)
# ================================================================

class NeuronPool(nn.Module):
    n_qk: int
    n_v: int
    n_know: int
    d_model: int
    d_bottleneck: int    # = rank R
    n_basis_qk: int
    n_basis_v: int
    n_basis_know: int

    def setup(self):
        db = self.d_bottleneck
        dm = self.d_model
        orth = per_slice_orthogonal()

        # Sense/routing embeddings (21000 neurons, low-dim)
        self.qk_emb = self.param('qk_emb', unit_norm_init(), (self.n_qk, db))
        self.v_emb = self.param('v_emb', unit_norm_init(), (self.n_v, db))
        self.know_emb = self.param('know_emb', unit_norm_init(), (self.n_know, db))

        # F-R basis (M basis neurons, full-rank projections)
        self.qk_f = self.param('qk_f', orth, (self.n_basis_qk, dm, db))
        self.qk_r = self.param('qk_r', orth, (self.n_basis_qk, db, dm))
        self.v_f = self.param('v_f', orth, (self.n_basis_v, dm, db))
        self.v_r = self.param('v_r', orth, (self.n_basis_v, db, dm))
        self.know_f = self.param('know_f', orth, (self.n_basis_know, dm, db))
        self.know_r = self.param('know_r', orth, (self.n_basis_know, db, dm))


# ================================================================
# 6. Router -- proj + tau (unchanged from v3.5.0)
# ================================================================

class Router(nn.Module):
    d_model: int
    d_bottleneck: int
    n_qk: int
    n_v: int
    n_know: int
    max_k_qk: int
    max_k_v: int
    max_k_know: int
    router_dropout: float = 0.1

    def setup(self):
        db = self.d_bottleneck
        self.proj_attn = nn.Dense(db * 3, name='proj_attn')
        self.proj_know = nn.Dense(db, name='proj_know')
        self.tau_attn = nn.Dense(3, name='tau_attn')
        self.tau_know = nn.Dense(1, name='tau_know')

    def get_attention_gates(self, x, neuron_pool, deterministic, rng):
        qk_norm = neuron_pool.qk_emb / (
            jnp.linalg.norm(neuron_pool.qk_emb, axis=-1, keepdims=True) + 1e-8)
        v_norm = neuron_pool.v_emb / (
            jnp.linalg.norm(neuron_pool.v_emb, axis=-1, keepdims=True) + 1e-8)

        rng, rng_drop = jax.random.split(rng)
        h_all = self.proj_attn(x)
        h_all = safe_dropout(h_all, self.router_dropout, deterministic, rng_drop)
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

        tau_all = self.tau_attn(x)
        g_Q = threshold_gate(h_Q @ qk_norm.T, tau_all[:, :, 0:1])
        g_K = threshold_gate(h_K @ qk_norm.T, tau_all[:, :, 1:2])
        g_V = threshold_gate(h_V @ v_norm.T, tau_all[:, :, 2:3])

        t_qk = 1.0 / self.n_qk
        t_v = 1.0 / self.n_v
        aux = (
            ((g_Q.mean(axis=(0, 1)) - t_qk) ** 2).sum() * self.n_qk +
            ((g_K.mean(axis=(0, 1)) - t_qk) ** 2).sum() * self.n_qk +
            ((g_V.mean(axis=(0, 1)) - t_v) ** 2).sum() * self.n_v
        )
        return g_Q, g_K, g_V, aux

    def get_knowledge_gates(self, x, neuron_pool, deterministic, rng):
        know_norm = neuron_pool.know_emb / (
            jnp.linalg.norm(neuron_pool.know_emb, axis=-1, keepdims=True) + 1e-8)

        rng, rng_drop = jax.random.split(rng)
        h = self.proj_know(x)
        h = safe_dropout(h, self.router_dropout, deterministic, rng_drop)

        tau = self.tau_know(x)
        gate = threshold_gate(h @ know_norm.T, tau)

        t = 1.0 / self.n_know
        aux = ((gate.mean(axis=(0, 1)) - t) ** 2).sum() * self.n_know
        return gate, aux


# ================================================================
# 7. Pure functions for scan body
# ================================================================

def _attn_forward(x, pool_params, router_params, expand_O_kernel, rng,
                  n_qk, n_v, n_basis_qk, n_basis_v,
                  max_k_qk, max_k_v, n_heads, d_model,
                  router_dropout, dropout_rate, deterministic):
    B, S, D = x.shape
    qk_emb = pool_params['qk_emb']
    qk_f = pool_params['qk_f']
    qk_r = pool_params['qk_r']
    v_emb = pool_params['v_emb']
    v_f = pool_params['v_f']
    v_r = pool_params['v_r']

    qk_norm = qk_emb / (jnp.linalg.norm(qk_emb, axis=-1, keepdims=True) + 1e-8)
    v_norm = v_emb / (jnp.linalg.norm(v_emb, axis=-1, keepdims=True) + 1e-8)

    rng, rng_drop = jax.random.split(rng)
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_all = safe_dropout(h_all, router_dropout, deterministic, rng_drop)
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

    g_Q = threshold_gate(h_Q @ qk_norm.T, tau_all[:, :, 0:1])
    g_K = threshold_gate(h_K @ qk_norm.T, tau_all[:, :, 1:2])
    g_V = threshold_gate(h_V @ v_norm.T, tau_all[:, :, 2:3])

    # Group gates -> basis weights
    gQ_b = group_gate(g_Q, n_basis_qk)
    gK_b = group_gate(g_K, n_basis_qk)
    gV_b = group_gate(g_V, n_basis_v)

    # F-R emit
    Q = restore_fn(feature_fn(x, qk_f, gQ_b), qk_r, gQ_b)
    K = restore_fn(feature_fn(x, qk_f, gK_b), qk_r, gK_b)
    V = restore_fn(feature_fn(x, v_f, gV_b), v_r, gV_b)

    d_head = d_model // n_heads
    Q = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    attn_scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn_scores = jnp.where(causal, attn_scores,
                            jnp.finfo(attn_scores.dtype).min)
    attn_w = jax.nn.softmax(attn_scores, axis=-1)

    rng, rng_attn, rng_out = jax.random.split(rng, 3)
    attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_attn)

    out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    t_qk = 1.0 / n_qk
    t_v = 1.0 / n_v
    aux = (
        ((g_Q.mean(axis=(0, 1)) - t_qk) ** 2).sum() * n_qk +
        ((g_K.mean(axis=(0, 1)) - t_qk) ** 2).sum() * n_qk +
        ((g_V.mean(axis=(0, 1)) - t_v) ** 2).sum() * n_v
    )
    return out, aux


def _know_forward(x, pool_params, router_params, rng,
                  n_know, n_basis_know, max_k_know,
                  router_dropout, dropout_rate, deterministic):
    know_emb = pool_params['know_emb']
    know_f = pool_params['know_f']
    know_r = pool_params['know_r']

    know_norm = know_emb / (jnp.linalg.norm(know_emb, axis=-1, keepdims=True) + 1e-8)

    rng, rng_drop = jax.random.split(rng)
    h = x @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
    h = safe_dropout(h, router_dropout, deterministic, rng_drop)

    tau = x @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
    gate = threshold_gate(h @ know_norm.T, tau)

    # Group -> F-R emit
    gate_b = group_gate(gate, n_basis_know)
    h_feat = feature_fn(x, know_f, gate_b)
    out = restore_fn(h_feat, know_r, gate_b)

    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    t = 1.0 / know_emb.shape[0]
    aux = ((gate.mean(axis=(0, 1)) - t) ** 2).sum() * know_emb.shape[0]
    return out, aux


# ================================================================
# 8. Flax modules (init path only)
# ================================================================

class AttentionCircuit(nn.Module):
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.expand_O = nn.Dense(
            self.d_model, use_bias=False, kernel_init=scaled_normal(0.02))

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        rng = self.make_rng('dropout')
        rng, rng_r, rng_d, rng_o = jax.random.split(rng, 4)

        g_Q, g_K, g_V, aux = router.get_attention_gates(
            x, neuron_pool, deterministic, rng_r)

        n_b_qk = neuron_pool.qk_f.shape[0]
        n_b_v = neuron_pool.v_f.shape[0]

        gQ_b = group_gate(g_Q, n_b_qk)
        gK_b = group_gate(g_K, n_b_qk)
        gV_b = group_gate(g_V, n_b_v)

        Q = restore_fn(feature_fn(x, neuron_pool.qk_f, gQ_b),
                        neuron_pool.qk_r, gQ_b)
        K = restore_fn(feature_fn(x, neuron_pool.qk_f, gK_b),
                        neuron_pool.qk_r, gK_b)
        V = restore_fn(feature_fn(x, neuron_pool.v_f, gV_b),
                        neuron_pool.v_r, gV_b)

        B, S, D = x.shape
        d_head = D // self.n_heads
        Q = Q.reshape(B, S, self.n_heads, d_head).transpose(0, 2, 1, 3)
        K = K.reshape(B, S, self.n_heads, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, self.n_heads, d_head).transpose(0, 2, 1, 3)

        scale = jnp.sqrt(jnp.float32(d_head))
        scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_w = safe_dropout(attn_w, self.dropout_rate, deterministic, rng_d)

        out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        out = self.expand_O(out)
        out = safe_dropout(out, self.dropout_rate, deterministic, rng_o)
        return out, aux


class KnowledgeCircuit(nn.Module):
    d_model: int
    dropout_rate: float = 0.1

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        rng = self.make_rng('dropout')
        rng, rng_r = jax.random.split(rng)
        gate, aux = router.get_knowledge_gates(
            x, neuron_pool, deterministic, rng_r)

        n_b = neuron_pool.know_f.shape[0]
        gate_b = group_gate(gate, n_b)
        h_feat = feature_fn(x, neuron_pool.know_f, gate_b)
        out = restore_fn(h_feat, neuron_pool.know_r, gate_b)

        out = safe_dropout(out, self.dropout_rate, deterministic, rng)
        return out, aux


class DAWNBlock(nn.Module):
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.attn = AttentionCircuit(
            d_model=self.d_model, n_heads=self.n_heads,
            dropout_rate=self.dropout_rate)
        self.knowledge = KnowledgeCircuit(
            d_model=self.d_model, dropout_rate=self.dropout_rate)

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        normed = self.norm1(x)
        attn_out, a_aux = self.attn(
            normed, neuron_pool, router, attention_mask, deterministic)
        x = x + attn_out
        normed = self.norm2(x)
        know_out, k_aux = self.knowledge(
            normed, neuron_pool, router, attention_mask, deterministic)
        x = x + know_out
        return x, a_aux + k_aux


# ================================================================
# 9. DAWN Model
# ================================================================

class DAWN(nn.Module):
    """DAWN-Spatial v3.6: Threshold Gate + F-R Basis Dynamic Synthesis."""
    __version__ = "spatial-r1-v3.6.0"

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    d_bottleneck: int = 64
    n_qk: int = 1570
    n_v: int = 2620
    n_know: int = 21000
    n_basis_qk: int = 32
    n_basis_v: int = 32
    n_basis_know: int = 256
    max_k_qk: int = 157      # config compat
    max_k_v: int = 262
    max_k_know: int = 1536
    router_dropout: float = 0.1

    def setup(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})")
        self.token_emb = nn.Embed(
            self.vocab_size, self.d_model, embedding_init=scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model, embedding_init=scaled_normal(0.02))
        self.neuron_pool = NeuronPool(
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            d_model=self.d_model, d_bottleneck=self.d_bottleneck,
            n_basis_qk=self.n_basis_qk, n_basis_v=self.n_basis_v,
            n_basis_know=self.n_basis_know)
        self.router = Router(
            d_model=self.d_model, d_bottleneck=self.d_bottleneck,
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            max_k_qk=self.max_k_qk, max_k_v=self.max_k_v,
            max_k_know=self.max_k_know, router_dropout=self.router_dropout)
        self.layers = [
            DAWNBlock(d_model=self.d_model, n_heads=self.n_heads,
                      dropout_rate=self.dropout_rate, name=f'block_{i}')
            for i in range(self.n_layers)]
        self.norm = nn.LayerNorm()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False):
        B, S = input_ids.shape
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len")

        positions = jnp.arange(S)[jnp.newaxis, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        emb_rng = self.make_rng('dropout')
        x = safe_dropout(x, self.dropout_rate, deterministic, emb_rng)

        if self.is_initializing():
            total_aux = jnp.float32(0.0)
            for layer in self.layers:
                x, aux = layer(x, self.neuron_pool, self.router,
                               attention_mask, deterministic)
                total_aux = total_aux + aux
        else:
            all_params = self.variables['params']
            pool_params = all_params['neuron_pool']
            router_params = all_params['router']

            block_params_list = [all_params[f'block_{i}']
                                 for i in range(self.n_layers)]
            stacked = jax.tree.map(
                lambda *arrays: jnp.stack(arrays), *block_params_list)

            base_rng = self.make_rng('dropout')
            layer_rngs = jax.random.split(base_rng, self.n_layers)

            def scan_body(carry, xs):
                x = carry
                bp = xs['params']
                rng = xs['rng']
                rng, rng_attn, rng_know = jax.random.split(rng, 3)

                normed = _layer_norm(
                    x, bp['norm1']['scale'], bp['norm1']['bias'])
                attn_out, attn_aux = _attn_forward(
                    normed, pool_params, router_params,
                    bp['attn']['expand_O']['kernel'], rng_attn,
                    self.n_qk, self.n_v,
                    self.n_basis_qk, self.n_basis_v,
                    self.max_k_qk, self.max_k_v,
                    self.n_heads, self.d_model,
                    self.router_dropout, self.dropout_rate, deterministic)
                x = x + attn_out

                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                know_out, know_aux = _know_forward(
                    normed, pool_params, router_params, rng_know,
                    self.n_know, self.n_basis_know, self.max_k_know,
                    self.router_dropout, self.dropout_rate, deterministic)
                x = x + know_out
                return x, attn_aux + know_aux

            if self.gradient_checkpointing:
                scan_body = jax.checkpoint(scan_body)

            xs = {'params': stacked, 'rng': layer_rngs}
            x, aux_losses = jax.lax.scan(scan_body, x, xs)
            total_aux = aux_losses.sum()

        x = self.norm(x)
        result = {'aux_loss': total_aux}

        if labels is not None:
            embedding_matrix = self.token_emb.embedding
            shift_x = x[:, :-1, :]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            valid_mask = (shift_labels != -100)

            @jax.checkpoint
            def compute_loss_and_acc(x_chunk, emb, labs, vmask):
                logits = x_chunk @ emb.T
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                safe = jnp.where(vmask, labs, 0)
                tl = -jnp.take_along_axis(
                    log_probs, safe[..., jnp.newaxis], axis=-1).squeeze(-1)
                loss = (tl * vmask).sum() / (vmask.sum() + 1e-8)
                preds = jnp.argmax(logits, axis=-1)
                correct = jnp.sum((preds == labs) & vmask)
                return loss, correct, jnp.sum(vmask)

            loss, correct, valid_count = compute_loss_and_acc(
                shift_x, embedding_matrix, shift_labels, valid_mask)
            result['loss'] = loss
            result['correct'] = correct
            result['valid_count'] = valid_count
        else:
            result['logits'] = self.token_emb.attend(x)

        return result

    def diversity_loss(self):
        """Diversity on sense embeddings. F-R basis uses orth init."""
        def _div(neurons, max_sample=4096):
            N = neurons.shape[0]
            if N > max_sample:
                stride = N // max_sample
                neurons = neurons[::stride][:max_sample]
            n = neurons / (jnp.linalg.norm(neurons, axis=-1, keepdims=True) + 1e-8)
            sim = n @ n.T
            mask = ~jnp.eye(sim.shape[0], dtype=jnp.bool_)
            return jnp.abs(sim * mask).sum() / mask.sum()
        pool = self.neuron_pool
        return (_div(pool.qk_emb) + _div(pool.v_emb) +
                _div(pool.know_emb)) / 3

    def get_auxiliary_losses(self):
        return {'neuron_diversity': self.diversity_loss()}

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
            'd_bottleneck': self.d_bottleneck,
            'n_qk': self.n_qk, 'n_v': self.n_v, 'n_know': self.n_know,
            'n_basis_qk': self.n_basis_qk, 'n_basis_v': self.n_basis_v,
            'n_basis_know': self.n_basis_know,
            'max_k_qk': self.max_k_qk, 'max_k_v': self.max_k_v,
            'max_k_know': self.max_k_know,
        }

    def get_model_info(self):
        return [
            f"DAWN v{self.__version__}: Threshold Gate + F-R Basis Synthesis",
            f"  d_model={self.d_model}, d_bottleneck={self.d_bottleneck}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  Sense: QK={self.n_qk}, V={self.n_v}, Know={self.n_know}",
            f"  F-R basis: QK={self.n_basis_qk}, V={self.n_basis_v}, "
            f"Know={self.n_basis_know}",
        ]
