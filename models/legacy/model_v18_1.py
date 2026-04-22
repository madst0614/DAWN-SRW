"""
DAWN v18.1: Soft Mask + Learnable Tau Routing

Key Concept:
- Based on v18.0 with differentiable threshold selection
- Soft mask: sigmoid((score - tau) / temp) instead of hard threshold
- Learnable tau: separate nn.Parameter for each neuron type
- Penalty-based masking: low-score neurons get penalty instead of -inf

Changes from v18.0:
- Hard mask (score > tau) → Soft mask (sigmoid((score - tau) / temp))
- Fixed tau → Learnable tau (nn.Parameter for each pool)
- -inf masking → Penalty-based masking (-soft_mask_penalty)
- Gradients flow through all neurons (differentiable routing)

Architecture:
- UnifiedNeuronRouter: same as v18.0
- GlobalRouters: soft mask + learnable tau parameters
- AttentionCircuit: same as v18.0 (multi-path Q, K, V aggregation)
- KnowledgeCircuit: same as v18.0 (multi-path aggregation)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed, using slow for-loop SSM")


class UnifiedNeuronRouter(nn.Module):
    """
    v18.0: 6 attention projections + 2 knowledge projections
    Fixed threshold (tau) for neuron selection
    """
    def __init__(self, d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
                 n_feature_know, n_restore_know,
                 d_space=64, dropout=0.1, fixed_tau=0.0, **kwargs):
        super().__init__()
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.d_space = d_space
        self.fixed_tau = fixed_tau
        self.ema_alpha = kwargs.get('excitability_ema_alpha', 0.01)

        # 6 attention + 2 knowledge pools
        total_neurons = (n_feature_qk + n_feature_v + n_restore_qk + n_restore_v +
                        n_feature_know + n_restore_know)
        self.total_neurons = total_neurons

        # Index boundaries
        self.feature_qk_end = n_feature_qk
        self.feature_v_end = n_feature_qk + n_feature_v
        self.restore_qk_end = n_feature_qk + n_feature_v + n_restore_qk
        self.restore_v_end = n_feature_qk + n_feature_v + n_restore_qk + n_restore_v
        self.feature_know_end = self.restore_v_end + n_feature_know
        # restore_know: feature_know_end ~ total_neurons

        # 6 attention projections + 2 knowledge projections
        self.proj_all = nn.Linear(d_model, d_space * 6)  # fqk_Q, fqk_K, fv, rqk_Q, rqk_K, rv
        self.proj_feature_know = nn.Linear(d_model, d_space)
        self.proj_restore_know = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # Unified neuron embeddings (std=0.02 is standard transformer initialization)
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # Usage tracking (for logging)
        self.register_buffer('usage_ema_feature_q', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_k', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_v', torch.zeros(n_feature_v))
        self.register_buffer('usage_ema_restore_q', torch.zeros(n_restore_qk))
        self.register_buffer('usage_ema_restore_k', torch.zeros(n_restore_qk))
        self.register_buffer('usage_ema_restore_v', torch.zeros(n_restore_v))
        self.register_buffer('usage_ema_feature_know', torch.zeros(n_feature_know))
        self.register_buffer('usage_ema_restore_know', torch.zeros(n_restore_know))

    def get_thresholds(self, x):
        """
        Return fixed tau for all pools (no learnable threshold)
        x: [B, S, d_model]
        Returns: dict of scalar thresholds
        """
        return {
            'fqk': self.fixed_tau,
            'fv': self.fixed_tau,
            'rqk': self.fixed_tau,
            'rv': self.fixed_tau,
            'feature_know': self.fixed_tau,
            'restore_know': self.fixed_tau,
        }

    def get_knowledge_logits(self, x):
        """
        Return 2 knowledge logits (feature_know, restore_know)
        x: [B, S, d_model]
        """
        emb_norm = F.normalize(self.neuron_emb, dim=-1)

        # Feature_know
        h_feature_know = self.dropout(self.proj_feature_know(x))
        emb_feature_know = emb_norm[self.restore_v_end:self.feature_know_end]
        logits_feature_know = torch.einsum('bsd,nd->bsn', h_feature_know, emb_feature_know)

        # Restore_know
        h_restore_know = self.dropout(self.proj_restore_know(x))
        emb_restore_know = emb_norm[self.feature_know_end:]
        logits_restore_know = torch.einsum('bsd,nd->bsn', h_restore_know, emb_restore_know)

        return logits_feature_know, logits_restore_know

    def get_all_logits(self, x):
        """6 attention logits at once"""
        emb_norm = F.normalize(self.neuron_emb, dim=-1)

        all_proj = self.dropout(self.proj_all(x))
        h_fqk_Q, h_fqk_K, h_fv, h_rqk_Q, h_rqk_K, h_rv = all_proj.chunk(6, dim=-1)

        fqk_emb = emb_norm[:self.feature_qk_end]
        fv_emb = emb_norm[self.feature_qk_end:self.feature_v_end]
        rqk_emb = emb_norm[self.feature_v_end:self.restore_qk_end]
        rv_emb = emb_norm[self.restore_qk_end:self.restore_v_end]

        logits_fqk_Q = torch.einsum('bsd,nd->bsn', h_fqk_Q, fqk_emb)
        logits_fqk_K = torch.einsum('bsd,nd->bsn', h_fqk_K, fqk_emb)
        logits_fv = torch.einsum('bsd,nd->bsn', h_fv, fv_emb)
        logits_rqk_Q = torch.einsum('bsd,nd->bsn', h_rqk_Q, rqk_emb)
        logits_rqk_K = torch.einsum('bsd,nd->bsn', h_rqk_K, rqk_emb)
        logits_rv = torch.einsum('bsd,nd->bsn', h_rv, rv_emb)

        return logits_fqk_Q, logits_fqk_K, logits_fv, logits_rqk_Q, logits_rqk_K, logits_rv

    def update_usage(self, weights, neuron_type, attention_mask=None):
        """
        Update usage EMA for neuron tracking.
        v18 uses soft weights (sigmoid outputs 0~1), so we use the actual weight values
        as usage intensity, not binary active/inactive.
        """
        if not self.training:
            return

        if weights.dim() == 3:
            # Use actual soft weight values (not binary) for v18
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                weights_masked = weights * mask
                count = mask.sum() + 1e-8
                usage = weights_masked.sum(dim=[0, 1]) / count
            else:
                usage = weights.mean(dim=[0, 1])
        else:
            usage = weights.mean(dim=0)

        decay = 1 - self.ema_alpha
        if neuron_type == 'feature_q':
            self.usage_ema_feature_q.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'feature_k':
            self.usage_ema_feature_k.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'feature_v':
            self.usage_ema_feature_v.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_q':
            self.usage_ema_restore_q.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_k':
            self.usage_ema_restore_k.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_v':
            self.usage_ema_restore_v.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'feature_know':
            self.usage_ema_feature_know.mul_(decay).add_(usage, alpha=self.ema_alpha)
        elif neuron_type == 'restore_know':
            self.usage_ema_restore_know.mul_(decay).add_(usage, alpha=self.ema_alpha)


class SharedNeurons(nn.Module):
    """
    v18.0: Both Attention + Knowledge use Feature-Restore pattern
    Same structure as v17.1, rank reduced to 16
    """
    def __init__(
        self,
        d_model: int,
        rank: int,
        n_feature_qk: int,
        n_feature_v: int,
        n_restore_qk: int,
        n_restore_v: int,
        n_feature_know: int,
        n_restore_know: int,
        knowledge_rank: int = 128,
    ):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know

        # Attention neurons (contiguous memory)
        self.f_neurons = nn.Parameter(torch.zeros(n_feature_qk + n_feature_v, d_model, rank))
        self.r_neurons = nn.Parameter(torch.zeros(n_restore_qk + n_restore_v, rank, d_model))

        # Knowledge neurons: Feature-Restore pattern (separate routing)
        self.feature_know = nn.Parameter(torch.zeros(n_feature_know, d_model, knowledge_rank))
        self.restore_know = nn.Parameter(torch.zeros(n_restore_know, knowledge_rank, d_model))

        self._init_parameters()

    @property
    def feature_qk_neurons(self):
        return self.f_neurons[:self.n_feature_qk]

    @property
    def feature_v_neurons(self):
        return self.f_neurons[self.n_feature_qk:]

    @property
    def restore_qk_neurons(self):
        return self.r_neurons[:self.n_restore_qk]

    @property
    def restore_v_neurons(self):
        return self.r_neurons[self.n_restore_qk:]

    def _init_parameters(self):
        # Attention neurons
        for i in range(self.n_feature_qk + self.n_feature_v):
            nn.init.orthogonal_(self.f_neurons.data[i])
        for i in range(self.n_restore_qk + self.n_restore_v):
            nn.init.orthogonal_(self.r_neurons.data[i])
        # Knowledge neurons (orthogonal init)
        for i in range(self.n_feature_know):
            nn.init.orthogonal_(self.feature_know.data[i])
        for i in range(self.n_restore_know):
            nn.init.orthogonal_(self.restore_know.data[i])


class GlobalSSM(nn.Module):
    """Selective SSM (same as v17.1)"""
    def __init__(self, d_model: int, state_dim: int, return_context: bool = True):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        self.return_context = return_context

        # A_log initialization (small values for stable SSM dynamics)
        self.A_log = nn.Parameter(torch.randn(d_model, state_dim) * 0.1)
        self.W_delta = nn.Linear(d_model, d_model, bias=False)
        self.W_B = nn.Linear(d_model, state_dim, bias=False)
        self.W_C = nn.Linear(d_model, state_dim, bias=False)

        self.ssm_norm = nn.LayerNorm(d_model)
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        # Initial context scale (small to avoid disrupting early training)
        self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.importance_proj = nn.Linear(d_model, d_model, bias=False)
        # Temperature for importance softmax (lower = sharper distribution)
        self.importance_temperature = 0.5

        self._init_weights()

    def _init_weights(self):
        # std=0.02 is standard transformer initialization (GPT-2, BERT)
        nn.init.normal_(self.W_delta.weight, std=0.02)
        nn.init.normal_(self.W_B.weight, std=0.02)
        nn.init.normal_(self.W_C.weight, std=0.02)
        nn.init.normal_(self.context_proj.weight, std=0.02)
        nn.init.normal_(self.importance_proj.weight, std=0.02)

    def forward(self, x, attention_mask=None):
        B, S, D = x.shape

        delta = F.softplus(self.W_delta(x))
        B_sel = self.W_B(x)
        C_sel = self.W_C(x)
        A = -torch.exp(self.A_log)

        if MAMBA_AVAILABLE:
            dtype = x.dtype
            x_mamba = x.transpose(1, 2).contiguous()
            delta_mamba = delta.transpose(1, 2).contiguous()
            B_mamba = B_sel.transpose(1, 2).contiguous().to(dtype)
            C_mamba = C_sel.transpose(1, 2).contiguous().to(dtype)
            A = A.to(dtype)

            y = selective_scan_fn(
                x_mamba, delta_mamba, A, B_mamba, C_mamba,
                D=None, z=None, delta_bias=None,
                delta_softplus=False, return_last_state=False
            )
            ssm_out = y.transpose(1, 2).contiguous()
        else:
            ssm_out = self._slow_forward(x, delta, A, B_sel, C_sel)

        ssm_out = self.ssm_norm(ssm_out)

        # Position-wise importance (no future leakage)
        h_proj = self.importance_proj(ssm_out)  # [B, S, D]
        raw_importance = (x * h_proj).sum(dim=-1)  # [B, S]

        if attention_mask is not None:
            masked_importance = raw_importance.masked_fill(attention_mask == 0, float('-inf'))
            importance = F.softmax(masked_importance / self.importance_temperature, dim=-1)
        else:
            importance = F.softmax(raw_importance / self.importance_temperature, dim=-1)

        if self.return_context:
            context = self.context_proj(ssm_out) * self.context_scale
        else:
            context = None

        return importance, context, raw_importance

    def _slow_forward(self, x, delta, A, B_sel, C_sel):
        B, S, D = x.shape
        N = self.state_dim

        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(S):
            delta_t = delta[:, t, :, None]
            A_exp = A[None, :, :]
            decay = torch.exp(delta_t * A_exp)

            B_t = B_sel[:, t, None, :]
            x_t = x[:, t, :, None]

            h = h * decay + (delta_t * x_t) * B_t
            C_t = C_sel[:, t, :]
            y_t = torch.einsum('bdn,bn->bd', h, C_t)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class GlobalRouters(nn.Module):
    """
    v18.0: Fixed threshold + softmax multi-path routing
    v18.1: Soft mask + softmax with learnable tau (when use_soft_mask=True)

    v18.0 uses:
    1. Fixed threshold (tau) for initial neuron filtering
    2. Minimum/maximum neuron guarantees (path_min_k, path_max_k * max_paths)
    3. Masked softmax for differentiable weighting
    4. Chunking into multiple paths by score ranking

    v18.1 adds:
    1. Soft mask: sigmoid((score - tau) / temp) instead of hard threshold
    2. Learnable tau: tau becomes nn.Parameter
    3. Penalty-based masking: low-score neurons get penalty instead of -inf
    """
    def __init__(self, d_model: int, n_feature_qk: int, n_feature_v: int,
                 n_restore_qk: int, n_restore_v: int,
                 n_feature_know: int, n_restore_know: int,
                 rank: int = 16,
                 max_paths: int = 4,
                 fixed_tau: float = 0.0,
                 path_min_k: int = 8,
                 path_max_k: int = 16,
                 d_space: int = 64, router_dropout: float = 0.1,
                 attention_token_routing: bool = False,
                 knowledge_token_routing: bool = False,
                 # v18.1 parameters
                 use_soft_mask: bool = False,
                 learnable_tau: bool = False,
                 soft_mask_temp: float = 1.0,
                 soft_mask_penalty: float = 10.0,
                 **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.rank = rank
        self.max_paths = max_paths
        self.fixed_tau = fixed_tau
        self.path_min_k = path_min_k
        self.path_max_k = path_max_k
        self.attention_token_routing = attention_token_routing
        self.knowledge_token_routing = knowledge_token_routing

        # v18.1 parameters
        self.use_soft_mask = use_soft_mask
        self.learnable_tau = learnable_tau
        self.soft_mask_temp = soft_mask_temp
        self.soft_mask_penalty = soft_mask_penalty

        # Learnable tau parameters (v18.1) - token-level projection
        if learnable_tau:
            # Token-level tau projection (8 pools - Q/K separated)
            # Output: [fq, fk, fv, rq, rk, rv, feature_know, restore_know]
            self.tau_proj = nn.Linear(d_model, 8)
            nn.init.zeros_(self.tau_proj.weight)
            nn.init.constant_(self.tau_proj.bias, fixed_tau)

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
            n_feature_know, n_restore_know,
            d_space=d_space, dropout=router_dropout, fixed_tau=fixed_tau, **kwargs
        )

    def get_tau_all(self, x):
        """
        Compute tau_proj once for all pools.

        Args:
            x: [B, S, d_model] input tensor

        Returns:
            [B, S, 8] tau values for all pools, or None if not learnable
            Order: [fq, fk, fv, rq, rk, rv, feature_know, restore_know]
        """
        if self.learnable_tau:
            return self.tau_proj(x)  # [B, S, 8]
        return None

    def get_tau_reg_loss(self):
        """
        Compute tau regularization loss (v18.1 only).
        L2 regularization on tau projection weights.
        Returns 0 if learnable_tau=False.
        """
        if not self.learnable_tau:
            return 0.0

        # L2 regularization on tau projection weights and bias
        return (self.tau_proj.weight ** 2).mean() + (self.tau_proj.bias ** 2).mean()

    def get_all_tau_values(self):
        """Get all tau values as a dict (for logging). Returns bias values for learnable tau."""
        if self.learnable_tau:
            bias = self.tau_proj.bias.detach()
            return {
                'fq': bias[0].item(),
                'fk': bias[1].item(),
                'fv': bias[2].item(),
                'rq': bias[3].item(),
                'rk': bias[4].item(),
                'rv': bias[5].item(),
                'feature_know': bias[6].item(),
                'restore_know': bias[7].item(),
            }
        else:
            return {
                'fq': self.fixed_tau,
                'fk': self.fixed_tau,
                'fv': self.fixed_tau,
                'rq': self.fixed_tau,
                'rk': self.fixed_tau,
                'rv': self.fixed_tau,
                'feature_know': self.fixed_tau,
                'restore_know': self.fixed_tau,
            }

    def _threshold_select(self, scores, tau, path_min_k, path_max_k, max_paths):
        """
        Apply threshold selection with min/max guarantees and masked softmax.

        v18.0: Hard mask (scores > tau)
        v18.1: Gate-based soft mask: gate = sigmoid((scores - tau) / temp), weights = softmax(scores * gate)

        Args:
            scores: [B, S, N] neuron scores
            tau: scalar or tensor threshold
            path_min_k: minimum neurons to select
            path_max_k: max neurons per path
            max_paths: maximum number of paths

        Returns:
            weights: [B, S, N] softmax weights (masked)
            mask: [B, S, N] boolean mask of selected neurons
            gate: [B, S, N] gate values (v18.1 only, None for v18.0)
        """
        B, S, N = scores.shape

        if self.use_soft_mask:
            # v18.1: Gate-based soft mask approach
            # 1. Compute gate using sigmoid
            gate = torch.sigmoid((scores - tau) / self.soft_mask_temp)

            # 2. Apply gate to scores
            gated_scores = scores * gate

            # 3. Cap at path_max_k * max_paths neurons (mask out rest)
            max_neurons = path_max_k * max_paths
            if max_neurons < N:
                _, topk_idx = scores.topk(max_neurons, dim=-1)
                keep_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_idx, True)
                gated_scores = gated_scores.masked_fill(~keep_mask, float('-inf'))

            # 4. Softmax
            weights = F.softmax(gated_scores, dim=-1)

            # Hard mask for logging (gate > 0.5)
            mask = gate > 0.5

            return weights, mask, gate
        else:
            # v18.0: Hard mask approach
            # 1. Threshold mask
            mask = scores > tau

            # 2. Ensure minimum path_min_k neurons
            if path_min_k > 0:
                topk_vals, topk_idx = scores.topk(min(path_min_k, N), dim=-1)
                min_mask = torch.zeros_like(mask).scatter_(-1, topk_idx, True)
                mask = mask | min_mask

            # 3. Cap at path_max_k * max_paths neurons
            max_neurons = path_max_k * max_paths
            if max_neurons < N:
                topk_vals, topk_idx = scores.topk(max_neurons, dim=-1)
                max_mask = torch.zeros_like(mask).scatter_(-1, topk_idx, True)
                mask = mask & max_mask

            # 4. Masked softmax
            masked_scores = scores.masked_fill(~mask, float('-inf'))
            weights = F.softmax(masked_scores, dim=-1)

            return weights, mask, None

    def _chunk_to_paths(self, weights, mask, scores, path_max_k, max_paths):
        """
        Chunk selected neurons into path_max_k-sized paths by score ranking.

        Args:
            weights: [B, S, N] softmax weights
            mask: [B, S, N] selection mask
            scores: [B, S, N] original scores for ranking
            path_max_k: neurons per path
            max_paths: maximum number of paths

        Returns:
            path_weights_list: list of [B, S, N] weights (length = max_paths)
        """
        B, S, N = scores.shape

        # Sort neurons by score (descending)
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        sorted_weights = torch.gather(weights, dim=-1, index=sorted_indices)
        sorted_mask = torch.gather(mask.float(), dim=-1, index=sorted_indices)

        path_weights_list = []

        for p in range(max_paths):
            start_idx = p * path_max_k
            end_idx = min((p + 1) * path_max_k, N)

            if start_idx >= N:
                path_weights_list.append(torch.zeros_like(weights))
                continue

            # Create path weights
            path_weights = torch.zeros_like(weights)

            # Get indices and weights for this path's neurons
            path_indices = sorted_indices[:, :, start_idx:end_idx]
            path_w = sorted_weights[:, :, start_idx:end_idx]
            path_m = sorted_mask[:, :, start_idx:end_idx]

            # Only include weights for masked (selected) neurons
            path_w = path_w * path_m

            # Scatter weights back to original positions
            path_weights.scatter_(dim=-1, index=path_indices, src=path_w)

            path_weights_list.append(path_weights)

        return path_weights_list

    def get_attention_weights(self, x, importance, attention_mask=None, tau_all=None):
        """
        v18.0: Threshold + masked softmax multi-path routing for attention
        v18.1: Soft mask + learnable tau (when use_soft_mask=True)

        Args:
            tau_all: [B, S, 8] pre-computed tau values (optional, avoids recomputation)
                     Order: [fq, fk, fv, rq, rk, rv, feature_know, restore_know]

        Returns:
            path_weights_dict: dict with lists of path weights for each neuron type
            routing_info: dict with routing statistics
            aux_loss: auxiliary loss for load balancing
        """
        (fqk_logits_Q, fqk_logits_K, fv_logits,
         rqk_logits_Q, rqk_logits_K, rv_logits) = self.neuron_router.get_all_logits(x)

        # Get thresholds (v18.0: fixed scalar, v18.1: token-level [B, S, 1])
        # Q/K have separate tau values
        if self.learnable_tau:
            if tau_all is None:
                tau_all = self.tau_proj(x)  # fallback if not provided
            tau_fq = tau_all[..., 0:1]
            tau_fk = tau_all[..., 1:2]
            tau_fv = tau_all[..., 2:3]
            tau_rq = tau_all[..., 3:4]
            tau_rk = tau_all[..., 4:5]
            tau_rv = tau_all[..., 5:6]
        else:
            tau_fq = tau_fk = tau_fv = tau_rq = tau_rk = tau_rv = self.fixed_tau

        # Apply threshold selection (returns weights, mask, and gate for v18.1)
        fqk_weights_Q, fqk_mask_Q, fqk_gate_Q = self._threshold_select(
            fqk_logits_Q, tau_fq, self.path_min_k, self.path_max_k, self.max_paths)
        fqk_weights_K, fqk_mask_K, fqk_gate_K = self._threshold_select(
            fqk_logits_K, tau_fk, self.path_min_k, self.path_max_k, self.max_paths)
        fv_weights, fv_mask, fv_gate = self._threshold_select(
            fv_logits, tau_fv, self.path_min_k, self.path_max_k, self.max_paths)
        rqk_weights_Q, rqk_mask_Q, rqk_gate_Q = self._threshold_select(
            rqk_logits_Q, tau_rq, self.path_min_k, self.path_max_k, self.max_paths)
        rqk_weights_K, rqk_mask_K, rqk_gate_K = self._threshold_select(
            rqk_logits_K, tau_rk, self.path_min_k, self.path_max_k, self.max_paths)
        rv_weights, rv_mask, rv_gate = self._threshold_select(
            rv_logits, tau_rv, self.path_min_k, self.path_max_k, self.max_paths)

        # Chunk to paths
        fqk_paths_Q = self._chunk_to_paths(fqk_weights_Q, fqk_mask_Q, fqk_logits_Q, self.path_max_k, self.max_paths)
        fqk_paths_K = self._chunk_to_paths(fqk_weights_K, fqk_mask_K, fqk_logits_K, self.path_max_k, self.max_paths)
        fv_paths = self._chunk_to_paths(fv_weights, fv_mask, fv_logits, self.path_max_k, self.max_paths)
        rqk_paths_Q = self._chunk_to_paths(rqk_weights_Q, rqk_mask_Q, rqk_logits_Q, self.path_max_k, self.max_paths)
        rqk_paths_K = self._chunk_to_paths(rqk_weights_K, rqk_mask_K, rqk_logits_K, self.path_max_k, self.max_paths)
        rv_paths = self._chunk_to_paths(rv_weights, rv_mask, rv_logits, self.path_max_k, self.max_paths)

        # Compute aux_loss for load balancing (using softmax weights)
        aux_loss = 0.0
        if self.training:
            if attention_mask is not None:
                seq_mask = attention_mask.unsqueeze(-1).float()
                count = seq_mask.sum() + 1e-8
                usage_fqk_Q = (fqk_weights_Q * seq_mask).sum(dim=(0, 1)) / count
                usage_fqk_K = (fqk_weights_K * seq_mask).sum(dim=(0, 1)) / count
                usage_fv = (fv_weights * seq_mask).sum(dim=(0, 1)) / count
                usage_rqk_Q = (rqk_weights_Q * seq_mask).sum(dim=(0, 1)) / count
                usage_rqk_K = (rqk_weights_K * seq_mask).sum(dim=(0, 1)) / count
                usage_rv = (rv_weights * seq_mask).sum(dim=(0, 1)) / count
            else:
                usage_fqk_Q = fqk_weights_Q.mean(dim=(0, 1))
                usage_fqk_K = fqk_weights_K.mean(dim=(0, 1))
                usage_fv = fv_weights.mean(dim=(0, 1))
                usage_rqk_Q = rqk_weights_Q.mean(dim=(0, 1))
                usage_rqk_K = rqk_weights_K.mean(dim=(0, 1))
                usage_rv = rv_weights.mean(dim=(0, 1))

            target_fqk = 1.0 / self.n_feature_qk
            target_fv = 1.0 / self.n_feature_v
            target_rqk = 1.0 / self.n_restore_qk
            target_rv = 1.0 / self.n_restore_v

            aux_loss += ((usage_fqk_Q - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss += ((usage_fqk_K - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss += ((usage_fv - target_fv) ** 2).sum() * self.n_feature_v
            aux_loss += ((usage_rqk_Q - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss += ((usage_rqk_K - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss += ((usage_rv - target_rv) ** 2).sum() * self.n_restore_v

        # Package path weights
        path_weights = {
            'fqk_Q': fqk_paths_Q,
            'fqk_K': fqk_paths_K,
            'fv': fv_paths,
            'rqk_Q': rqk_paths_Q,
            'rqk_K': rqk_paths_K,
            'rv': rv_paths,
        }

        # Helper: average paths used per token (sum of fraction of tokens using each path)
        def avg_paths_per_token(paths):
            return sum((p.sum(dim=-1) > 0).float().mean().item() for p in paths)

        # Helper: average selected neurons per token
        def avg_selected_per_token(mask):
            # mask: [B, S, N] -> sum over N, mean over B, S
            return mask.float().sum(dim=-1).mean().item()

        # Helper: Q/K overlap ratio (how much Q and K share the same neurons)
        def qk_overlap(mask_q, mask_k):
            # mask: [B, S, N] boolean
            # overlap = intersection / max(q_count, k_count) per token, then average
            intersection = (mask_q & mask_k).float().sum(dim=-1)  # [B, S]
            q_count = mask_q.float().sum(dim=-1)  # [B, S]
            k_count = mask_k.float().sum(dim=-1)  # [B, S]
            max_count = torch.maximum(q_count, k_count).clamp(min=1)
            overlap = (intersection / max_count).mean().item()
            return overlap

        # Helper: get tau mean/std (for token-level tau)
        def tau_mean(tau):
            return tau.mean().item() if torch.is_tensor(tau) else tau

        def tau_std(tau):
            return tau.std().item() if torch.is_tensor(tau) else 0.0

        routing_info = {
            # Softmax weights (before mask multiplication) - for analysis
            'fqk_weights_Q': fqk_weights_Q.detach(),
            'fqk_weights_K': fqk_weights_K.detach(),
            'fv_weights': fv_weights.detach(),
            'rqk_weights_Q': rqk_weights_Q.detach(),
            'rqk_weights_K': rqk_weights_K.detach(),
            'rv_weights': rv_weights.detach(),
            # Binary selection masks (scores > tau) - for POS analysis
            'fqk_mask_Q': fqk_mask_Q.detach(),
            'fqk_mask_K': fqk_mask_K.detach(),
            'fv_mask': fv_mask.detach(),
            'rqk_mask_Q': rqk_mask_Q.detach(),
            'rqk_mask_K': rqk_mask_K.detach(),
            'rv_mask': rv_mask.detach(),
            # Average paths used per token
            'n_paths_fqk_Q': avg_paths_per_token(fqk_paths_Q),
            'n_paths_fqk_K': avg_paths_per_token(fqk_paths_K),
            'n_paths_fv': avg_paths_per_token(fv_paths),
            'n_paths_rqk_Q': avg_paths_per_token(rqk_paths_Q),
            'n_paths_rqk_K': avg_paths_per_token(rqk_paths_K),
            'n_paths_rv': avg_paths_per_token(rv_paths),
            # Average selected neurons per token
            'selected_fqk_Q': avg_selected_per_token(fqk_mask_Q),
            'selected_fqk_K': avg_selected_per_token(fqk_mask_K),
            'selected_fv': avg_selected_per_token(fv_mask),
            'selected_rqk_Q': avg_selected_per_token(rqk_mask_Q),
            'selected_rqk_K': avg_selected_per_token(rqk_mask_K),
            'selected_rv': avg_selected_per_token(rv_mask),
            # Q/K overlap ratio
            'overlap_fqk': qk_overlap(fqk_mask_Q, fqk_mask_K),
            'overlap_rqk': qk_overlap(rqk_mask_Q, rqk_mask_K),
            # Score statistics (logits mean ± std)
            'score_fqk_Q_mean': fqk_logits_Q.mean().item(),
            'score_fqk_Q_std': fqk_logits_Q.std().item(),
            'score_fqk_K_mean': fqk_logits_K.mean().item(),
            'score_fqk_K_std': fqk_logits_K.std().item(),
            'score_fv_mean': fv_logits.mean().item(),
            'score_fv_std': fv_logits.std().item(),
            'score_rqk_Q_mean': rqk_logits_Q.mean().item(),
            'score_rqk_Q_std': rqk_logits_Q.std().item(),
            'score_rqk_K_mean': rqk_logits_K.mean().item(),
            'score_rqk_K_std': rqk_logits_K.std().item(),
            'score_rv_mean': rv_logits.mean().item(),
            'score_rv_std': rv_logits.std().item(),
            # tau (v18.0: fixed scalar, v18.1: token-level [B, S, 1], Q/K separated)
            'tau_fq': tau_mean(tau_fq),
            'tau_fq_std': tau_std(tau_fq),
            'tau_fk': tau_mean(tau_fk),
            'tau_fk_std': tau_std(tau_fk),
            'tau_fv': tau_mean(tau_fv),
            'tau_fv_std': tau_std(tau_fv),
            'tau_rq': tau_mean(tau_rq),
            'tau_rq_std': tau_std(tau_rq),
            'tau_rk': tau_mean(tau_rk),
            'tau_rk_std': tau_std(tau_rk),
            'tau_rv': tau_mean(tau_rv),
            'tau_rv_std': tau_std(tau_rv),
            'learnable_tau': self.learnable_tau,
            'use_soft_mask': self.use_soft_mask,
            'token_routing': self.attention_token_routing,
        }

        # Add gate activation mean (v18.1 only)
        if self.use_soft_mask:
            routing_info['gate_fq'] = fqk_gate_Q.mean().item()
            routing_info['gate_fk'] = fqk_gate_K.mean().item()
            routing_info['gate_fv'] = fv_gate.mean().item()
            routing_info['gate_rq'] = rqk_gate_Q.mean().item()
            routing_info['gate_rk'] = rqk_gate_K.mean().item()
            routing_info['gate_rv'] = rv_gate.mean().item()

        # Update usage with mask (binary selection)
        if self.training:
            self.neuron_router.update_usage(fqk_mask_Q.float(), 'feature_q', attention_mask)
            self.neuron_router.update_usage(fqk_mask_K.float(), 'feature_k', attention_mask)
            self.neuron_router.update_usage(fv_mask.float(), 'feature_v', attention_mask)
            self.neuron_router.update_usage(rqk_mask_Q.float(), 'restore_q', attention_mask)
            self.neuron_router.update_usage(rqk_mask_K.float(), 'restore_k', attention_mask)
            self.neuron_router.update_usage(rv_mask.float(), 'restore_v', attention_mask)

        return path_weights, routing_info, aux_loss

    def get_knowledge_weights(self, x, importance, attention_mask=None, tau_all=None):
        """
        v18.0: Threshold + masked softmax multi-path routing for knowledge
        v18.1: Soft mask + learnable tau (when use_soft_mask=True)

        Args:
            tau_all: [B, S, 8] pre-computed tau values (optional, avoids recomputation)
                     Order: [fq, fk, fv, rq, rk, rv, feature_know, restore_know]
        """
        logits_f, logits_r = self.neuron_router.get_knowledge_logits(x)

        # Get thresholds (v18.0: fixed scalar, v18.1: token-level [B, S, 1])
        if self.learnable_tau:
            if tau_all is None:
                tau_all = self.tau_proj(x)  # fallback if not provided
            tau_f = tau_all[..., 6:7]
            tau_r = tau_all[..., 7:8]
        else:
            tau_f = tau_r = self.fixed_tau

        # Apply threshold selection (returns weights, mask, and gate for v18.1)
        f_weights, f_mask, f_gate = self._threshold_select(
            logits_f, tau_f, self.path_min_k, self.path_max_k, self.max_paths)
        r_weights, r_mask, r_gate = self._threshold_select(
            logits_r, tau_r, self.path_min_k, self.path_max_k, self.max_paths)

        # Chunk to paths
        f_paths = self._chunk_to_paths(f_weights, f_mask, logits_f, self.path_max_k, self.max_paths)
        r_paths = self._chunk_to_paths(r_weights, r_mask, logits_r, self.path_max_k, self.max_paths)

        # Update usage with mask (binary selection)
        if self.training:
            self.neuron_router.update_usage(f_mask.float(), 'feature_know', attention_mask)
            self.neuron_router.update_usage(r_mask.float(), 'restore_know', attention_mask)

        # Helper: average paths used per token
        def avg_paths_per_token(paths):
            return sum((p.sum(dim=-1) > 0).float().mean().item() for p in paths)

        # Helper: average selected neurons per token
        def avg_selected_per_token(mask):
            return mask.float().sum(dim=-1).mean().item()

        # Helper: get tau mean/std (for token-level tau)
        def tau_mean(tau):
            return tau.mean().item() if torch.is_tensor(tau) else tau

        def tau_std(tau):
            return tau.std().item() if torch.is_tensor(tau) else 0.0

        know_info = {
            # Softmax weights (before mask multiplication) - for analysis
            'feature_know_w': f_weights.detach(),
            'restore_know_w': r_weights.detach(),
            # Binary selection masks - for POS analysis
            'feature_know_mask': f_mask.detach(),
            'restore_know_mask': r_mask.detach(),
            'n_paths_feature': avg_paths_per_token(f_paths),
            'n_paths_restore': avg_paths_per_token(r_paths),
            # Average selected neurons per token
            'selected_feature': avg_selected_per_token(f_mask),
            'selected_restore': avg_selected_per_token(r_mask),
            # Score statistics (logits mean ± std)
            'score_feature_mean': logits_f.mean().item(),
            'score_feature_std': logits_f.std().item(),
            'score_restore_mean': logits_r.mean().item(),
            'score_restore_std': logits_r.std().item(),
            # tau (v18.0: fixed scalar, v18.1: token-level [B, S, 1])
            'tau_feature': tau_mean(tau_f),
            'tau_feature_std': tau_std(tau_f),
            'tau_restore': tau_mean(tau_r),
            'tau_restore_std': tau_std(tau_r),
        }

        # Add gate activation mean (v18.1 only)
        if self.use_soft_mask:
            know_info['gate_feature'] = f_gate.mean().item()
            know_info['gate_restore'] = r_gate.mean().item()

        return f_paths, r_paths, know_info


class AttentionCircuit(nn.Module):
    """
    v18.0: Multi-path attention circuit

    Each path generates Q, K, V independently, then all paths are summed.
    Attention operation remains unchanged.
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank

        self.expand_O = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def _process_single_path(self, x, fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w):
        """
        Process a single path to generate Q, K, V contributions.
        Uses batched einsum for better memory efficiency and throughput.
        """
        B, S, D = x.shape

        # Feature QK: batched einsum
        f_qk = self.shared_neurons.feature_qk_neurons  # [N, D, R]
        all_h_qk = torch.einsum('bsd,ndr->bsnr', x, f_qk)  # [B, S, N, R]
        h_q = torch.einsum('bsnr,bsn->bsr', all_h_qk, fqk_w_Q)  # [B, S, R]
        h_k = torch.einsum('bsnr,bsn->bsr', all_h_qk, fqk_w_K)  # [B, S, R]

        # Feature V: batched einsum
        f_v = self.shared_neurons.feature_v_neurons  # [N, D, R]
        all_h_v = torch.einsum('bsd,ndr->bsnr', x, f_v)  # [B, S, N, R]
        h_v = torch.einsum('bsnr,bsn->bsr', all_h_v, fv_w)  # [B, S, R]

        # Restore: batched einsum
        r_qk = self.shared_neurons.restore_qk_neurons  # [N, R, D]
        r_v = self.shared_neurons.restore_v_neurons  # [N, R, D]

        Q = torch.einsum('bsr,nrd,bsn->bsd', h_q, r_qk, rqk_w_Q)  # [B, S, D]
        K = torch.einsum('bsr,nrd,bsn->bsd', h_k, r_qk, rqk_w_K)  # [B, S, D]
        V = torch.einsum('bsr,nrd,bsn->bsd', h_v, r_v, rv_w)  # [B, S, D]

        return Q, K, V

    def forward(self, x, path_weights, attention_mask=None):
        """
        Args:
            x: [B, S, D] input
            path_weights: dict with lists of weights for each neuron type
                - 'fqk_Q': list of [B, S, N] weights
                - 'fqk_K': list of [B, S, N] weights
                - 'fv': list of [B, S, N] weights
                - 'rqk_Q': list of [B, S, N] weights
                - 'rqk_K': list of [B, S, N] weights
                - 'rv': list of [B, S, N] weights
        """
        B, S, D = x.shape

        # Stack path weights: [P, B, S, N]
        fqk_w_Q_stacked = torch.stack(path_weights['fqk_Q'], dim=0)
        fqk_w_K_stacked = torch.stack(path_weights['fqk_K'], dim=0)
        fv_w_stacked = torch.stack(path_weights['fv'], dim=0)
        rqk_w_Q_stacked = torch.stack(path_weights['rqk_Q'], dim=0)
        rqk_w_K_stacked = torch.stack(path_weights['rqk_K'], dim=0)
        rv_w_stacked = torch.stack(path_weights['rv'], dim=0)

        # Feature QK: common computation once
        f_qk = self.shared_neurons.feature_qk_neurons  # [N, D, R]
        all_h_qk = torch.einsum('bsd,ndr->bsnr', x, f_qk)  # [B, S, N, R]

        # Path-parallel bottleneck for Q, K
        h_q_all = torch.einsum('bsnr,pbsn->pbsr', all_h_qk, fqk_w_Q_stacked)  # [P, B, S, R]
        h_k_all = torch.einsum('bsnr,pbsn->pbsr', all_h_qk, fqk_w_K_stacked)  # [P, B, S, R]

        # Feature V: common computation once
        f_v = self.shared_neurons.feature_v_neurons  # [N, D, R]
        all_h_v = torch.einsum('bsd,ndr->bsnr', x, f_v)  # [B, S, N, R]

        # Path-parallel bottleneck for V
        h_v_all = torch.einsum('bsnr,pbsn->pbsr', all_h_v, fv_w_stacked)  # [P, B, S, R]

        # Path-parallel restore
        r_qk = self.shared_neurons.restore_qk_neurons  # [N, R, D]
        r_v = self.shared_neurons.restore_v_neurons  # [N, R, D]

        Q_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_q_all, r_qk, rqk_w_Q_stacked)  # [P, B, S, D]
        K_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_k_all, r_qk, rqk_w_K_stacked)  # [P, B, S, D]
        V_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_v_all, r_v, rv_w_stacked)  # [P, B, S, D]

        # Sum over paths (each path has own bottleneck, total capacity = R * n_paths)
        Q_total = Q_all.sum(dim=0)  # [B, S, D]
        K_total = K_all.sum(dim=0)
        V_total = V_all.sum(dim=0)

        # Multi-head attention
        Q = Q_total.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K_total.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V_total.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        # FlashAttention (PyTorch 2.0+)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            is_causal=True,
            dropout_p=dropout_p,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        return output, None


class KnowledgeCircuit(nn.Module):
    """
    v18.0: Multi-path Knowledge Circuit

    Each path processes independently, then all paths are summed.
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_feature_know: int,
        n_restore_know: int,
        knowledge_rank: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.knowledge_rank = knowledge_rank
        self.dropout = nn.Dropout(dropout)

    def _process_single_path(self, x, feature_w, restore_w):
        """
        Process a single path.
        Uses batched einsum for better memory efficiency and throughput.
        """
        # Feature: batched einsum
        f_know = self.shared_neurons.feature_know  # [N, D, R]
        all_h = torch.einsum('bsd,ndr->bsnr', x, f_know)  # [B, S, N, R]
        h = torch.einsum('bsnr,bsn->bsr', all_h, feature_w)  # [B, S, R]

        # Restore: batched einsum
        r_know = self.shared_neurons.restore_know  # [N, R, D]
        output = torch.einsum('bsr,nrd,bsn->bsd', h, r_know, restore_w)  # [B, S, D]

        return output

    def forward(self, x, feature_paths, restore_paths, attention_mask=None):
        """
        Args:
            x: [B, S, D] input
            feature_paths: list of [B, S, N] weights
            restore_paths: list of [B, S, N] weights
        """
        n_paths = max(len(feature_paths), len(restore_paths))

        # Stack path weights: [P, B, S, N]
        feature_stacked = torch.stack(feature_paths, dim=0)
        restore_stacked = torch.stack(restore_paths, dim=0)

        # Feature: common computation once
        f_know = self.shared_neurons.feature_know  # [N, D, R]
        all_h = torch.einsum('bsd,ndr->bsnr', x, f_know)  # [B, S, N, R]

        # Path-parallel bottleneck
        h_all = torch.einsum('bsnr,pbsn->pbsr', all_h, feature_stacked)  # [P, B, S, R]

        # Path-parallel restore
        r_know = self.shared_neurons.restore_know  # [N, R, D]
        output_all = torch.einsum('pbsr,nrd,pbsn->pbsd', h_all, r_know, restore_stacked)  # [P, B, S, D]

        # Sum over paths
        output = output_all.sum(dim=0)  # [B, S, D]

        return self.dropout(output)


class DAWNBlock(nn.Module):
    """DAWN v18.0 block: multi-path routing"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_feature_know: int,
        n_restore_know: int,
        knowledge_rank: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = AttentionCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.knowledge = KnowledgeCircuit(
            shared_neurons, d_model, n_feature_know, n_restore_know,
            knowledge_rank=knowledge_rank,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, importance, global_routers: GlobalRouters, attention_mask=None):
        # Attention
        normed_x_attn = self.norm1(x)
        tau_all_attn = global_routers.get_tau_all(normed_x_attn)
        path_weights, attn_info, attn_aux_loss = global_routers.get_attention_weights(
            normed_x_attn, importance, attention_mask, tau_all=tau_all_attn
        )
        attn_out, _ = self.attn(normed_x_attn, path_weights, attention_mask)
        x = x + attn_out

        # Knowledge
        normed_x_know = self.norm2(x)
        tau_all_know = global_routers.get_tau_all(normed_x_know)
        feature_paths, restore_paths, know_info = global_routers.get_knowledge_weights(
            normed_x_know, importance, attention_mask, tau_all=tau_all_know
        )
        know_out = self.knowledge(normed_x_know, feature_paths, restore_paths, attention_mask)
        x = x + know_out

        # Routing info (all scalar values - no tensor storage to avoid memory leak)
        routing_info = {
            'attention': attn_info,
            'knowledge': know_info,
            'attn_out_norm': attn_out.norm(dim=-1).mean().item(),
            'know_out_norm': know_out.norm(dim=-1).mean().item(),
        }

        return x, routing_info, attn_aux_loss


class DAWN(nn.Module):
    """
    DAWN v18.1: Soft Mask + Learnable Tau Routing

    Key Features:
    - Based on v18.0 multi-path routing
    - Soft mask: sigmoid((score - tau) / temp) for differentiable threshold
    - Learnable tau: separate nn.Parameter for each neuron type
    - Penalty-based masking: low-score neurons get penalty instead of -inf
    - Gradients flow through all neurons (differentiable routing)

    Architecture:
    - UnifiedNeuronRouter: same as v18.0
    - GlobalRouters: soft mask + learnable tau parameters
    - AttentionCircuit: Multi-path Q,K,V summation (same as v18.0)
    - KnowledgeCircuit: Multi-path summation (same as v18.0)
    """
    __version__ = "18.1"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 8,
        rank: int = 16,  # v18: reduced from 64
        max_seq_len: int = 512,
        state_dim: int = 64,
        d_space: int = 64,
        # v18: Multi-path parameters
        max_paths: int = 4,
        fixed_tau: float = 0.0,
        path_min_k: int = 8,
        path_max_k: int = 16,
        # v18.1: Soft mask parameters (defaults for v18.1)
        use_soft_mask: bool = True,
        learnable_tau: bool = True,
        soft_mask_temp: float = 1.0,
        soft_mask_penalty: float = 10.0,
        # Attention - shared Q/K pool
        n_feature_qk: int = 56,
        n_feature_v: int = 24,
        n_restore_qk: int = 56,
        n_restore_v: int = 24,
        # Knowledge - Feature/Restore separation
        n_feature_know: int = 24,
        n_restore_know: int = 24,
        knowledge_rank: int = 128,
        # Others
        dropout: float = 0.1,
        attention_token_routing: bool = False,
        knowledge_token_routing: bool = False,
        router_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        use_ssm_context: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Validation checks
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rank = rank
        self.knowledge_rank = knowledge_rank
        self.max_seq_len = max_seq_len
        self.state_dim = state_dim
        self.d_space = d_space
        self.dropout_rate = dropout
        self.attention_token_routing = attention_token_routing
        self.knowledge_token_routing = knowledge_token_routing
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context

        # v18 specific
        self.max_paths = max_paths
        self.fixed_tau = fixed_tau
        self.path_min_k = path_min_k
        self.path_max_k = path_max_k

        # v18.1 specific
        self.use_soft_mask = use_soft_mask
        self.learnable_tau = learnable_tau
        self.soft_mask_temp = soft_mask_temp
        self.soft_mask_penalty = soft_mask_penalty

        # Neuron counts
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know

        # v15 compat
        self.n_feature = n_feature_qk
        self.n_neurons = n_feature_qk
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.global_ssm = GlobalSSM(d_model, state_dim, return_context=use_ssm_context)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank=rank,
            n_feature_qk=n_feature_qk, n_feature_v=n_feature_v,
            n_restore_qk=n_restore_qk, n_restore_v=n_restore_v,
            n_feature_know=n_feature_know, n_restore_know=n_restore_know,
            knowledge_rank=knowledge_rank,
        )

        self.router = GlobalRouters(
            d_model=d_model,
            n_feature_qk=n_feature_qk, n_feature_v=n_feature_v,
            n_restore_qk=n_restore_qk, n_restore_v=n_restore_v,
            n_feature_know=n_feature_know, n_restore_know=n_restore_know,
            rank=rank,
            max_paths=max_paths,
            fixed_tau=fixed_tau,
            path_min_k=path_min_k,
            path_max_k=path_max_k,
            d_space=d_space, router_dropout=router_dropout,
            attention_token_routing=attention_token_routing,
            knowledge_token_routing=knowledge_token_routing,
            # v18.1 parameters
            use_soft_mask=use_soft_mask,
            learnable_tau=learnable_tau,
            soft_mask_temp=soft_mask_temp,
            soft_mask_penalty=soft_mask_penalty,
        )

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, n_feature_know=n_feature_know, n_restore_know=n_restore_know,
                knowledge_rank=knowledge_rank,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.aux_loss = 0.0

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None, return_routing_info=False):
        B, S = input_ids.shape
        device = input_ids.device

        # Validate sequence length
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        # SSM only skipped when BOTH attention and knowledge use token routing
        if self.attention_token_routing and self.knowledge_token_routing:
            importance = None
            context = None
        else:
            importance, context, raw_importance = self.global_ssm(x, attention_mask)
            if context is not None:
                if attention_mask is not None:
                    context = context * attention_mask.unsqueeze(-1)
                x = x + context

        routing_infos = []
        total_aux_loss = 0.0

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, routing_info, aux_loss = checkpoint(
                    layer, x, importance, self.router, attention_mask,
                    use_reentrant=False
                )
            else:
                x, routing_info, aux_loss = layer(x, importance, self.router, attention_mask)

            routing_infos.append(routing_info)
            total_aux_loss += aux_loss

        self.aux_loss = total_aux_loss
        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().long()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)
            if return_routing_info:
                return loss, logits, routing_infos
            return loss, logits

        if return_routing_info:
            return logits, routing_infos
        return logits

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank': self.rank, 'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'max_paths': self.max_paths,
            'fixed_tau': self.fixed_tau,
            'path_min_k': self.path_min_k,
            'path_max_k': self.path_max_k,
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'n_restore_qk': self.n_restore_qk, 'n_restore_v': self.n_restore_v,
            'n_feature_know': self.n_feature_know, 'n_restore_know': self.n_restore_know,
            'state_dim': self.state_dim, 'd_space': self.d_space,
            'knowledge_token_routing': self.knowledge_token_routing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: Soft Mask + Learnable Tau Routing",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  max_paths={self.max_paths}, fixed_tau={self.fixed_tau}, path_min_k={self.path_min_k}, path_max_k={self.path_max_k}",
            f"  soft_mask_temp={self.soft_mask_temp}, soft_mask_penalty={self.soft_mask_penalty}",
            f"  max_seq_len={self.max_seq_len}, state_dim={self.state_dim}, dropout={self.dropout_rate}",
            f"",
            f"  [Attention - Q/K Shared Pool] (soft mask + learnable tau, max_paths={self.max_paths})",
            f"  Feature_QK: {self.n_feature_qk} × {self.d_model} × {self.rank}",
            f"  Feature_V: {self.n_feature_v} × {self.d_model} × {self.rank}",
            f"  Restore_QK: {self.n_restore_qk} × {self.rank} × {self.d_model}",
            f"  Restore_V: {self.n_restore_v} × {self.rank} × {self.d_model}",
            f"",
            f"  [Knowledge - Feature-Restore] (soft mask + learnable tau, max_paths={self.max_paths})",
            f"  Feature_Know: {self.n_feature_know} × {self.d_model} × {self.knowledge_rank}",
            f"  Restore_Know: {self.n_restore_know} × {self.knowledge_rank} × {self.d_model}",
            f"",
            f"  [Router - Soft Mask + Learnable Tau]",
            f"  d_space={self.d_space}, router_dropout={self.router_dropout}",
            f"  attention_token_routing={self.attention_token_routing}, knowledge_token_routing={self.knowledge_token_routing}",
            f"  use_ssm_context={self.use_ssm_context}",
            f"",
            f"  [Other]",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
        ]

    def knowledge_diversity_loss(self):
        feat_know = self.shared_neurons.feature_know
        rest_know = self.shared_neurons.restore_know

        feat_flat = feat_know.view(feat_know.size(0), -1)
        feat_norm = F.normalize(feat_flat, dim=-1)
        feat_sim = torch.matmul(feat_norm, feat_norm.T)
        mask_f = ~torch.eye(feat_sim.size(0), dtype=torch.bool, device=feat_sim.device)
        feat_loss = feat_sim[mask_f].abs().mean()

        rest_flat = rest_know.view(rest_know.size(0), -1)
        rest_norm = F.normalize(rest_flat, dim=-1)
        rest_sim = torch.matmul(rest_norm, rest_norm.T)
        mask_r = ~torch.eye(rest_sim.size(0), dtype=torch.bool, device=rest_sim.device)
        rest_loss = rest_sim[mask_r].abs().mean()

        return (feat_loss + rest_loss) / 2

    def orthogonality_loss(self):
        I = torch.eye(self.rank, device=self.shared_neurons.f_neurons.device).unsqueeze(0)

        W_fqk = self.shared_neurons.feature_qk_neurons
        WtW_fqk = torch.bmm(W_fqk.transpose(1, 2), W_fqk)
        loss_fqk = ((WtW_fqk - I) ** 2).mean()

        W_fv = self.shared_neurons.feature_v_neurons
        WtW_fv = torch.bmm(W_fv.transpose(1, 2), W_fv)
        loss_fv = ((WtW_fv - I) ** 2).mean()

        W_rqk = self.shared_neurons.restore_qk_neurons
        WWt_rqk = torch.bmm(W_rqk, W_rqk.transpose(1, 2))
        loss_rqk = ((WWt_rqk - I) ** 2).mean()

        W_rv = self.shared_neurons.restore_v_neurons
        WWt_rv = torch.bmm(W_rv, W_rv.transpose(1, 2))
        loss_rv = ((WWt_rv - I) ** 2).mean()

        I_know = torch.eye(self.knowledge_rank, device=self.shared_neurons.feature_know.device).unsqueeze(0)

        W_fknow = self.shared_neurons.feature_know
        WtW_fknow = torch.bmm(W_fknow.transpose(1, 2), W_fknow)
        loss_fknow = ((WtW_fknow - I_know) ** 2).mean()

        W_rknow = self.shared_neurons.restore_know
        WWt_rknow = torch.bmm(W_rknow, W_rknow.transpose(1, 2))
        loss_rknow = ((WWt_rknow - I_know) ** 2).mean()

        return (loss_fqk + loss_fv + loss_rqk + loss_rv + loss_fknow + loss_rknow) / 6

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters()) / 1e6
        attn_neurons = self.n_feature_qk + self.n_feature_v + self.n_restore_qk + self.n_restore_v
        know_neurons = self.n_feature_know + self.n_restore_know
        return (
            f"DAWN v{self.__version__}: Soft Mask + Learnable Tau\n"
            f"  Params: {params:.1f}M\n"
            f"  rank={self.rank}, max_paths={self.max_paths}, path_min_k={self.path_min_k}, path_max_k={self.path_max_k}\n"
            f"  soft_mask_temp={self.soft_mask_temp}, soft_mask_penalty={self.soft_mask_penalty}\n"
            f"  Attention: Feature_QK={self.n_feature_qk}, Feature_V={self.n_feature_v}\n"
            f"            Restore_QK={self.n_restore_qk}, Restore_V={self.n_restore_v}\n"
            f"  Knowledge: Feature={self.n_feature_know}, Restore={self.n_restore_know}\n"
            f"  Total neurons: {attn_neurons} (attn) + {know_neurons} (know) = {attn_neurons + know_neurons}"
        )
