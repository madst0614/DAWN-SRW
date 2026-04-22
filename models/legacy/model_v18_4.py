"""
DAWN v18.4: Relative Confidence Scaling

Key Concept:
- Based on v18.3 with relative confidence scaling
- Gate: gate = ReLU(scores - tau), can be 0
- Confidence: confidence = gate / gate_sum (relative, sums to 1)
- Scaled weights: weights * confidence

Changes from v18.3:
- confidence = gate / (gate + 1) → gate / gate_sum
- Relative scaling instead of absolute
- tau decoupled from load-balance loss pressure
- True zero possible (hard sparsity)
- Healthy gradients (no vanishing)

Benefits over v18.3:
- tau stays in reasonable range (-3 ~ +3 instead of -170)
- conf distribution is differentiated (not all 0.99)
- lb_loss doesn't push tau to extremes

Architecture:
- UnifiedNeuronRouter: same as v18.0
- GlobalRouters: Relative confidence gating + learnable tau (8 pools)
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

        # LayerNorm for each projection output (separate for each pool)
        self.norm_fqk_Q = nn.LayerNorm(d_space)
        self.norm_fqk_K = nn.LayerNorm(d_space)
        self.norm_fv = nn.LayerNorm(d_space)
        self.norm_rqk_Q = nn.LayerNorm(d_space)
        self.norm_rqk_K = nn.LayerNorm(d_space)
        self.norm_rv = nn.LayerNorm(d_space)
        self.norm_feature_know = nn.LayerNorm(d_space)
        self.norm_restore_know = nn.LayerNorm(d_space)

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
        h_feature_know = self.norm_feature_know(self.dropout(self.proj_feature_know(x)))
        emb_feature_know = emb_norm[self.restore_v_end:self.feature_know_end]
        logits_feature_know = torch.einsum('bsd,nd->bsn', h_feature_know, emb_feature_know)

        # Restore_know
        h_restore_know = self.norm_restore_know(self.dropout(self.proj_restore_know(x)))
        emb_restore_know = emb_norm[self.feature_know_end:]
        logits_restore_know = torch.einsum('bsd,nd->bsn', h_restore_know, emb_restore_know)

        return logits_feature_know, logits_restore_know

    def get_all_logits(self, x):
        """6 attention logits at once"""
        emb_norm = F.normalize(self.neuron_emb, dim=-1)

        all_proj = self.dropout(self.proj_all(x))
        h_fqk_Q, h_fqk_K, h_fv, h_rqk_Q, h_rqk_K, h_rv = all_proj.chunk(6, dim=-1)

        # Apply LayerNorm to each projection (separate for each pool)
        h_fqk_Q = self.norm_fqk_Q(h_fqk_Q)
        h_fqk_K = self.norm_fqk_K(h_fqk_K)
        h_fv = self.norm_fv(h_fv)
        h_rqk_Q = self.norm_rqk_Q(h_rqk_Q)
        h_rqk_K = self.norm_rqk_K(h_rqk_K)
        h_rv = self.norm_rv(h_rv)

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

        # Detach to prevent memory leak from computation graph retention
        usage = usage.detach()

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
    v18.2: ReLU-masked learnable tau routing

    Uses:
    1. ReLU-based threshold mask: mask = (scores - tau) > 0
    2. Learnable tau: token-level tau projection (8 pools, Q/K separated)
    3. Cap at path_max_k * max_paths neurons
    4. Masked softmax for differentiable weighting
    5. Chunking into multiple paths by score ranking
    """
    def __init__(self, d_model: int, n_feature_qk: int, n_feature_v: int,
                 n_restore_qk: int, n_restore_v: int,
                 n_feature_know: int, n_restore_know: int,
                 rank: int = 16,
                 max_paths: int = 4,
                 fixed_tau: float = 0.0,
                 path_max_k: int = 16,
                 d_space: int = 64, router_dropout: float = 0.1,
                 attention_token_routing: bool = False,
                 knowledge_token_routing: bool = False,
                 learnable_tau: bool = True,
                 tau_reg_weight: float = 0.0,
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
        self.path_max_k = path_max_k
        self.attention_token_routing = attention_token_routing
        self.knowledge_token_routing = knowledge_token_routing
        self.learnable_tau = learnable_tau
        self.tau_reg_weight = tau_reg_weight
        self.inference_hard_mask = False  # Set True for clean hard mask during inference
        # ============================================================
        # MODE FLAGS (mutually exclusive usage patterns)
        # ============================================================
        # debug_mode: Enables routing stats (.item() calls) for 100-step logging
        #   - train.py toggles this on log steps only via set_v18_debug_mode()
        #   - Adds GPU sync overhead, use sparingly
        # store_pref_tensors: Enables storing pref/weight tensors for detailed analysis
        #   - analyze_dawn.py sets this explicitly when needed
        #   - Large memory footprint, never enable during training/validation
        # ============================================================
        self.debug_mode = False
        self.store_pref_tensors = False
        self.store_path_weights = False  # For returning path_weights in routing_info

        # Learnable tau parameters - token-level projection
        if learnable_tau:
            # Token-level tau projection (8 pools - Q/K separated)
            # Output: [fq, fk, fv, rq, rk, rv, feature_know, restore_know]
            self.tau_proj = nn.Linear(d_model, 8)
            nn.init.zeros_(self.tau_proj.weight)
            # Initialize bias to -0.5: relative tau starts at mean - 0.5σ
            nn.init.constant_(self.tau_proj.bias, -0.5)

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
        DEPRECATED in v18.4: tau_reg is now computed inside routing functions
        and added to aux_loss with tau_reg_weight. This method returns 0.

        For v18.4, use tau_reg_weight parameter in model config instead.
        tau_reg = relu(tau - score_mean) penalizes tau > score_mean.
        """
        return 0.0

    def get_all_tau_offset_values(self):
        """
        Get all tau offset values as a dict (for logging).

        v18.4: Returns tau_offset values (in std units from score mean).
        Actual tau = score_mean + tau_offset * score_std

        A tau_offset of -0.5 means threshold is 0.5 std below the mean.
        """
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
                'fq': 0.0,
                'fk': 0.0,
                'fv': 0.0,
                'rq': 0.0,
                'rk': 0.0,
                'rv': 0.0,
                'feature_know': 0.0,
                'restore_know': 0.0,
            }

    def _topk_select_and_chunk(self, scores, tau, path_max_k, max_paths):
        """
        Optimized routing: top-k selection → tau threshold → exp gate → chunking

        v18.4: Relative tau (calculated from full scores by caller)
        - tau = score_mean + tau_offset * score_std (relative to full neuron pool)
        - gate = score - tau (positive) or 1e-8 * exp(score - tau) (negative, for gradient)
        - exp_gate = exp(gate) - 1 (amplifies differences, 0 when gate=0)
        - gate_strength = tanh(max(exp_gate))
        - scaled_weights = (exp_gate / sum) * gate_strength

        Args:
            scores: [B, S, N] neuron scores
            tau: scalar or [B, S, 1] tensor (threshold value)
            path_max_k: neurons per path
            max_paths: maximum number of paths

        Returns:
            path_weights_list: list of [B, S, N] weights (length = max_paths)
            weights: [B, S, N] sparse weights for aux_loss
            mask: [B, S, N] boolean mask for statistics
            gate: [B, S, k] gate values (with gradient flow)
            scaled_weights: [B, S, k] normalized exp gate weights
        """
        B, S, N = scores.shape
        k = min(path_max_k * max_paths, N)

        # 1. Top-k selection (sorted by descending scores)
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1, sorted=True)

        # 2. Threshold mask for top-k neurons (tau already computed by caller)
        topk_mask = (topk_scores > tau)

        # 4. Gate with gradient flow for dead neurons
        raw_gate = topk_scores - tau
        gate = torch.where(
            raw_gate > 0,
            raw_gate,
            1e-8 * torch.exp(raw_gate)  # 음수여도 작은 값 + gradient 흐름
        )

        # 5. Exponential scaling for sparsity + normalization
        exp_gate = torch.exp(gate) - 1  # 차이 극대화, gate=0이면 0
        exp_gate_sum = exp_gate.sum(dim=-1, keepdim=True)

        # 비율 계산
        ratio_weights = torch.where(
            exp_gate_sum > 1e-8,
            exp_gate / (exp_gate_sum + 1e-8),
            exp_gate * 1e-8  # gradient 유지
        )

        # gate_strength: max exp_gate의 tanh로 전체 강도 조절
        gate_strength = torch.tanh(exp_gate.max(dim=-1, keepdim=True).values)
        scaled_weights = ratio_weights * gate_strength

        # 6. Chunk to paths (already sorted by topk)
        out_dtype = scaled_weights.dtype
        path_weights_list = []
        for p in range(max_paths):
            start_idx = p * path_max_k
            end_idx = min((p + 1) * path_max_k, k)

            if start_idx >= k:
                path_weights_list.append(torch.zeros(B, S, N, device=scores.device, dtype=out_dtype))
                continue

            path_weights = torch.zeros(B, S, N, device=scores.device, dtype=out_dtype)
            path_indices = topk_indices[:, :, start_idx:end_idx]
            path_w = scaled_weights[:, :, start_idx:end_idx]

            path_weights.scatter_(dim=-1, index=path_indices, src=path_w)
            path_weights_list.append(path_weights)

        # 7. Create sparse full weights for aux_loss (use scaled_weights)
        weights = torch.zeros(B, S, N, device=scores.device, dtype=out_dtype)
        weights.scatter_(dim=-1, index=topk_indices, src=scaled_weights)

        # 8. Create full mask for statistics
        mask = torch.zeros(B, S, N, device=scores.device, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk_indices, src=topk_mask)

        return path_weights_list, weights, mask, gate, scaled_weights

    def get_attention_weights(self, x, importance, attention_mask=None, tau_all=None):
        """
        v18.4: Relative tau routing - tau calculated from full scores

        tau = score_mean + tau_offset * score_std (relative to full neuron pool)

        Args:
            tau_all: [B, S, 8] pre-computed tau_offset values (optional)
                     Order: [fq, fk, fv, rq, rk, rv, feature_know, restore_know]

        Returns:
            path_weights_dict: dict with lists of path weights for each neuron type
            routing_info: dict with routing statistics
            aux_loss: auxiliary loss for load balancing
        """
        (fqk_logits_Q, fqk_logits_K, fv_logits,
         rqk_logits_Q, rqk_logits_K, rv_logits) = self.neuron_router.get_all_logits(x)

        # Get tau_offset and compute tau from full scores
        if self.learnable_tau:
            if tau_all is None:
                tau_all = self.tau_proj(x)
            tau_offset_fq = tau_all[..., 0:1]
            tau_offset_fk = tau_all[..., 1:2]
            tau_offset_fv = tau_all[..., 2:3]
            tau_offset_rq = tau_all[..., 3:4]
            tau_offset_rk = tau_all[..., 4:5]
            tau_offset_rv = tau_all[..., 5:6]

            # Compute tau from full scores: tau = mean + offset * std
            tau_fq = fqk_logits_Q.mean(dim=-1, keepdim=True) + tau_offset_fq * (fqk_logits_Q.std(dim=-1, keepdim=True) + 1e-8)
            tau_fk = fqk_logits_K.mean(dim=-1, keepdim=True) + tau_offset_fk * (fqk_logits_K.std(dim=-1, keepdim=True) + 1e-8)
            tau_fv = fv_logits.mean(dim=-1, keepdim=True) + tau_offset_fv * (fv_logits.std(dim=-1, keepdim=True) + 1e-8)
            tau_rq = rqk_logits_Q.mean(dim=-1, keepdim=True) + tau_offset_rq * (rqk_logits_Q.std(dim=-1, keepdim=True) + 1e-8)
            tau_rk = rqk_logits_K.mean(dim=-1, keepdim=True) + tau_offset_rk * (rqk_logits_K.std(dim=-1, keepdim=True) + 1e-8)
            tau_rv = rv_logits.mean(dim=-1, keepdim=True) + tau_offset_rv * (rv_logits.std(dim=-1, keepdim=True) + 1e-8)
        else:
            # For fixed tau, use score mean (offset = 0)
            tau_offset_fq = tau_offset_fk = tau_offset_fv = 0.0
            tau_offset_rq = tau_offset_rk = tau_offset_rv = 0.0
            tau_fq = fqk_logits_Q.mean(dim=-1, keepdim=True)
            tau_fk = fqk_logits_K.mean(dim=-1, keepdim=True)
            tau_fv = fv_logits.mean(dim=-1, keepdim=True)
            tau_rq = rqk_logits_Q.mean(dim=-1, keepdim=True)
            tau_rk = rqk_logits_K.mean(dim=-1, keepdim=True)
            tau_rv = rv_logits.mean(dim=-1, keepdim=True)

        # Top-k selection with tau threshold
        fqk_paths_Q, fqk_weights_Q, fqk_mask_Q, fqk_gate_Q, fqk_conf_Q = self._topk_select_and_chunk(
            fqk_logits_Q, tau_fq, self.path_max_k, self.max_paths)
        fqk_paths_K, fqk_weights_K, fqk_mask_K, fqk_gate_K, fqk_conf_K = self._topk_select_and_chunk(
            fqk_logits_K, tau_fk, self.path_max_k, self.max_paths)
        fv_paths, fv_weights, fv_mask, fv_gate, fv_conf = self._topk_select_and_chunk(
            fv_logits, tau_fv, self.path_max_k, self.max_paths)
        rqk_paths_Q, rqk_weights_Q, rqk_mask_Q, rqk_gate_Q, rqk_conf_Q = self._topk_select_and_chunk(
            rqk_logits_Q, tau_rq, self.path_max_k, self.max_paths)
        rqk_paths_K, rqk_weights_K, rqk_mask_K, rqk_gate_K, rqk_conf_K = self._topk_select_and_chunk(
            rqk_logits_K, tau_rk, self.path_max_k, self.max_paths)
        rv_paths, rv_weights, rv_mask, rv_gate, rv_conf = self._topk_select_and_chunk(
            rv_logits, tau_rv, self.path_max_k, self.max_paths)

        # Compute aux_loss for load balancing (score-based: softmax before top-k)
        aux_loss = 0.0
        if self.training:
            fqk_pref_Q = F.softmax(fqk_logits_Q, dim=-1)
            fqk_pref_K = F.softmax(fqk_logits_K, dim=-1)
            fv_pref = F.softmax(fv_logits, dim=-1)
            rqk_pref_Q = F.softmax(rqk_logits_Q, dim=-1)
            rqk_pref_K = F.softmax(rqk_logits_K, dim=-1)
            rv_pref = F.softmax(rv_logits, dim=-1)

            if attention_mask is not None:
                seq_mask = attention_mask.unsqueeze(-1).float()
                count = seq_mask.sum() + 1e-8
                usage_fqk_Q = (fqk_pref_Q * seq_mask).sum(dim=(0, 1)) / count
                usage_fqk_K = (fqk_pref_K * seq_mask).sum(dim=(0, 1)) / count
                usage_fv = (fv_pref * seq_mask).sum(dim=(0, 1)) / count
                usage_rqk_Q = (rqk_pref_Q * seq_mask).sum(dim=(0, 1)) / count
                usage_rqk_K = (rqk_pref_K * seq_mask).sum(dim=(0, 1)) / count
                usage_rv = (rv_pref * seq_mask).sum(dim=(0, 1)) / count
            else:
                usage_fqk_Q = fqk_pref_Q.mean(dim=(0, 1))
                usage_fqk_K = fqk_pref_K.mean(dim=(0, 1))
                usage_fv = fv_pref.mean(dim=(0, 1))
                usage_rqk_Q = rqk_pref_Q.mean(dim=(0, 1))
                usage_rqk_K = rqk_pref_K.mean(dim=(0, 1))
                usage_rv = rv_pref.mean(dim=(0, 1))

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

        # Compute tau regularization loss (penalize positive tau_offset)
        if self.training and self.learnable_tau and self.tau_reg_weight > 0:
            tau_reg = F.relu(tau_offset_fq).mean()
            tau_reg += F.relu(tau_offset_fk).mean()
            tau_reg += F.relu(tau_offset_fv).mean()
            tau_reg += F.relu(tau_offset_rq).mean()
            tau_reg += F.relu(tau_offset_rk).mean()
            tau_reg += F.relu(tau_offset_rv).mean()
            aux_loss += tau_reg * self.tau_reg_weight

        # Package path weights
        path_weights = {
            'fqk_Q': fqk_paths_Q,
            'fqk_K': fqk_paths_K,
            'fv': fv_paths,
            'rqk_Q': rqk_paths_Q,
            'rqk_K': rqk_paths_K,
            'rv': rv_paths,
        }

        # ============================================================
        # ROUTING INFO (for logging/analysis only)
        # ============================================================
        # IMPORTANT: debug_mode=True only on log steps, ~20 metrics, torch.no_grad
        # ============================================================
        if self.debug_mode:
            with torch.no_grad():
                routing_info = {
                    'top_k': self.path_max_k * self.max_paths,
                    # Selected neurons (pass rate = selected / top_k)
                    'selected_fqk_Q': fqk_mask_Q.float().sum(dim=-1).mean().item(),
                    'selected_fqk_K': fqk_mask_K.float().sum(dim=-1).mean().item(),
                    'selected_fv': fv_mask.float().sum(dim=-1).mean().item(),
                    'selected_rqk_Q': rqk_mask_Q.float().sum(dim=-1).mean().item(),
                    'selected_rqk_K': rqk_mask_K.float().sum(dim=-1).mean().item(),
                    'selected_rv': rv_mask.float().sum(dim=-1).mean().item(),
                    # tau_offset (the learned parameter, in std units)
                    'tau_offset_fq': tau_offset_fq.mean().item() if self.learnable_tau else 0.0,
                    'tau_offset_fk': tau_offset_fk.mean().item() if self.learnable_tau else 0.0,
                    'tau_offset_fv': tau_offset_fv.mean().item() if self.learnable_tau else 0.0,
                    'tau_offset_rq': tau_offset_rq.mean().item() if self.learnable_tau else 0.0,
                    'tau_offset_rk': tau_offset_rk.mean().item() if self.learnable_tau else 0.0,
                    'tau_offset_rv': tau_offset_rv.mean().item() if self.learnable_tau else 0.0,
                    # gate_strength (0~1, low = more pass-through)
                    'gstr_fq': torch.tanh(fqk_gate_Q.max(dim=-1).values).mean().item(),
                    'gstr_fk': torch.tanh(fqk_gate_K.max(dim=-1).values).mean().item(),
                    'gstr_fv': torch.tanh(fv_gate.max(dim=-1).values).mean().item(),
                    'gstr_rq': torch.tanh(rqk_gate_Q.max(dim=-1).values).mean().item(),
                    'gstr_rk': torch.tanh(rqk_gate_K.max(dim=-1).values).mean().item(),
                    'gstr_rv': torch.tanh(rv_gate.max(dim=-1).values).mean().item(),
                    # Q/K overlap
                    'overlap_fqk': ((fqk_mask_Q & fqk_mask_K).float().sum(dim=-1) /
                                   torch.maximum(fqk_mask_Q.float().sum(dim=-1),
                                                fqk_mask_K.float().sum(dim=-1)).clamp(min=1)).mean().item(),
                    'overlap_rqk': ((rqk_mask_Q & rqk_mask_K).float().sum(dim=-1) /
                                   torch.maximum(rqk_mask_Q.float().sum(dim=-1),
                                                rqk_mask_K.float().sum(dim=-1)).clamp(min=1)).mean().item(),
                }
        else:
            routing_info = {}

        # Include preference/weight tensors for analysis (only when explicitly enabled)
        # NOTE: This is SEPARATE from debug_mode and self.training
        # - debug_mode: scalar stats for logging (100-step)
        # - store_pref_tensors: full tensors for detailed analysis (analyze_dawn.py)
        # - NEVER enable during training or validation to avoid memory leaks
        if self.store_pref_tensors:
            routing_info['fqk_q_pref'] = fqk_logits_Q
            routing_info['fqk_k_pref'] = fqk_logits_K
            routing_info['fv_pref'] = fv_logits
            routing_info['rqk_q_pref'] = rqk_logits_Q
            routing_info['rqk_k_pref'] = rqk_logits_K
            routing_info['rv_pref'] = rv_logits
            routing_info['fqk_weights_Q'] = fqk_weights_Q
            routing_info['fqk_weights_K'] = fqk_weights_K
            routing_info['fv_weights'] = fv_weights
            routing_info['rqk_weights_Q'] = rqk_weights_Q
            routing_info['rqk_weights_K'] = rqk_weights_K
            routing_info['rv_weights'] = rv_weights

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
        v18.4: Relative tau routing - tau calculated from full scores

        tau = score_mean + tau_offset * score_std (relative to full neuron pool)

        Args:
            tau_all: [B, S, 8] pre-computed tau_offset values (optional)
                     Order: [fq, fk, fv, rq, rk, rv, feature_know, restore_know]
        """
        logits_f, logits_r = self.neuron_router.get_knowledge_logits(x)

        # Get tau_offset and compute tau from full scores
        if self.learnable_tau:
            if tau_all is None:
                tau_all = self.tau_proj(x)
            tau_offset_f = tau_all[..., 6:7]
            tau_offset_r = tau_all[..., 7:8]

            # Compute tau from full scores: tau = mean + offset * std
            tau_f = logits_f.mean(dim=-1, keepdim=True) + tau_offset_f * (logits_f.std(dim=-1, keepdim=True) + 1e-8)
            tau_r = logits_r.mean(dim=-1, keepdim=True) + tau_offset_r * (logits_r.std(dim=-1, keepdim=True) + 1e-8)
        else:
            tau_offset_f = tau_offset_r = 0.0
            tau_f = logits_f.mean(dim=-1, keepdim=True)
            tau_r = logits_r.mean(dim=-1, keepdim=True)

        # Top-k selection with tau threshold
        f_paths, f_weights, f_mask, f_gate, f_conf = self._topk_select_and_chunk(
            logits_f, tau_f, self.path_max_k, self.max_paths)
        r_paths, r_weights, r_mask, r_gate, r_conf = self._topk_select_and_chunk(
            logits_r, tau_r, self.path_max_k, self.max_paths)

        # Update usage with mask (binary selection)
        if self.training:
            self.neuron_router.update_usage(f_mask.float(), 'feature_know', attention_mask)
            self.neuron_router.update_usage(r_mask.float(), 'restore_know', attention_mask)

        # Compute aux_loss for load balancing (score-based: softmax before top-k)
        aux_loss = 0.0
        if self.training:
            f_pref = F.softmax(logits_f, dim=-1)
            r_pref = F.softmax(logits_r, dim=-1)

            if attention_mask is not None:
                seq_mask = attention_mask.unsqueeze(-1).float()
                count = seq_mask.sum() + 1e-8
                usage_f = (f_pref * seq_mask).sum(dim=(0, 1)) / count
                usage_r = (r_pref * seq_mask).sum(dim=(0, 1)) / count
            else:
                usage_f = f_pref.mean(dim=(0, 1))
                usage_r = r_pref.mean(dim=(0, 1))

            target_f = 1.0 / self.n_feature_know
            target_r = 1.0 / self.n_restore_know

            aux_loss += ((usage_f - target_f) ** 2).sum() * self.n_feature_know
            aux_loss += ((usage_r - target_r) ** 2).sum() * self.n_restore_know

        # Compute tau regularization loss (penalize positive tau_offset)
        if self.training and self.learnable_tau and self.tau_reg_weight > 0:
            tau_reg = F.relu(tau_offset_f).mean()
            tau_reg += F.relu(tau_offset_r).mean()
            aux_loss += tau_reg * self.tau_reg_weight

        # Routing stats only in debug mode (avoid GPU sync overhead)
        if self.debug_mode:
            with torch.no_grad():
                know_info = {
                    'top_k': self.path_max_k * self.max_paths,
                    # Selected counts (what % passed tau)
                    'selected_feature': f_mask.float().sum(dim=-1).mean().item(),
                    'selected_restore': r_mask.float().sum(dim=-1).mean().item(),
                    # v18.4: tau_offset (learned parameter, in std units)
                    'tau_offset_feature': tau_offset_f.mean().item() if self.learnable_tau else 0.0,
                    'tau_offset_restore': tau_offset_r.mean().item() if self.learnable_tau else 0.0,
                    # v18.4: gate_strength from gate (0~1, low = pass-through)
                    'gstr_feature': torch.tanh((torch.exp(f_gate) - 1).max(dim=-1).values).mean().item(),
                    'gstr_restore': torch.tanh((torch.exp(r_gate) - 1).max(dim=-1).values).mean().item(),
                }
        else:
            know_info = {}

        return f_paths, r_paths, know_info, aux_loss


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

        # Q norm for dead routing detection
        q_norm = Q_total.norm(dim=-1, keepdim=True)  # [B, S, 1]

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

        # Q≈0 토큰은 attn 출력 억제
        scale = torch.where(
            q_norm > 1e-6,
            torch.ones_like(q_norm),
            q_norm * 1e-6
        )
        attn_out = attn_out * scale

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
        feature_paths, restore_paths, know_info, know_aux_loss = global_routers.get_knowledge_weights(
            normed_x_know, importance, attention_mask, tau_all=tau_all_know
        )
        know_out = self.knowledge(normed_x_know, feature_paths, restore_paths, attention_mask)
        x = x + know_out

        # Routing info
        routing_info = {
            'attention': attn_info,
            'knowledge': know_info,
        }
        # Store path_weights for analysis (when enabled)
        if getattr(global_routers, 'store_path_weights', False):
            routing_info['path_weights'] = {
                'fv': path_weights['fv'],
                'rv': path_weights['rv'],
                'fqk_Q': path_weights['fqk_Q'],
                'fqk_K': path_weights['fqk_K'],
                'rqk_Q': path_weights['rqk_Q'],
                'rqk_K': path_weights['rqk_K'],
                'feature_know': feature_paths,
                'restore_know': restore_paths,
            }
        # Norms only in debug mode (avoid GPU sync overhead)
        if global_routers.debug_mode:
            routing_info['attn_out_norm'] = attn_out.norm(dim=-1).mean().item()
            routing_info['know_out_norm'] = know_out.norm(dim=-1).mean().item()

        return x, routing_info, attn_aux_loss + know_aux_loss


class DAWN(nn.Module):
    """
    DAWN v18.4: Relative Confidence Scaling

    Key Features:
    - Based on v18.3 with relative confidence scaling
    - Gate: gate = ReLU(scores - tau), can be 0
    - Confidence: confidence = gate / gate_sum (relative, sums to 1)
    - Scaled weights: weights * confidence
    - tau decoupled from lb_loss, healthy gradient flow

    Benefits over v18.3:
    - tau stays in reasonable range (-3 ~ +3 instead of -170)
    - conf distribution is differentiated (not all 0.99)
    - lb_loss doesn't push tau to extremes

    Architecture:
    - UnifiedNeuronRouter: same as v18.0
    - GlobalRouters: Relative confidence gating + learnable tau (8 pools)
    - AttentionCircuit: Multi-path Q,K,V summation (same as v18.0)
    - KnowledgeCircuit: Multi-path summation (same as v18.0)
    """
    __version__ = "18.4"

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
        path_max_k: int = 16,
        learnable_tau: bool = True,
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
        tau_reg_weight: float = 0.0,
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
        self.path_max_k = path_max_k
        self.learnable_tau = learnable_tau
        self.tau_reg_weight = tau_reg_weight

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
            path_max_k=path_max_k,
            d_space=d_space, router_dropout=router_dropout,
            attention_token_routing=attention_token_routing,
            knowledge_token_routing=knowledge_token_routing,
            learnable_tau=learnable_tau,
            tau_reg_weight=tau_reg_weight,
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
                # Skip tau_proj (has its own initialization)
                if hasattr(self, 'router') and hasattr(self.router, 'tau_proj'):
                    if module is self.router.tau_proj:
                        continue
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None, return_routing_info=False, return_path_weights=False):
        B, S = input_ids.shape
        device = input_ids.device

        # Enable path_weights storage for analysis
        if return_path_weights:
            self.router.store_path_weights = True

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

        # Reset path_weights flag after forward
        if return_path_weights:
            self.router.store_path_weights = False

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().long()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)
            if return_routing_info or return_path_weights:
                return loss, logits, routing_infos
            return loss, logits

        if return_routing_info or return_path_weights:
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
            'path_max_k': self.path_max_k,
            'learnable_tau': self.learnable_tau,
            'tau_reg_weight': self.tau_reg_weight,
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'n_restore_qk': self.n_restore_qk, 'n_restore_v': self.n_restore_v,
            'n_feature_know': self.n_feature_know, 'n_restore_know': self.n_restore_know,
            'state_dim': self.state_dim, 'd_space': self.d_space,
            'knowledge_token_routing': self.knowledge_token_routing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: ReLU-Masked Learnable Tau Routing",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  max_paths={self.max_paths}, fixed_tau={self.fixed_tau}, path_max_k={self.path_max_k}",
            f"  max_seq_len={self.max_seq_len}, state_dim={self.state_dim}, dropout={self.dropout_rate}",
            f"",
            f"  [Attention - Q/K Separate Tau] (ReLU mask + learnable tau={self.learnable_tau})",
            f"  Feature_QK: {self.n_feature_qk} × {self.d_model} × {self.rank}",
            f"  Feature_V: {self.n_feature_v} × {self.d_model} × {self.rank}",
            f"  Restore_QK: {self.n_restore_qk} × {self.rank} × {self.d_model}",
            f"  Restore_V: {self.n_restore_v} × {self.rank} × {self.d_model}",
            f"",
            f"  [Knowledge - Feature-Restore] (ReLU mask + learnable tau={self.learnable_tau})",
            f"  Feature_Know: {self.n_feature_know} × {self.d_model} × {self.knowledge_rank}",
            f"  Restore_Know: {self.n_restore_know} × {self.knowledge_rank} × {self.d_model}",
            f"",
            f"  [Router - ReLU Mask + Learnable Tau (8 pools)]",
            f"  d_space={self.d_space}, router_dropout={self.router_dropout}",
            f"  attention_token_routing={self.attention_token_routing}, knowledge_token_routing={self.knowledge_token_routing}",
            f"  use_ssm_context={self.use_ssm_context}, tau_reg_weight={self.tau_reg_weight}",
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
            f"DAWN v{self.__version__}: ReLU-Masked Learnable Tau\n"
            f"  Params: {params:.1f}M\n"
            f"  rank={self.rank}, max_paths={self.max_paths}, path_max_k={self.path_max_k}\n"
            f"  Attention: Feature_QK={self.n_feature_qk}, Feature_V={self.n_feature_v}\n"
            f"            Restore_QK={self.n_restore_qk}, Restore_V={self.n_restore_v}\n"
            f"  Knowledge: Feature={self.n_feature_know}, Restore={self.n_restore_know}\n"
            f"  Total neurons: {attn_neurons} (attn) + {know_neurons} (know) = {attn_neurons + know_neurons}"
        )
