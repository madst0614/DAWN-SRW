"""
DAWN v17.2: Feature QK Unified Routing

Key change from v17.1:
- Feature stage: Q/K share single routing (h_qk)
- Restore stage: Q/K have separate routing

Motivation:
- v17.1 showed Feature_K entropy collapse to 0 (single neuron dominance)
- Feature_Q stayed healthy at ~45
- Unified Feature routing eliminates K collapse issue

Architecture:
x → Feature_QK routing → h_qk → Restore_Q routing → Q
                              → Restore_K routing → K
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
    v17.2: 5 attention projections + 2 knowledge projections
    Feature QK unified, Restore Q/K separate
    """
    def __init__(self, d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
                 n_feature_know, n_restore_know,
                 d_space=64, dropout=0.1, **kwargs):
        super().__init__()
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.d_space = d_space
        self.ema_alpha = kwargs.get('excitability_ema_alpha', 0.01)

        # 5 attention + 2 knowledge pools (Feature QK unified)
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

        # 5 attention projections + 2 knowledge projections (Feature QK unified)
        self.proj_all = nn.Linear(d_model, d_space * 5)  # fqk, fv, rqk_Q, rqk_K, rv
        self.proj_feature_know = nn.Linear(d_model, d_space)
        self.proj_restore_know = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        # Unified neuron embeddings (std=0.02 is standard transformer initialization)
        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        # Usage tracking (for logging)
        self.register_buffer('usage_ema_feature_qk', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_v', torch.zeros(n_feature_v))
        self.register_buffer('usage_ema_restore_q', torch.zeros(n_restore_qk))
        self.register_buffer('usage_ema_restore_k', torch.zeros(n_restore_qk))
        self.register_buffer('usage_ema_restore_v', torch.zeros(n_restore_v))
        self.register_buffer('usage_ema_feature_know', torch.zeros(n_feature_know))
        self.register_buffer('usage_ema_restore_know', torch.zeros(n_restore_know))

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
        """5 attention logits (Feature QK unified, Restore Q/K separate)"""
        emb_norm = F.normalize(self.neuron_emb, dim=-1)

        all_proj = self.dropout(self.proj_all(x))
        h_fqk, h_fv, h_rqk_Q, h_rqk_K, h_rv = all_proj.chunk(5, dim=-1)

        fqk_emb = emb_norm[:self.feature_qk_end]
        fv_emb = emb_norm[self.feature_qk_end:self.feature_v_end]
        rqk_emb = emb_norm[self.feature_v_end:self.restore_qk_end]
        rv_emb = emb_norm[self.restore_qk_end:self.restore_v_end]

        logits_fqk = torch.einsum('bsd,nd->bsn', h_fqk, fqk_emb)
        logits_fv = torch.einsum('bsd,nd->bsn', h_fv, fv_emb)
        logits_rqk_Q = torch.einsum('bsd,nd->bsn', h_rqk_Q, rqk_emb)
        logits_rqk_K = torch.einsum('bsd,nd->bsn', h_rqk_K, rqk_emb)
        logits_rv = torch.einsum('bsd,nd->bsn', h_rv, rv_emb)

        return logits_fqk, logits_fv, logits_rqk_Q, logits_rqk_K, logits_rv

    def update_usage(self, weights, neuron_type, attention_mask=None):
        if not self.training:
            return

        if weights.dim() == 3:
            active = (weights > 0).float()
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                active = active * mask
                count = mask.sum() + 1e-8
                usage = active.sum(dim=[0, 1]) / count
            else:
                usage = active.mean(dim=[0, 1])
        else:
            usage = (weights > 0).float().mean(dim=0)

        decay = 1 - self.ema_alpha
        if neuron_type == 'feature_qk':
            self.usage_ema_feature_qk.mul_(decay).add_(usage, alpha=self.ema_alpha)
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
    v17.1: Both Attention + Knowledge use Feature-Restore pattern
    Knowledge uses Feature/Restore separate routing
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
    """Selective SSM (same as v16.4)"""
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

        h_final = ssm_out[:, -1, :]
        h_proj = self.importance_proj(h_final)
        raw_importance = torch.einsum('bsd,bd->bs', x, h_proj)

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
    """v17.2 routing with Feature QK unified + Knowledge Feature/Restore separation"""
    def __init__(self, d_model: int, n_feature_qk: int, n_feature_v: int,
                 n_restore_qk: int, n_restore_v: int,
                 n_feature_know: int, n_restore_know: int,
                 top_k_feature_qk: int = 8, top_k_feature_v: int = 8,
                 top_k_restore_qk: int = 8, top_k_restore_v: int = 8,
                 top_k_feature_know: int = 4, top_k_restore_know: int = 4,
                 d_space: int = 64, router_dropout: float = 0.1, token_routing: bool = False,
                 knowledge_token_routing: bool = False, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v
        self.top_k_restore_qk = top_k_restore_qk
        self.top_k_restore_v = top_k_restore_v
        self.top_k_feature_know = top_k_feature_know
        self.top_k_restore_know = top_k_restore_know
        self.token_routing = token_routing
        self.knowledge_token_routing = knowledge_token_routing

        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
            n_feature_know, n_restore_know,
            d_space=d_space, dropout=router_dropout, **kwargs
        )

    def _topk_sparsify(self, weights, k):
        topk_vals, topk_idx = torch.topk(weights, k, dim=-1)
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, topk_idx, topk_vals)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse_weights, topk_idx

    def get_attention_weights(self, x, importance, attention_mask=None):
        # v17.2: Feature QK unified, Restore Q/K separate
        (fqk_logits, fv_logits,
         rqk_logits_Q, rqk_logits_K, rv_logits) = self.neuron_router.get_all_logits(x)

        fqk_pref = F.softmax(fqk_logits, dim=-1)
        fv_pref = F.softmax(fv_logits, dim=-1)
        rqk_pref_Q = F.softmax(rqk_logits_Q, dim=-1)
        rqk_pref_K = F.softmax(rqk_logits_K, dim=-1)
        rv_pref = F.softmax(rv_logits, dim=-1)

        aux_loss = 0.0
        if self.training:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                count = mask.sum() + 1e-8
                usage_fqk = (fqk_pref * mask).sum(dim=(0, 1)) / count
                usage_fv = (fv_pref * mask).sum(dim=(0, 1)) / count
                usage_rqk_Q = (rqk_pref_Q * mask).sum(dim=(0, 1)) / count
                usage_rqk_K = (rqk_pref_K * mask).sum(dim=(0, 1)) / count
                usage_rv = (rv_pref * mask).sum(dim=(0, 1)) / count
            else:
                usage_fqk = fqk_pref.mean(dim=(0, 1))
                usage_fv = fv_pref.mean(dim=(0, 1))
                usage_rqk_Q = rqk_pref_Q.mean(dim=(0, 1))
                usage_rqk_K = rqk_pref_K.mean(dim=(0, 1))
                usage_rv = rv_pref.mean(dim=(0, 1))

            target_fqk = 1.0 / self.n_feature_qk
            target_fv = 1.0 / self.n_feature_v
            target_rqk = 1.0 / self.n_restore_qk
            target_rv = 1.0 / self.n_restore_v

            aux_loss += ((usage_fqk - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss += ((usage_fv - target_fv) ** 2).sum() * self.n_feature_v
            aux_loss += ((usage_rqk_Q - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss += ((usage_rqk_K - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss += ((usage_rv - target_rv) ** 2).sum() * self.n_restore_v

        if self.token_routing:
            fqk_weights = fqk_pref
            fv_weights = fv_pref
            rqk_weights_Q = rqk_pref_Q
            rqk_weights_K = rqk_pref_K
            rv_weights = rv_pref

            routing_info = {
                'fqk_weights': fqk_weights.detach(),
                'fv_weights': fv_weights.detach(),
                'rqk_weights_Q': rqk_weights_Q.detach(),
                'rqk_weights_K': rqk_weights_K.detach(),
                'rv_weights': rv_weights.detach(),
                'fqk_pref': fqk_pref.detach(),
                'fv_pref': fv_pref.detach(),
                'rqk_q_pref': rqk_pref_Q.detach(),
                'rqk_k_pref': rqk_pref_K.detach(),
                'rv_pref': rv_pref.detach(),
                'token_routing': True,
            }
        else:
            # Batch-level routing
            fqk_weights_dense = torch.einsum('bs,bsn->bn', importance, fqk_pref)
            fv_weights_dense = torch.einsum('bs,bsn->bn', importance, fv_pref)
            rqk_weights_Q_dense = torch.einsum('bs,bsn->bn', importance, rqk_pref_Q)
            rqk_weights_K_dense = torch.einsum('bs,bsn->bn', importance, rqk_pref_K)
            rv_weights_dense = torch.einsum('bs,bsn->bn', importance, rv_pref)

            fqk_weights, _ = self._topk_sparsify(fqk_weights_dense, self.top_k_feature_qk)
            fv_weights, _ = self._topk_sparsify(fv_weights_dense, self.top_k_feature_v)
            rqk_weights_Q, _ = self._topk_sparsify(rqk_weights_Q_dense, self.top_k_restore_qk)
            rqk_weights_K, _ = self._topk_sparsify(rqk_weights_K_dense, self.top_k_restore_qk)
            rv_weights, _ = self._topk_sparsify(rv_weights_dense, self.top_k_restore_v)

            routing_info = {
                'fqk_weights': fqk_weights.detach(),
                'fv_weights': fv_weights.detach(),
                'rqk_weights_Q': rqk_weights_Q.detach(),
                'rqk_weights_K': rqk_weights_K.detach(),
                'rv_weights': rv_weights.detach(),
                'fqk_pref': fqk_pref.detach(),
                'fv_pref': fv_pref.detach(),
                'rqk_q_pref': rqk_pref_Q.detach(),
                'rqk_k_pref': rqk_pref_K.detach(),
                'rv_pref': rv_pref.detach(),
                'token_routing': False,
            }

        # Update usage (Feature QK unified, Restore Q/K separate)
        if self.training:
            self.neuron_router.update_usage(fqk_weights, 'feature_qk', attention_mask)
            self.neuron_router.update_usage(fv_weights, 'feature_v', attention_mask)
            self.neuron_router.update_usage(rqk_weights_Q, 'restore_q', attention_mask)
            self.neuron_router.update_usage(rqk_weights_K, 'restore_k', attention_mask)
            self.neuron_router.update_usage(rv_weights, 'restore_v', attention_mask)

        return fqk_weights, fv_weights, rqk_weights_Q, rqk_weights_K, rv_weights, routing_info, aux_loss

    def get_knowledge_weights(self, x, importance, attention_mask=None):
        """Knowledge neuron routing - Feature/Restore separation"""
        logits_f, logits_r = self.neuron_router.get_knowledge_logits(x)

        pref_f = F.softmax(logits_f, dim=-1)  # [B, S, n]
        pref_r = F.softmax(logits_r, dim=-1)  # [B, S, n]

        if self.knowledge_token_routing:
            # Token-level: top-k per token
            feature_know_w, _ = self._topk_sparsify(pref_f, self.top_k_feature_know)
            restore_know_w, _ = self._topk_sparsify(pref_r, self.top_k_restore_know)
        else:
            # Batch-level (default)
            f_dense = torch.einsum('bs,bsn->bn', importance, pref_f)  # [B, n]
            r_dense = torch.einsum('bs,bsn->bn', importance, pref_r)  # [B, n]
            feature_know_w, _ = self._topk_sparsify(f_dense, self.top_k_feature_know)
            restore_know_w, _ = self._topk_sparsify(r_dense, self.top_k_restore_know)

        if self.training:
            self.neuron_router.update_usage(feature_know_w, 'feature_know', attention_mask)
            self.neuron_router.update_usage(restore_know_w, 'restore_know', attention_mask)

        return feature_know_w, restore_know_w


class AttentionCircuit(nn.Module):
    """v17.2 attention circuit: Feature QK unified, Restore Q/K separate"""
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

    def forward(self, x, fqk_weights, fv_weights, rqk_weights_Q, rqk_weights_K, rv_weights, attention_mask=None):
        """
        v17.2: Feature QK unified routing, Restore Q/K separate
        x -> fqk_weights -> h_qk -> rqk_weights_Q -> Q
                                 -> rqk_weights_K -> K
        """
        B, S, D = x.shape
        R = self.rank
        token_routing = fqk_weights.dim() == 3

        if token_routing:
            # Token-level routing
            # Feature: unified QK
            shared_fqk = torch.einsum('bsn,ndr->bsdr', fqk_weights, self.shared_neurons.feature_qk_neurons)
            shared_fv = torch.einsum('bsn,ndr->bsdr', fv_weights, self.shared_neurons.feature_v_neurons)

            h_qk = torch.einsum('bsd,bsdr->bsr', x, shared_fqk)  # Unified for Q and K
            h_v = torch.einsum('bsd,bsdr->bsr', x, shared_fv)

            # Restore: separate Q/K
            shared_rqk_Q = torch.einsum('bsn,nrd->bsrd', rqk_weights_Q, self.shared_neurons.restore_qk_neurons)
            shared_rqk_K = torch.einsum('bsn,nrd->bsrd', rqk_weights_K, self.shared_neurons.restore_qk_neurons)
            shared_rv = torch.einsum('bsn,nrd->bsrd', rv_weights, self.shared_neurons.restore_v_neurons)

            Q = torch.einsum('bsr,bsrd->bsd', h_qk, shared_rqk_Q)
            K = torch.einsum('bsr,bsrd->bsd', h_qk, shared_rqk_K)
            V = torch.einsum('bsr,bsrd->bsd', h_v, shared_rv)
        else:
            # Batch-level routing - matmul optimized
            # Feature: unified QK
            fqk_flat = self.shared_neurons.feature_qk_neurons.view(-1, D * R)
            shared_fqk = (fqk_weights @ fqk_flat).view(B, D, R)

            fv_flat = self.shared_neurons.feature_v_neurons.view(-1, D * R)
            shared_fv = (fv_weights @ fv_flat).view(B, D, R)

            h_qk = torch.bmm(x, shared_fqk)  # Unified for Q and K
            h_v = torch.bmm(x, shared_fv)

            # Restore: separate Q/K
            rqk_flat = self.shared_neurons.restore_qk_neurons.view(-1, R * D)
            shared_rqk_Q = (rqk_weights_Q @ rqk_flat).view(B, R, D)
            shared_rqk_K = (rqk_weights_K @ rqk_flat).view(B, R, D)

            rv_flat = self.shared_neurons.restore_v_neurons.view(-1, R * D)
            shared_rv = (rv_weights @ rv_flat).view(B, R, D)

            Q = torch.bmm(h_qk, shared_rqk_Q)
            K = torch.bmm(h_qk, shared_rqk_K)
            V = torch.bmm(h_v, shared_rv)

        # Multi-head attention
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Causal mask (language modeling: cannot see future tokens)
        causal_mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask[None, None, :, :], float('-inf'))

        # Padding mask (optional)
        if attention_mask is not None:
            mask = attention_mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        output = self.expand_O(attn_out)
        output = self.out_dropout(output)

        # Returns (output, None) - second value is unused (kept for API compatibility)
        return output, None


class KnowledgeCircuit(nn.Module):
    """
    v17.1 Knowledge Circuit: Feature-Restore separation (like v17)

    Routing: batch-level (importance weighted average) - same as Attention

    Flow:
    x -> feature_know_w [B,n] selects feature_know neurons -> h (compression)
      -> restore_know_w [B,n] selects restore_know neurons -> output (restoration)
    """
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_feature_know: int,
        n_restore_know: int,
        knowledge_rank: int,
        top_k_feature_know: int = 4,
        top_k_restore_know: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.knowledge_rank = knowledge_rank
        self.top_k_feature_know = top_k_feature_know
        self.top_k_restore_know = top_k_restore_know
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, feature_know_w, restore_know_w, attention_mask=None):
        """
        x: [B, S, D]
        feature_know_w: [B, n] (batch-level) or [B, S, n] (token-level)
        restore_know_w: [B, n] (batch-level) or [B, S, n] (token-level)
        """
        B, S, D = x.shape
        R = self.knowledge_rank
        token_routing = feature_know_w.dim() == 3

        if token_routing:
            # Token-level: [B, S, n] weights
            shared_f = torch.einsum('bsn,ndr->bsdr', feature_know_w,
                                    self.shared_neurons.feature_know)  # [B, S, D, R]
            h = torch.einsum('bsd,bsdr->bsr', x, shared_f)  # [B, S, R]

            shared_r = torch.einsum('bsn,nrd->bsrd', restore_know_w,
                                    self.shared_neurons.restore_know)  # [B, S, R, D]
            output = torch.einsum('bsr,bsrd->bsd', h, shared_r)  # [B, S, D]
        else:
            # Batch-level: [B, n] weights
            f_flat = self.shared_neurons.feature_know.view(-1, D * R)
            shared_f = (feature_know_w @ f_flat).view(B, D, R)
            h = torch.bmm(x, shared_f)  # [B, S, R]

            r_flat = self.shared_neurons.restore_know.view(-1, R * D)
            shared_r = (restore_know_w @ r_flat).view(B, R, D)
            output = torch.bmm(h, shared_r)  # [B, S, D]

        return self.dropout(output)


class DAWNBlock(nn.Module):
    """DAWN v17.2 block: Feature QK unified + Knowledge Feature-Restore separation"""
    def __init__(
        self,
        shared_neurons: SharedNeurons,
        d_model: int,
        n_heads: int,
        rank: int,
        n_feature_know: int,
        n_restore_know: int,
        knowledge_rank: int = 128,
        top_k_feature_know: int = 4,
        top_k_restore_know: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn = AttentionCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.knowledge = KnowledgeCircuit(
            shared_neurons, d_model, n_feature_know, n_restore_know,
            knowledge_rank=knowledge_rank,
            top_k_feature_know=top_k_feature_know,
            top_k_restore_know=top_k_restore_know,
            dropout=dropout
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, importance, global_routers: GlobalRouters, attention_mask=None):
        # Attention (v17.2: Feature QK unified, Restore Q/K separate)
        normed_x = self.norm1(x)
        fqk_w, fv_w, rqk_w_Q, rqk_w_K, rv_w, attn_routing, attn_aux_loss = \
            global_routers.get_attention_weights(normed_x, importance, attention_mask)

        attn_out, _ = self.attn(normed_x, fqk_w, fv_w, rqk_w_Q, rqk_w_K, rv_w, attention_mask)
        x = x + attn_out

        # Knowledge
        normed_x = self.norm2(x)
        feature_know_w, restore_know_w = global_routers.get_knowledge_weights(normed_x, importance, attention_mask)
        know_out = self.knowledge(normed_x, feature_know_w, restore_know_w, attention_mask)
        x = x + know_out

        # Routing info
        attn_out_norm = attn_out.norm(dim=-1).mean().detach()
        know_out_norm = know_out.norm(dim=-1).mean().detach()

        routing_info = {
            'attention': attn_routing,
            'knowledge': {
                'feature_know_w': feature_know_w.detach(),
                'restore_know_w': restore_know_w.detach(),
            },
            'attn_out_norm': attn_out_norm,
            'know_out_norm': know_out_norm,
        }

        return x, routing_info, attn_aux_loss


class DAWN(nn.Module):
    """
    DAWN v17.2: Feature QK Unified + Restore Q/K Separate

    Architecture:
    - Feature stage: QK unified routing (fixes K entropy collapse)
    - Restore stage: Q/K separate routing
    - Knowledge: Feature/Restore separate routing
    """
    __version__ = "17.2"

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 320,
        n_layers: int = 4,
        n_heads: int = 8,
        rank: int = 64,
        max_seq_len: int = 512,
        state_dim: int = 64,
        d_space: int = 64,
        # Attention - shared Q/K pool
        n_feature_qk: int = 56,
        n_feature_v: int = 24,
        top_k_feature_qk: int = 16,
        top_k_feature_v: int = 6,
        n_restore_qk: int = 56,
        n_restore_v: int = 24,
        top_k_restore_qk: int = 16,
        top_k_restore_v: int = 6,
        # Knowledge - Feature/Restore separation
        n_feature_know: int = 24,
        n_restore_know: int = 24,
        top_k_feature_know: int = 4,
        top_k_restore_know: int = 4,
        knowledge_rank: int = 128,
        # Others
        dropout: float = 0.1,
        token_routing: bool = False,
        knowledge_token_routing: bool = False,
        router_dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        use_ssm_context: bool = True,
        **kwargs,  # Accept but ignore excitability params for backward compatibility
    ):
        super().__init__()

        # Validation checks
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        if top_k_feature_qk > n_feature_qk:
            raise ValueError(f"top_k_feature_qk ({top_k_feature_qk}) cannot exceed n_feature_qk ({n_feature_qk})")
        if top_k_feature_v > n_feature_v:
            raise ValueError(f"top_k_feature_v ({top_k_feature_v}) cannot exceed n_feature_v ({n_feature_v})")
        if top_k_restore_qk > n_restore_qk:
            raise ValueError(f"top_k_restore_qk ({top_k_restore_qk}) cannot exceed n_restore_qk ({n_restore_qk})")
        if top_k_restore_v > n_restore_v:
            raise ValueError(f"top_k_restore_v ({top_k_restore_v}) cannot exceed n_restore_v ({n_restore_v})")
        if top_k_feature_know > n_feature_know:
            raise ValueError(f"top_k_feature_know ({top_k_feature_know}) cannot exceed n_feature_know ({n_feature_know})")
        if top_k_restore_know > n_restore_know:
            raise ValueError(f"top_k_restore_know ({top_k_restore_know}) cannot exceed n_restore_know ({n_restore_know})")

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
        self.token_routing = token_routing
        self.knowledge_token_routing = knowledge_token_routing
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_ssm_context = use_ssm_context

        # Shared Q/K pool
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.top_k_restore_qk = top_k_restore_qk
        self.top_k_restore_v = top_k_restore_v

        # Knowledge Feature/Restore separation
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.top_k_feature_know = top_k_feature_know
        self.top_k_restore_know = top_k_restore_know

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
            top_k_feature_qk=top_k_feature_qk, top_k_feature_v=top_k_feature_v,
            top_k_restore_qk=top_k_restore_qk, top_k_restore_v=top_k_restore_v,
            top_k_feature_know=top_k_feature_know, top_k_restore_know=top_k_restore_know,
            d_space=d_space, router_dropout=router_dropout, token_routing=token_routing,
            knowledge_token_routing=knowledge_token_routing,
        )

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, n_feature_know=n_feature_know, n_restore_know=n_restore_know,
                knowledge_rank=knowledge_rank,
                top_k_feature_know=top_k_feature_know, top_k_restore_know=top_k_restore_know,
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

        importance, context, raw_importance = self.global_ssm(x, attention_mask)
        if context is not None:
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

        # Note: decay_excitability() is called in train.py, not here

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
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'top_k_feature_qk': self.top_k_feature_qk, 'top_k_feature_v': self.top_k_feature_v,
            'n_restore_qk': self.n_restore_qk, 'n_restore_v': self.n_restore_v,
            'top_k_restore_qk': self.top_k_restore_qk, 'top_k_restore_v': self.top_k_restore_v,
            'n_feature_know': self.n_feature_know, 'n_restore_know': self.n_restore_know,
            'top_k_feature_know': self.top_k_feature_know, 'top_k_restore_know': self.top_k_restore_know,
            'state_dim': self.state_dim, 'd_space': self.d_space,
            'knowledge_token_routing': self.knowledge_token_routing,
        }

    def get_model_info(self):
        """Return model architecture info for logging"""
        return [
            f"DAWN v{self.__version__}: Feature QK Unified + Restore Q/K Separate",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  max_seq_len={self.max_seq_len}, state_dim={self.state_dim}, dropout={self.dropout_rate}",
            f"",
            f"  [Attention - Feature QK Unified, Restore Q/K Separate]",
            f"  Feature_QK: {self.n_feature_qk} × {self.d_model} × {self.rank} (top-k={self.top_k_feature_qk})",
            f"  Feature_V: {self.n_feature_v} × {self.d_model} × {self.rank} (top-k={self.top_k_feature_v})",
            f"  Restore_QK: {self.n_restore_qk} × {self.rank} × {self.d_model} (top-k={self.top_k_restore_qk})",
            f"  Restore_V: {self.n_restore_v} × {self.rank} × {self.d_model} (top-k={self.top_k_restore_v})",
            f"",
            f"  [Knowledge - Feature-Restore]",
            f"  Feature_Know: {self.n_feature_know} × {self.d_model} × {self.knowledge_rank} (top-k={self.top_k_feature_know})",
            f"  Restore_Know: {self.n_restore_know} × {self.knowledge_rank} × {self.d_model} (top-k={self.top_k_restore_know})",
            f"",
            f"  [Router]",
            f"  d_space={self.d_space}, router_dropout={self.router_dropout}",
            f"  token_routing={self.token_routing}, knowledge_token_routing={self.knowledge_token_routing}",
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
            f"DAWN v{self.__version__}: Feature QK Unified + Restore Q/K Separate\n"
            f"  Params: {params:.1f}M\n"
            f"  Attention: Feature_QK={self.n_feature_qk} (k={self.top_k_feature_qk}), "
            f"Feature_V={self.n_feature_v} (k={self.top_k_feature_v})\n"
            f"            Restore_QK={self.n_restore_qk} (k={self.top_k_restore_qk}), "
            f"Restore_V={self.n_restore_v} (k={self.top_k_restore_v})\n"
            f"  Knowledge: Feature={self.n_feature_know} (k={self.top_k_feature_know}), "
            f"Restore={self.n_restore_know} (k={self.top_k_restore_know})\n"
            f"  Total neurons: {attn_neurons} (attn) + {know_neurons} (know) = {attn_neurons + know_neurons}"
        )
