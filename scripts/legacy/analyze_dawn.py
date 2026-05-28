#!/usr/bin/env python3
"""
DAWN Model Analysis Suite
=========================
Comprehensive analysis toolkit for the DAWN (Dynamic Attention and Weight Network) model.
Designed for paper-ready analysis and visualization.

This script now serves as a CLI wrapper around the modular `scripts.analysis` package.
For programmatic usage, prefer importing directly from `scripts.analysis`:

    from scripts.analysis import (
        PaperFigureGenerator,
        NeuronHealthAnalyzer,
        RoutingAnalyzer,
        load_model,
        get_router,
    )

Analysis Categories:
1. Usage Analysis       - EMA distribution, diversity, dead neurons
2. Routing Analysis     - Entropy, selection frequency, Q/K overlap, selection diversity
3. Embedding Analysis   - Similarity, clustering, t-SNE/PCA visualization
4. Weight Analysis      - SVD decomposition, effective rank
5. Behavioral Analysis  - Token trajectory, probing classifier, ablation study
6. Semantic Analysis    - Path similarity, context-dependent routing, POS patterns
7. Co-selection         - Feature/Restore neuron pairing analysis
8. V18 Analysis         - Learnable tau, gate distribution, Q/K tau differentiation, confidence (v18.2+)

Usage:
    python analyze_dawn.py --checkpoint path/to/ckpt --mode all
    python analyze_dawn.py --checkpoint path/to/ckpt --mode usage
    python analyze_dawn.py --checkpoint path/to/ckpt --mode routing --val_data path/to/data
    python analyze_dawn.py --checkpoint path/to/ckpt --mode semantic --val_data path/to/data
    python analyze_dawn.py --checkpoint path/to/ckpt --mode paper --val_data path/to/data
    python analyze_dawn.py --checkpoint path/to/ckpt --mode v18 --val_data path/to/data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import argparse
from typing import Dict, Optional

# Import from new modular package
from scripts.analysis import (
    # Model loading
    load_model, get_router, get_neurons, create_dataloader,

    # Constants
    NEURON_TYPES, ROUTING_KEYS, NEURON_ATTRS,

    # Utilities
    gini_coefficient, calc_entropy, calc_entropy_ratio,
    convert_to_serializable, save_results,

    # Analyzers
    NeuronHealthAnalyzer, RoutingAnalyzer, EmbeddingAnalyzer,
    WeightAnalyzer, BehavioralAnalyzer, SemanticAnalyzer,
    CoselectionAnalyzer, PaperFigureGenerator,

    # Flags
    HAS_MATPLOTLIB, HAS_SKLEARN, HAS_TQDM,
)


# ============================================================
# Legacy Analyzer Classes (for backward compatibility)
# ============================================================

class UsageAnalyzer:
    """Legacy wrapper around NeuronHealthAnalyzer."""

    def __init__(self, router):
        self._health = NeuronHealthAnalyzer(router)
        self.router = router

    def analyze_ema_distribution(self) -> Dict:
        return self._health.analyze_ema_distribution()

    def analyze_diversity(self) -> Dict:
        return self._health.analyze_diversity()

    def analyze_dead_neurons(self, output_dir: str = None) -> Dict:
        return self._health.analyze_dead_neurons(output_dir)

    def visualize_usage(self, output_dir: str) -> Dict:
        return self._health.visualize_usage(output_dir)


# ============================================================
# Main Analyzer Class
# ============================================================

class DAWNAnalyzer:
    """DAWN model integrated analyzer.

    This class provides a unified interface to all analysis functionality.
    It now delegates to the modular analyzers in scripts.analysis.
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.router = get_router(model)
        self.neurons = get_neurons(model)

        if self.router is None:
            raise ValueError("Could not find router in model")

        # Initialize modular analyzers
        self.usage = UsageAnalyzer(self.router)  # Legacy wrapper
        self.health = NeuronHealthAnalyzer(self.router)
        self.embedding = EmbeddingAnalyzer(self.router)
        self.routing = RoutingAnalyzer(model, self.router, device)
        self.weight = WeightAnalyzer(self.neurons)
        self.behavioral = BehavioralAnalyzer(model, self.router, tokenizer, device)
        self.semantic = SemanticAnalyzer(model, self.router, tokenizer, device)
        self.coselection = CoselectionAnalyzer(model, self.router, self.neurons, device)

        print("DAWN Analyzer initialized (modular)")

    def run_usage_analysis(self, output_dir: str = None) -> Dict:
        """Usage analysis (delegates to NeuronHealthAnalyzer)."""
        print("\n" + "="*60)
        print("USAGE ANALYSIS")
        print("="*60)

        results = self.health.run_all(output_dir)

        # Print summary
        print("\n--- EMA Distribution ---")
        for name, data in results.get('ema_distribution', {}).items():
            if isinstance(data, dict) and 'display' in data:
                print(f"  {data['display']:>4}: {data['active']:>4}/{data['total']:<4} active "
                      f"({data['active_ratio']:.1%}), gini={data['gini']:.2f}")

        print("\n--- Diversity ---")
        div = results.get('diversity', {})
        for name, data in div.items():
            if name == 'overall':
                continue
            if isinstance(data, dict) and 'display' in data:
                print(f"  {data['display']:>4}: entropy={data.get('normalized_entropy', 0):.2f}, "
                      f"effective={data.get('effective_count', 0):.1f}")
        if 'overall' in div:
            print(f"\n  Overall health: {div['overall']['health']} "
                  f"(score={div['overall']['diversity_score']:.2f})")

        # v18.x Q/K EMA Overlap
        qk_overlap = results.get('qk_ema_overlap', {})
        if qk_overlap and 'error' not in qk_overlap:
            print("\n--- Q/K Neuron Specialization (v18.x) ---")
            for pool_name, data in qk_overlap.items():
                n = data['n_total']
                print(f"\n  {pool_name.upper()} Pool ({n} neurons):")
                print(f"    Q only:  {data['q_only']:>4} ({data['q_only']/n*100:5.1f}%)")
                print(f"    K only:  {data['k_only']:>4} ({data['k_only']/n*100:5.1f}%)")
                print(f"    Shared:  {data['shared']:>4} ({data['shared']/n*100:5.1f}%)")
                print(f"    Dead:    {data['dead']:>4} ({data['dead']/n*100:5.1f}%)")
                print(f"    Q/K corr: {data['correlation']:.3f}")

        return results

    def run_routing_analysis(self, dataloader, n_batches: int = 50, output_dir: str = None) -> Dict:
        """Routing analysis."""
        print("\n" + "="*60)
        print("ROUTING ANALYSIS")
        print("="*60)

        results = self.routing.run_all(dataloader, output_dir, n_batches)

        print("\n--- Routing Entropy ---")
        for key, data in results.get('entropy', {}).items():
            if isinstance(data, dict) and 'display' in data:
                print(f"  {data['display']:>6}: {data['mean_entropy']:.1f}% (std={data['std_entropy']:.1f})")

        print("\n--- Q/K Usage ---")
        for key, data in results.get('qk_usage', {}).items():
            if isinstance(data, dict) and 'display' in data:
                print(f"  {data['display']:>6}: correlation={data.get('correlation', 0):.3f}")

        return results

    def run_embedding_analysis(self, output_dir: str = None) -> Dict:
        """Embedding analysis."""
        print("\n" + "="*60)
        print("EMBEDDING ANALYSIS")
        print("="*60)

        results = self.embedding.run_all(output_dir)

        print("\n--- Intra-type Similarity ---")
        for name, data in results.get('similarity', {}).items():
            if name == 'visualization':
                continue
            if isinstance(data, dict) and 'display' in data:
                print(f"  {data['display']:>4}: avg={data['avg_similarity']:.3f}, "
                      f"max={data['max_similarity']:.3f}")

        return results

    def run_weight_analysis(self, output_dir: str = None) -> Dict:
        """Weight SVD analysis."""
        print("\n" + "="*60)
        print("WEIGHT SVD ANALYSIS")
        print("="*60)

        results = self.weight.run_all(output_dir)

        for name, data in results.get('svd', {}).items():
            if name == 'visualization' or 'error' in str(data):
                continue
            if isinstance(data, dict):
                print(f"\n{data.get('display', name)}:")
                print(f"  Effective rank: {data.get('effective_rank', 0):.2f}")
                print(f"  Var explained by top 5: {data.get('var_explained_by_top5', 0):.2%}")
                print(f"  Condition number: {data.get('condition_number', 0):.2f}")

        return results

    def run_clustering_analysis(self, n_clusters: int = 5, output_dir: str = None) -> Dict:
        """Clustering analysis."""
        print("\n" + "="*60)
        print("CLUSTERING ANALYSIS")
        print("="*60)

        results = self.embedding.analyze_clustering(n_clusters, output_dir)

        for name, data in results.items():
            if name == 'visualization' or 'error' in str(data):
                continue
            if isinstance(data, dict) and 'display' in data:
                print(f"\n{data['display']}: {data['n_clusters']} clusters")
                for c in data.get('clusters', [])[:3]:
                    print(f"  Cluster {c['cluster_id']}: size={c['size']}, "
                          f"active={c.get('active_count', 'N/A')}, usage={c.get('avg_usage', 0):.4f}")

        return results

    def run_semantic_analysis(self, dataloader=None, max_batches: int = 50, output_dir: str = None) -> Dict:
        """Semantic analysis (NEW)."""
        print("\n" + "="*60)
        print("SEMANTIC ANALYSIS")
        print("="*60)

        results = self.semantic.run_all(dataloader, output_dir, max_batches=max_batches)

        # Path similarity
        path_sim = results.get('path_similarity', {})
        if 'similar_pairs' in path_sim:
            print("\n--- Semantic Path Similarity ---")
            sim = path_sim['similar_pairs']
            diff = path_sim['different_pairs']
            print(f"  Similar pairs: cosine={sim.get('cosine_mean', 0):.3f}, "
                  f"jaccard={sim.get('jaccard_mean', 0):.3f}")
            print(f"  Different pairs: cosine={diff.get('cosine_mean', 0):.3f}, "
                  f"jaccard={diff.get('jaccard_mean', 0):.3f}")
            print(f"  Interpretation: {path_sim.get('interpretation', 'N/A')}")

        # Context routing
        context = results.get('context_routing', {})
        if 'summary' in context:
            print(f"\n--- Context-Dependent Routing ---")
            print(f"  Overall variance: {context['summary'].get('overall_context_variance', 0):.4f}")

        return results

    def run_coselection_analysis(self, dataloader, n_batches: int = 50, output_dir: str = None) -> Dict:
        """Co-selection analysis."""
        print("\n" + "="*60)
        print("CO-SELECTION ANALYSIS")
        print("="*60)

        results = self.coselection.run_all(dataloader, output_dir, n_batches=n_batches)

        for pair_key, data in results.items():
            if 'concentration' in data:
                conc = data['concentration']
                print(f"\n{data.get('pair_name', pair_key)}:")
                print(f"  Top 10 pairs: {conc.get('top10_pct', 0):.1f}% of co-selections")
                print(f"  Entropy: {conc.get('normalized_entropy', 0):.3f}")

        return results

    def run_v18_analysis(self, dataloader=None, n_batches: int = 50, output_dir: str = None) -> Dict:
        """v18.x-specific analysis: tau, gate, confidence, Q/K differentiation.

        Analyzes v18.x specific features:
        1. Learnable Tau - pool tau values (structure varies by version)
        2. Gate Distribution - log-gated threshold selection
        3. Token-level Tau - tau variance across tokens/positions
        4. Q/K Tau Differentiation - how Q and K taus diverge
        5. Confidence Stats (v18.3+) - confidence scaling metrics

        Version differences:
        - v18.0-18.3: tau_proj with 8 outputs [fq,fk,fv,rq,rk,rv,f_know,r_know]
        - v18.4-18.5: tau_proj_feature with 4 outputs [fq,fk,fv,f_know]
        - v18.5: restore tau is context-based (tau_proj_restore_Q/K/v/know)
        """
        # Detect model version
        model_version = getattr(self.model, '__version__', '18.2')
        print("\n" + "="*60)
        print(f"V18.x SPECIFIC ANALYSIS (v{model_version})")
        print("="*60)

        results = {'model_version': model_version}

        # Check if model has v18 features - support both old and new structure
        has_old_tau = hasattr(self.model, 'router') and hasattr(self.model.router, 'tau_proj')
        has_new_tau = hasattr(self.model, 'router') and hasattr(self.model.router, 'tau_proj_feature')

        if not has_old_tau and not has_new_tau:
            print("Warning: Model does not have learnable tau (not v18.x)")
            return {'error': 'Not a v18.x model with learnable tau'}

        import torch.nn.functional as F

        # 1. Tau Parameters Analysis
        print("\n--- Tau Parameters ---")

        if has_new_tau:
            # v18.4/v18.5 structure
            tau_proj_feature = self.model.router.tau_proj_feature
            feature_names = ['fq', 'fk', 'fv', 'feature_know']

            weight_f = tau_proj_feature.weight.detach().cpu()  # [4, d_model]
            bias_f = tau_proj_feature.bias.detach().cpu()      # [4]

            tau_params = {
                'tau_bias': {name: bias_f[i].item() for i, name in enumerate(feature_names)},
                'tau_weight_norm': {name: weight_f[i].norm().item() for i, name in enumerate(feature_names)},
                'tau_weight_std': {name: weight_f[i].std().item() for i, name in enumerate(feature_names)},
            }

            print("Feature Tau (input-based):")
            print("Pool         Bias      Weight Norm   Weight Std")
            print("-" * 50)
            for name in feature_names:
                print(f"{name:12} {tau_params['tau_bias'][name]:8.4f}  "
                      f"{tau_params['tau_weight_norm'][name]:10.4f}   "
                      f"{tau_params['tau_weight_std'][name]:.4f}")

            # Q/K Differentiation (feature only for v18.4/v18.5)
            qk_diff = {
                'fqk_bias_diff': abs(bias_f[0].item() - bias_f[1].item()),
                'fqk_weight_cosine': F.cosine_similarity(weight_f[0:1], weight_f[1:2]).item(),
            }

            # v18.5: context-based restore tau (Q/K separated)
            if hasattr(self.model.router, 'tau_proj_restore_Q'):
                print("\nRestore Tau (context-based, v18.5):")
                restore_projs = {
                    'restore_Q': self.model.router.tau_proj_restore_Q,
                    'restore_K': self.model.router.tau_proj_restore_K,
                    'restore_v': self.model.router.tau_proj_restore_v,
                    'restore_know': self.model.router.tau_proj_restore_know,
                }
                print("Pool         Bias      Weight Norm   Weight Std")
                print("-" * 50)
                for name, proj in restore_projs.items():
                    w = proj.weight.detach().cpu()
                    b = proj.bias.detach().cpu().item()
                    tau_params['tau_weight_norm'][name] = w.norm().item()
                    tau_params['tau_weight_std'][name] = w.std().item()
                    tau_params['tau_bias'][name] = b
                    print(f"{name:12} {b:8.4f}  {w.norm().item():10.4f}   {w.std().item():.4f}")

            tau_params['qk_differentiation'] = qk_diff

            print("\n--- Q/K Tau Differentiation (Feature) ---")
            print(f"  Feature Q/K bias diff:     {qk_diff['fqk_bias_diff']:.4f}")
            print(f"  Feature Q/K weight cosine: {qk_diff['fqk_weight_cosine']:.4f}")
            print("  (Restore Q/K: context-based, differentiated by input)")

        else:
            # v18.0-18.3 structure (original code)
            tau_proj = self.model.router.tau_proj
            pool_names = ['fq', 'fk', 'fv', 'rq', 'rk', 'rv', 'feature_know', 'restore_know']

            weight = tau_proj.weight.detach().cpu()  # [8, d_model]
            bias = tau_proj.bias.detach().cpu()      # [8]

            tau_params = {
                'tau_bias': {name: bias[i].item() for i, name in enumerate(pool_names)},
                'tau_weight_norm': {name: weight[i].norm().item() for i, name in enumerate(pool_names)},
                'tau_weight_std': {name: weight[i].std().item() for i, name in enumerate(pool_names)},
            }

            print("Pool         Bias      Weight Norm   Weight Std")
            print("-" * 50)
            for name in pool_names:
                print(f"{name:12} {tau_params['tau_bias'][name]:8.4f}  "
                      f"{tau_params['tau_weight_norm'][name]:10.4f}   "
                      f"{tau_params['tau_weight_std'][name]:.4f}")

            # Q/K Differentiation
            qk_diff = {
                'fqk_bias_diff': abs(bias[0].item() - bias[1].item()),
                'rqk_bias_diff': abs(bias[3].item() - bias[4].item()),
                'fqk_weight_cosine': F.cosine_similarity(weight[0:1], weight[1:2]).item(),
                'rqk_weight_cosine': F.cosine_similarity(weight[3:4], weight[4:5]).item(),
            }
            tau_params['qk_differentiation'] = qk_diff

            print("\n--- Q/K Tau Differentiation ---")
            print(f"  Feature Q/K bias diff:     {qk_diff['fqk_bias_diff']:.4f}")
            print(f"  Restore Q/K bias diff:     {qk_diff['rqk_bias_diff']:.4f}")
            print(f"  Feature Q/K weight cosine: {qk_diff['fqk_weight_cosine']:.4f}")
            print(f"  Restore Q/K weight cosine: {qk_diff['rqk_weight_cosine']:.4f}")

            if qk_diff['fqk_bias_diff'] > 0.1 or qk_diff['rqk_bias_diff'] > 0.1:
                print("  -> Q and K have learned DIFFERENT tau values (good!)")
            else:
                print("  -> Q and K tau values are similar (may need more training)")

        results['tau_parameters'] = tau_params

        # 2. Runtime Analysis (requires dataloader)
        if dataloader is not None:
            from collections import defaultdict
            import numpy as np

            print("\n--- Runtime Gate/Tau Analysis ---")

            gate_stats = defaultdict(list)
            tau_runtime = defaultdict(list)
            qk_patterns = defaultdict(list)
            conf_stats = defaultdict(list)

            self.model.eval()

            # Enable debug_mode to get gate/tau/overlap stats in routing_info
            if hasattr(self.model, 'router') and hasattr(self.model.router, 'debug_mode'):
                self.model.router.debug_mode = True

            batch_count = 0

            with torch.no_grad():
                for batch in dataloader:
                    if batch_count >= n_batches:
                        break

                    if isinstance(batch, dict):
                        input_ids = batch['input_ids'].to(self.device)
                    elif isinstance(batch, (list, tuple)):
                        input_ids = batch[0].to(self.device)
                    else:
                        input_ids = batch.to(self.device)

                    outputs = self.model(input_ids, return_routing_info=True)

                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        routing_infos = outputs[-1]

                        for layer_idx, layer_info in enumerate(routing_infos):
                            attn = layer_info.get('attention', layer_info)
                            know = layer_info.get('knowledge', {})

                            # Gate statistics - v18.4/v18.5 use gstr_* keys
                            for pool in ['fq', 'fk', 'fv']:
                                # Try new key first, then old
                                gstr_key = f'gstr_{pool}'
                                old_key = f'gate_{pool}_mean'
                                if gstr_key in attn:
                                    gate_stats[f'L{layer_idx}_{pool}_gate'].append(attn[gstr_key])
                                elif old_key in attn:
                                    gate_stats[f'L{layer_idx}_{pool}_gate'].append(attn[old_key])

                            # v18.5 restore gate_strength (context-based)
                            for key, pool in [('gstr_rqk_Q', 'rq'), ('gstr_rqk_K', 'rk'), ('gstr_rv', 'rv')]:
                                if key in attn:
                                    gate_stats[f'L{layer_idx}_{pool}_gate'].append(attn[key])

                            # Old-style restore gates (v18.0-18.3)
                            for pool in ['rq', 'rk', 'rv']:
                                old_key = f'gate_{pool}_mean'
                                if old_key in attn and f'L{layer_idx}_{pool}_gate' not in gate_stats:
                                    gate_stats[f'L{layer_idx}_{pool}_gate'].append(attn[old_key])

                            # Tau statistics - v18.4/v18.5 use tau_offset_* keys
                            for pool in ['fq', 'fk', 'fv']:
                                new_key = f'tau_offset_{pool}'
                                old_key = f'tau_{pool}'
                                if new_key in attn:
                                    tau_runtime[f'L{layer_idx}_{pool}_tau'].append(attn[new_key])
                                elif old_key in attn:
                                    tau_runtime[f'L{layer_idx}_{pool}_tau'].append(attn[old_key])

                            # v18.5 restore tau_offset (context-based)
                            for key, pool in [('tau_offset_rqk_Q', 'rq'), ('tau_offset_rqk_K', 'rk'), ('tau_offset_rv', 'rv')]:
                                if key in attn:
                                    tau_runtime[f'L{layer_idx}_{pool}_tau'].append(attn[key])

                            # Knowledge tau
                            if 'tau_offset_feature' in know:
                                tau_runtime[f'L{layer_idx}_kf_tau'].append(know['tau_offset_feature'])
                            if 'tau_offset_restore' in know:
                                tau_runtime[f'L{layer_idx}_kr_tau'].append(know['tau_offset_restore'])

                            # Knowledge gate_strength
                            if 'gstr_feature' in know:
                                gate_stats[f'L{layer_idx}_kf_gate'].append(know['gstr_feature'])
                            if 'gstr_restore' in know:
                                gate_stats[f'L{layer_idx}_kr_gate'].append(know['gstr_restore'])

                            # Q/K patterns
                            if 'overlap_fqk' in attn:
                                qk_patterns[f'L{layer_idx}_fqk_overlap'].append(attn['overlap_fqk'])
                            if 'overlap_rqk' in attn:
                                qk_patterns[f'L{layer_idx}_rqk_overlap'].append(attn['overlap_rqk'])

                            # v18.3: Confidence statistics
                            for pool in ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']:
                                conf_key = f'conf_{pool}_mean'
                                if conf_key in attn:
                                    conf_stats[f'L{layer_idx}_{pool}_conf'].append(attn[conf_key])

                    batch_count += 1

            # Disable debug_mode after analysis
            if hasattr(self.model, 'router') and hasattr(self.model.router, 'debug_mode'):
                self.model.router.debug_mode = False

            # Aggregate gate stats
            gate_summary = {}
            for key, values in gate_stats.items():
                if values:
                    gate_summary[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                    }
            results['gate_distribution'] = gate_summary

            # Aggregate tau runtime
            tau_summary = {}
            for key, values in tau_runtime.items():
                if values:
                    tau_summary[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                    }
            results['tau_runtime'] = tau_summary

            # Aggregate Q/K patterns
            qk_summary = {}
            for key, values in qk_patterns.items():
                if values:
                    qk_summary[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                    }
            results['qk_patterns'] = qk_summary

            # Aggregate confidence stats (v18.3)
            conf_summary = {}
            for key, values in conf_stats.items():
                if values:
                    conf_summary[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                    }
            if conf_summary:
                results['confidence'] = conf_summary

            # Print per-layer summary
            n_layers = self.model.n_layers
            print("\nPer-Layer Gate Mean:")
            print("Layer   FQ      FK      FV      RQ      RK      RV")
            print("-" * 60)
            for layer_idx in range(n_layers):
                row = f"L{layer_idx:2}    "
                for pool in ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']:
                    key = f'L{layer_idx}_{pool}_gate'
                    val = gate_summary.get(key, {}).get('mean', 0)
                    row += f"{val:7.3f} "
                print(row)

            print("\nPer-Layer Q/K Overlap:")
            print("Layer   FQK     RQK")
            print("-" * 30)
            for layer_idx in range(n_layers):
                fqk = qk_summary.get(f'L{layer_idx}_fqk_overlap', {}).get('mean', 0)
                rqk = qk_summary.get(f'L{layer_idx}_rqk_overlap', {}).get('mean', 0)
                print(f"L{layer_idx:2}    {fqk:7.3f} {rqk:7.3f}")

            # v18.3+: Print confidence stats
            if conf_summary:
                print(f"\nPer-Layer Confidence Mean (v{model_version}):")
                print("Layer   FQ      FK      FV      RQ      RK      RV")
                print("-" * 60)
                for layer_idx in range(n_layers):
                    row = f"L{layer_idx:2}    "
                    for pool in ['fq', 'fk', 'fv', 'rq', 'rk', 'rv']:
                        key = f'L{layer_idx}_{pool}_conf'
                        val = conf_summary.get(key, {}).get('mean', 0)
                        row += f"{val:7.3f} "
                    print(row)

        # Save results
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'v18_analysis.json')
            save_results(results, output_path)
            print(f"\nResults saved to: {output_path}")

        return results

    def run_trajectory_analysis(self, dataloader, n_batches: int = 20, output_dir: str = None) -> Dict:
        """Token trajectory analysis."""
        print("\n" + "="*60)
        print("TOKEN TRAJECTORY ANALYSIS")
        print("="*60)

        results = self.behavioral.analyze_token_trajectory(dataloader, n_batches)

        for key, data in results.items():
            if isinstance(data, dict) and 'display' in data:
                print(f"\n{data['display']}:")
                print(f"  Early positions (0-9) avg entropy: {data.get('early_avg', 0):.1f}%")
                print(f"  Late positions (10+) avg entropy: {data.get('late_avg', 0):.1f}%")

        return results

    def run_probing_analysis(self, dataloader, max_batches: int = 50, output_dir: str = None) -> Dict:
        """Probing classifier analysis."""
        print("\n" + "="*60)
        print("PROBING CLASSIFIER")
        print("="*60)

        results = self.behavioral.run_probing(dataloader, max_batches)

        for key, data in results.items():
            if 'error' in data:
                print(f"{key}: {data['error']}")
            elif isinstance(data, dict) and 'display' in data:
                print(f"{data['display']}: accuracy={data['accuracy']:.2%}, samples={data['n_samples']}")

        return results

    def run_ablation_analysis(self, dataloader, max_batches: int = 20, output_dir: str = None) -> Dict:
        """Ablation study."""
        print("\n" + "="*60)
        print("ABLATION STUDY")
        print("="*60)

        results = self.behavioral.run_ablation(dataloader, max_batches)

        print(f"Baseline loss: {results.get('baseline_loss', 0):.4f}")
        for name, data in results.items():
            if name == 'baseline_loss' or not isinstance(data, list):
                continue
            print(f"\n{NEURON_TYPES.get(name, (name,))[0]}:")
            for r in data[:3]:
                print(f"  Neuron {r['neuron_id']}: delta={r['loss_delta']:.4f}, "
                      f"importance={r['importance']:.2%}")

        return results

    def analyze_single_neuron(self, neuron_id: int, neuron_type: str) -> Dict:
        """Single neuron analysis."""
        print("\n" + "="*60)
        print(f"SINGLE NEURON ANALYSIS: {neuron_type} #{neuron_id}")
        print("="*60)

        results = self.behavioral.analyze_single_neuron(neuron_id, neuron_type)

        print(f"Usage EMA: {results.get('usage_ema', 'N/A')}")
        print(f"Embedding norm: {results.get('embedding_norm', 'N/A')}")

        return results

    def run_all(self, dataloader=None, output_dir: str = './dawn_analysis', n_batches: int = 50) -> Dict:
        """Run all analyses."""
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'usage': self.run_usage_analysis(output_dir),
            'embedding': self.run_embedding_analysis(output_dir),
            'weight': self.run_weight_analysis(output_dir),
            'clustering': self.run_clustering_analysis(output_dir=output_dir),
            'semantic': self.run_semantic_analysis(dataloader, n_batches, output_dir),
        }

        if dataloader:
            results['routing'] = self.run_routing_analysis(dataloader, n_batches, output_dir)
            results['trajectory'] = self.run_trajectory_analysis(dataloader, n_batches, output_dir)
            results['coselection'] = self.run_coselection_analysis(dataloader, n_batches, output_dir)

        # v18.x specific analysis (auto-detect)
        if hasattr(self.model, 'router') and hasattr(self.model.router, 'tau_proj'):
            results['v18'] = self.run_v18_analysis(dataloader, n_batches, output_dir)

        output_path = os.path.join(output_dir, 'dawn_analysis.json')
        save_results(results, output_path)
        print(f"\nResults saved to: {output_path}")

        return results


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='DAWN Model Analysis Suite')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'usage', 'routing', 'embedding', 'weight_svd',
                                'clustering', 'trajectory', 'probing', 'ablation',
                                'neuron', 'semantic', 'coselection', 'paper', 'v18'],
                        help='Analysis mode')
    parser.add_argument('--figure', type=str, default=None,
                        help='Generate paper figures: 3,4,6,6a,6b,7 or "all"')
    parser.add_argument('--val_data', type=str, default=None, help='Validation data path')
    parser.add_argument('--output_dir', type=str, default='./dawn_analysis', help='Output directory')
    parser.add_argument('--max_batches', type=int, default=50, help='Number of batches')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of clusters')
    parser.add_argument('--neuron_type', type=str, default='feature_qk', help='Neuron type for single neuron analysis')
    parser.add_argument('--neuron_id', type=int, default=0, help='Neuron ID for single neuron analysis')
    parser.add_argument('--device', type=str, default='cuda', help='Device')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Handle --figure option (paper figure generation)
    if args.figure is not None:
        gen = PaperFigureGenerator(args.checkpoint, args.val_data, str(device))
        results = gen.generate(args.figure, args.output_dir, args.max_batches)
        save_results(results, os.path.join(args.output_dir, 'figure_results.json'))
        print(f"\n{'='*60}")
        print(f"Paper figures generated: {args.output_dir}")
        print(f"Figures: {args.figure}")
        print(f"{'='*60}")
        return

    # For paper mode, use PaperFigureGenerator directly
    if args.mode == 'paper':
        gen = PaperFigureGenerator(args.checkpoint, args.val_data, str(device))
        if args.val_data:
            results = gen.generate_all(args.output_dir, args.max_batches)
        else:
            results = gen.run_quick(args.output_dir)
        print(f"\n{'='*60}")
        print(f"Paper figures generated: {args.output_dir}")
        print(f"{'='*60}")
        return

    # Standard analysis
    model, tokenizer, config = load_model(args.checkpoint, device)
    analyzer = DAWNAnalyzer(model, tokenizer, device)

    dataloader = None
    if args.val_data and args.mode in ['all', 'routing', 'trajectory', 'probing',
                                        'ablation', 'semantic', 'coselection', 'v18']:
        dataloader = create_dataloader(args.val_data, tokenizer, args.batch_size)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == 'all':
        results = analyzer.run_all(dataloader, args.output_dir, args.max_batches)
    elif args.mode == 'usage':
        results = analyzer.run_usage_analysis(args.output_dir)
    elif args.mode == 'routing':
        if dataloader is None:
            print("ERROR: --val_data required for routing analysis")
            return
        results = analyzer.run_routing_analysis(dataloader, args.max_batches, args.output_dir)
    elif args.mode == 'embedding':
        results = analyzer.run_embedding_analysis(args.output_dir)
    elif args.mode == 'weight_svd':
        results = analyzer.run_weight_analysis(args.output_dir)
    elif args.mode == 'clustering':
        results = analyzer.run_clustering_analysis(args.n_clusters, args.output_dir)
    elif args.mode == 'trajectory':
        if dataloader is None:
            print("ERROR: --val_data required for trajectory analysis")
            return
        results = analyzer.run_trajectory_analysis(dataloader, args.max_batches, args.output_dir)
    elif args.mode == 'probing':
        if dataloader is None:
            print("ERROR: --val_data required for probing analysis")
            return
        results = analyzer.run_probing_analysis(dataloader, args.max_batches, args.output_dir)
    elif args.mode == 'ablation':
        if dataloader is None:
            print("ERROR: --val_data required for ablation analysis")
            return
        results = analyzer.run_ablation_analysis(dataloader, args.max_batches, args.output_dir)
    elif args.mode == 'neuron':
        results = analyzer.analyze_single_neuron(args.neuron_id, args.neuron_type)
    elif args.mode == 'semantic':
        results = analyzer.run_semantic_analysis(dataloader, args.max_batches, args.output_dir)
    elif args.mode == 'coselection':
        if dataloader is None:
            print("ERROR: --val_data required for coselection analysis")
            return
        results = analyzer.run_coselection_analysis(dataloader, args.max_batches, args.output_dir)
    elif args.mode == 'v18':
        # v18.x specific analysis (tau, gate, confidence, Q/K differentiation)
        # dataloader is optional - without it, only static tau params are analyzed
        # v18.3 adds confidence stats
        results = analyzer.run_v18_analysis(dataloader, args.max_batches, args.output_dir)

    output_path = os.path.join(args.output_dir, f'dawn_{args.mode}.json')
    save_results(results, output_path)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Results: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
