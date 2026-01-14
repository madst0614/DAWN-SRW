"""
Neuron Health Analysis
=======================
Analyze neuron health status in DAWN models.

Includes:
- EMA distribution analysis
- Dead/Active neuron ratio
- Diversity metrics (Gini, entropy)
- Q/K EMA overlap analysis (v18.x)
"""

import os
import numpy as np
import torch
from typing import Dict, Optional

from .base import BaseAnalyzer
from .utils import gini_coefficient


class NeuronHealthAnalyzer(BaseAnalyzer):
    """Neuron health and usage pattern analyzer."""

    def __init__(self, model, router=None, device: str = 'cuda'):
        """
        Initialize analyzer.

        Args:
            model: DAWN model instance
            router: NeuronRouter (auto-detected if None)
            device: Device for computation
        """
        super().__init__(model, router=router, device=device)

    def analyze_ema_distribution(self, threshold: float = 0.01) -> Dict:
        """
        Analyze EMA distribution across all neuron types.

        Args:
            threshold: Threshold for active/dead neuron classification

        Returns:
            Dictionary with per-type EMA statistics
        """
        results = {}
        neuron_types = self.get_neuron_types()

        for name, (display, ema_attr, n_attr, _) in neuron_types.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            n_total = getattr(self.router, n_attr)

            active = (ema > threshold).sum().item()
            dead = (ema < threshold).sum().item()

            results[name] = {
                'display': display,
                'total': n_total,
                'active': int(active),
                'dead': int(dead),
                'active_ratio': active / n_total,
                'dead_ratio': dead / n_total,
                'gini': gini_coefficient(ema),
                'stats': {
                    'min': float(ema.min()),
                    'max': float(ema.max()),
                    'mean': float(ema.mean()),
                    'std': float(ema.std()),
                    'median': float(ema.median()),
                }
            }

        return results

    def analyze_diversity(self, threshold: float = 0.01) -> Dict:
        """
        Analyze neuron diversity using entropy and effective count.

        Args:
            threshold: Threshold for active neuron classification

        Returns:
            Dictionary with diversity metrics
        """
        results = {}
        neuron_types = self.get_neuron_types()

        for name, (display, ema_attr, n_attr, _) in neuron_types.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            n_total = getattr(self.router, n_attr)

            active_mask = ema > threshold
            n_active = active_mask.sum().item()

            if n_active == 0:
                results[name] = {
                    'display': display,
                    'n_active': 0,
                    'entropy': 0,
                    'effective_count': 0,
                    'coverage': 0,
                }
                continue

            active_ema = ema[active_mask]
            p = active_ema / active_ema.sum()

            entropy = -torch.sum(p * torch.log(p + 1e-8)).item()
            max_entropy = np.log(n_active)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            effective_count = np.exp(entropy)

            top5 = torch.topk(active_ema, min(5, n_active))[0]
            top5_share = top5.sum() / active_ema.sum()

            results[name] = {
                'display': display,
                'n_active': n_active,
                'n_total': n_total,
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'effective_count': float(effective_count),
                'coverage': n_active / n_total,
                'top5_share': float(top5_share),
                'gini': gini_coefficient(ema),
            }

        # Overall diversity score
        entropies = [
            r['normalized_entropy']
            for r in results.values()
            if isinstance(r, dict) and 'normalized_entropy' in r
        ]
        overall = sum(entropies) / len(entropies) if entropies else 0

        results['overall'] = {
            'diversity_score': overall,
            'health': 'good' if overall > 0.7 else 'warning' if overall > 0.4 else 'critical'
        }

        return results

    def analyze_dead_neurons(self, output_dir: Optional[str] = None) -> Dict:
        """
        Analyze dead neurons and provide shrink recommendations.

        Args:
            output_dir: Directory for visualization output

        Returns:
            Dictionary with dead neuron analysis and recommendations
        """
        results = {}
        threshold = 0.01
        dying_threshold = 0.05
        neuron_types = self.get_neuron_types()

        for name, (display, ema_attr, n_attr, _) in neuron_types.items():
            if not hasattr(self.router, ema_attr):
                continue

            ema = getattr(self.router, ema_attr)
            n_total = getattr(self.router, n_attr)

            dead_mask = ema < threshold
            dying_mask = (ema >= threshold) & (ema < dying_threshold)
            active_mask = ema >= dying_threshold

            results[name] = {
                'display': display,
                'n_total': n_total,
                'n_active': int(active_mask.sum()),
                'n_dying': int(dying_mask.sum()),
                'n_dead': int(dead_mask.sum()),
                'n_removable': int(dead_mask.sum()),  # Dead neurons can be removed
                'dead_neuron_ids': dead_mask.nonzero().squeeze(-1).tolist()
                                   if dead_mask.sum() > 0 else [],
            }

        # Calculate recommendations
        type_names = [
            name for name in results.keys()
            if isinstance(results[name], dict) and 'n_total' in results[name]
        ]

        total_dead = sum(results[n]['n_dead'] for n in type_names)
        total_neurons = sum(results[n]['n_total'] for n in type_names)

        results['summary'] = {
            'total_dead': total_dead,
            'total_neurons': total_neurons,
            'dead_ratio': total_dead / total_neurons if total_neurons > 0 else 0,
            'per_type': {
                name: {
                    'total': results[name]['n_total'],
                    'active': results[name]['n_active'],
                    'dead': results[name]['n_dead'],
                }
                for name in type_names
            }
        }

        # Visualization
        if output_dir:
            from .visualizers import plot_dead_neurons
            os.makedirs(output_dir, exist_ok=True)
            path = plot_dead_neurons(results, type_names, os.path.join(output_dir, 'dead_neurons.png'))
            if path:
                results['visualization'] = path

        return results

    def visualize_usage(self, output_dir: str) -> Dict:
        """
        Create usage histogram plots.

        Args:
            output_dir: Directory for output

        Returns:
            Dictionary with visualization path
        """
        from .visualizers import plot_usage_histogram

        os.makedirs(output_dir, exist_ok=True)
        neuron_types = self.get_neuron_types()

        # Collect EMA data
        ema_data = []
        for name, (display, ema_attr, _, color) in neuron_types.items():
            if hasattr(self.router, ema_attr):
                ema = getattr(self.router, ema_attr)
                ema_data.append((display, ema.detach().cpu().numpy(), color))

        path = plot_usage_histogram(ema_data, os.path.join(output_dir, 'usage_histogram.png'))

        if path:
            return {'visualization': path}
        return {'error': 'matplotlib not available'}

    def analyze_qk_ema_overlap(self, threshold: float = 0.01) -> Dict:
        """
        Analyze Q/K neuron specialization from EMA (v18.x only).

        Classifies neurons into:
        - q_only: Used only by Q routing
        - k_only: Used only by K routing
        - shared: Used by both Q and K
        - dead: Not used by either

        Args:
            threshold: EMA threshold for active classification

        Returns:
            Dictionary with Q/K overlap statistics
        """
        results = {}

        # Check for v18.x separate Q/K EMA
        qk_pairs = [
            ('fqk', 'usage_ema_feature_q', 'usage_ema_feature_k', 'n_feature_qk'),
            ('rqk', 'usage_ema_restore_q', 'usage_ema_restore_k', 'n_restore_qk'),
        ]

        for pool_name, q_attr, k_attr, n_attr in qk_pairs:
            if not hasattr(self.router, q_attr) or not hasattr(self.router, k_attr):
                continue

            ema_q = getattr(self.router, q_attr).cpu().numpy()
            ema_k = getattr(self.router, k_attr).cpu().numpy()
            n_total = getattr(self.router, n_attr)

            q_active = ema_q > threshold
            k_active = ema_k > threshold

            q_only = int((q_active & ~k_active).sum())
            k_only = int((~q_active & k_active).sum())
            shared = int((q_active & k_active).sum())
            dead = int((~q_active & ~k_active).sum())

            # Correlation between Q and K usage
            if ema_q.sum() > 0 and ema_k.sum() > 0:
                corr = float(np.corrcoef(ema_q, ema_k)[0, 1])
            else:
                corr = 0.0

            # Jaccard similarity = intersection / union
            union = q_only + k_only + shared
            jaccard = shared / union if union > 0 else 0.0

            results[pool_name] = {
                'n_total': n_total,
                'q_only': q_only,
                'k_only': k_only,
                'shared': shared,
                'dead': dead,
                'q_active': int(q_active.sum()),
                'k_active': int(k_active.sum()),
                'correlation': corr,
                'jaccard': float(jaccard),
                'q_ema': ema_q.tolist(),
                'k_ema': ema_k.tolist(),
            }

        if not results:
            return {'error': 'Not a v18.x model with separate Q/K EMA'}

        return results

    def visualize_qk_overlap(self, output_dir: str) -> Optional[str]:
        """
        Visualize Q/K EMA overlap.

        Args:
            output_dir: Directory for output

        Returns:
            Path to saved figure or None
        """
        from .visualizers import plot_qk_ema_overlap

        qk_results = self.analyze_qk_ema_overlap()
        if 'error' in qk_results:
            return None

        os.makedirs(output_dir, exist_ok=True)
        return plot_qk_ema_overlap(qk_results, os.path.join(output_dir, 'qk_ema_overlap.png'))

    def run_all(self, output_dir: str = './neuron_health') -> Dict:
        """
        Run all neuron health analyses.

        Args:
            output_dir: Directory for outputs

        Returns:
            Combined results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'ema_distribution': self.analyze_ema_distribution(),
            'diversity': self.analyze_diversity(),
            'dead_neurons': self.analyze_dead_neurons(output_dir),
        }

        # v18.x Q/K overlap analysis
        qk_overlap = self.analyze_qk_ema_overlap()
        if 'error' not in qk_overlap:
            results['qk_ema_overlap'] = qk_overlap
            # Visualize Q/K overlap
            self.visualize_qk_overlap(output_dir)

        # Visualizations
        results['visualization'] = self.visualize_usage(output_dir)

        return results
