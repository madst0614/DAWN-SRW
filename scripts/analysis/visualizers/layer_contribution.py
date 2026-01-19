"""
Layer Contribution Visualizations
=================================
Visualization for layer-wise circuit contribution analysis.

Paper Figure 7b: Layer-wise Attention vs Knowledge Contribution

Style matches figures/fig4_routing_stats.py for paper-quality output.
"""

import os
import numpy as np
from typing import Dict, Optional
import matplotlib.patches as mpatches

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

# Paper-quality style settings
if HAS_MATPLOTLIB:
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

# Color palette (matching figures/fig4_routing_stats.py)
COLOR_ATTENTION = '#4A90D9'  # Blue
COLOR_KNOWLEDGE = '#50C878'  # Green
COLOR_Q = '#E74C3C'          # Red for Q routing
COLOR_K = '#3498DB'          # Blue for K routing
COLOR_V = '#9B59B6'          # Purple for V routing
COLOR_BLACK = '#2C3E50'
COLOR_GRAY = '#7F8C8D'


def plot_layer_contribution(
    contribution_data: Dict,
    output_path: str,
    dpi: int = 300
) -> Optional[str]:
    """
    Generate layer-wise contribution visualization (Paper Figure 7b style).

    Matches the style of figures/fig4_routing_stats.py:
    - Clean serif fonts
    - Fill between for attention vs knowledge areas
    - Proper legend with patches

    Args:
        contribution_data: Results from RoutingAnalyzer.analyze_layer_contribution()
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    per_layer = contribution_data.get('per_layer', {})
    if not per_layer:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Extract data
    layers = sorted(per_layer.keys(), key=lambda x: int(x[1:]))
    layer_indices = [int(l[1:]) + 1 for l in layers]  # 1-indexed for display
    n_layers = len(layers)

    # Check if we have pool_breakdown data
    has_pool_breakdown = 'pool_breakdown' in per_layer[layers[0]] if layers else False

    if has_pool_breakdown:
        # Create 2-panel figure matching fig4 style
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5), dpi=dpi)

        # === (a) Neuron Utilization by Pool ===
        pool_keys = list(per_layer[layers[0]].get('pool_breakdown', {}).keys())

        # Calculate average utilization per pool across all layers
        pool_totals = {}
        for key in pool_keys:
            values = [per_layer[l].get('pool_breakdown', {}).get(key, 0) for l in layers]
            pool_totals[key] = np.mean(values) if values else 0

        # Order pools
        pool_order = ['fv', 'fqk_q', 'fqk_k', 'rv', 'rqk_q', 'rqk_k', 'fknow', 'rknow']
        pools = [p for p in pool_order if p in pool_totals][::-1]  # Reverse for display
        values = [pool_totals[p] for p in pools]

        # Map pool names to display names
        display_names = {
            'fv': 'Feature_V', 'fqk_q': 'Feature_Q', 'fqk_k': 'Feature_K',
            'rv': 'Restore_V', 'rqk_q': 'Restore_Q', 'rqk_k': 'Restore_K',
            'fknow': 'Feature_Know', 'rknow': 'Restore_Know',
        }

        # Colors based on type
        colors = []
        for p in pools:
            if 'know' in p.lower():
                colors.append(COLOR_KNOWLEDGE)
            elif '_q' in p or p.endswith('q'):
                colors.append(COLOR_Q)
            elif '_k' in p or p.endswith('k'):
                colors.append(COLOR_K)
            else:
                colors.append(COLOR_V)

        y_pos = np.arange(len(pools))
        bars = ax1.barh(y_pos, values, height=0.7, color=colors, alpha=0.85,
                       edgecolor='white', linewidth=0.5)

        # Add value labels
        max_val = max(values) if values else 1
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax1.text(val + max_val * 0.02, i, f'{val:.0f}', va='center', fontsize=8, color=COLOR_BLACK)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([display_names.get(p, p) for p in pools], fontsize=8)
        ax1.set_xlim(0, max_val * 1.15)
        ax1.set_xlabel('Avg Selected Neurons', fontsize=9)
        ax1.set_title('(a) Pool Utilization', fontsize=10, fontweight='bold', pad=10)
        ax1.xaxis.grid(True, linestyle='--', alpha=0.3)
        ax1.set_axisbelow(True)

        # Legend
        legend_elements = [
            mpatches.Patch(color=COLOR_Q, label='Q routing', alpha=0.85),
            mpatches.Patch(color=COLOR_K, label='K routing', alpha=0.85),
            mpatches.Patch(color=COLOR_V, label='V routing', alpha=0.85),
            mpatches.Patch(color=COLOR_KNOWLEDGE, label='Knowledge', alpha=0.85),
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=7, framealpha=0.9)

        # === (b) Layer-wise Circuit Contribution ===
        attn_ratios = [per_layer[l]['attention_ratio'] * 100 for l in layers]

        ax2.plot(layer_indices, attn_ratios, 'o-', color=COLOR_ATTENTION, linewidth=2,
                markersize=6, markerfacecolor='white', markeredgewidth=1.5)

        # Fill areas
        ax2.fill_between(layer_indices, attn_ratios, 50,
                        where=[a >= 50 for a in attn_ratios],
                        color=COLOR_ATTENTION, alpha=0.3)
        ax2.fill_between(layer_indices, attn_ratios, 50,
                        where=[a < 50 for a in attn_ratios],
                        color=COLOR_KNOWLEDGE, alpha=0.3)

        ax2.axhline(y=50, color=COLOR_GRAY, linestyle='--', linewidth=1.5)

        ax2.set_xlim(0.5, n_layers + 0.5)
        ax2.set_ylim(35, 75)
        ax2.set_xticks(layer_indices)
        ax2.set_xlabel('Layer', fontsize=9)
        ax2.set_ylabel('Attention Contribution (%)', fontsize=9)
        ax2.set_title('(b) Layer-wise Circuit Contribution', fontsize=10, fontweight='bold', pad=10)
        ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax2.set_axisbelow(True)

        # Legend
        legend_elements2 = [
            mpatches.Patch(color=COLOR_ATTENTION, alpha=0.3, label='Attention > 50%'),
            mpatches.Patch(color=COLOR_KNOWLEDGE, alpha=0.3, label='Knowledge > 50%'),
            plt.Line2D([0], [0], color=COLOR_GRAY, linestyle='--', label='50% baseline'),
        ]
        ax2.legend(handles=legend_elements2, loc='upper right', fontsize=7, framealpha=0.9)

    else:
        # Simple single-panel plot (layer contribution only)
        fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)

        attn_ratios = [per_layer[l]['attention_ratio'] * 100 for l in layers]

        ax.plot(layer_indices, attn_ratios, 'o-', color=COLOR_ATTENTION, linewidth=2,
               markersize=6, markerfacecolor='white', markeredgewidth=1.5)

        # Fill areas
        ax.fill_between(layer_indices, attn_ratios, 50,
                       where=[a >= 50 for a in attn_ratios],
                       color=COLOR_ATTENTION, alpha=0.3)
        ax.fill_between(layer_indices, attn_ratios, 50,
                       where=[a < 50 for a in attn_ratios],
                       color=COLOR_KNOWLEDGE, alpha=0.3)

        ax.axhline(y=50, color=COLOR_GRAY, linestyle='--', linewidth=1.5)

        ax.set_xlim(0.5, n_layers + 0.5)
        ax.set_ylim(35, 75)
        ax.set_xticks(layer_indices)
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Attention Contribution (%)', fontsize=10)
        ax.set_title('Layer-wise Circuit Contribution', fontsize=11, fontweight='bold')
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        # Legend
        legend_elements = [
            mpatches.Patch(color=COLOR_ATTENTION, alpha=0.3, label='Attention > 50%'),
            mpatches.Patch(color=COLOR_KNOWLEDGE, alpha=0.3, label='Knowledge > 50%'),
            plt.Line2D([0], [0], color=COLOR_GRAY, linestyle='--', label='50% baseline'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    plt.tight_layout()

    # Save PNG and PDF
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none', bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, format='pdf', facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()

    return output_path


def plot_layer_contribution_detail(
    contribution_data: Dict,
    output_path: str,
    dpi: int = 300
) -> Optional[str]:
    """
    Generate detailed layer contribution heatmap.

    Args:
        contribution_data: Results from RoutingAnalyzer.analyze_layer_contribution()
        output_path: Path to save the figure
        dpi: Output resolution

    Returns:
        Path to saved figure or None
    """
    if not HAS_MATPLOTLIB:
        return None

    per_layer = contribution_data.get('per_layer', {})
    if not per_layer:
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Extract data
    layers = sorted(per_layer.keys(), key=lambda x: int(x[1:]))

    # Check for pool_breakdown
    if 'pool_breakdown' not in per_layer[layers[0]]:
        # Fallback to simple visualization
        return plot_layer_contribution(contribution_data, output_path, dpi)

    pool_keys = list(per_layer[layers[0]].get('pool_breakdown', {}).keys())

    # Build matrix [pools x layers]
    matrix = np.zeros((len(pool_keys), len(layers)))
    for j, layer in enumerate(layers):
        for i, pool in enumerate(pool_keys):
            matrix[i, j] = per_layer[layer].get('pool_breakdown', {}).get(pool, 0)

    # Normalize per pool (row)
    max_vals = matrix.max(axis=1, keepdims=True)
    max_vals[max_vals == 0] = 1
    matrix_norm = matrix / max_vals

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.6), max(4, len(pool_keys) * 0.5)), dpi=dpi)

    im = ax.imshow(matrix_norm, aspect='auto', cmap='Blues')
    ax.set_yticks(range(len(pool_keys)))
    ax.set_yticklabels(pool_keys, fontsize=8)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([l[1:] for l in layers], fontsize=8)
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel('Pool', fontsize=10)
    ax.set_title('Normalized Pool Activity by Layer', fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Normalized Activity', fontsize=9)

    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=dpi, facecolor='white', edgecolor='none', bbox_inches='tight')
    pdf_path = output_path.rsplit('.', 1)[0] + '.pdf'
    plt.savefig(pdf_path, format='pdf', facecolor='white', edgecolor='none', bbox_inches='tight')
    plt.close()

    return output_path
