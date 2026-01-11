"""
Visualization Tools for Refusal Circuit Analysis

Publication-quality figures for presenting mechanistic interpretability results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple
from pathlib import Path

# Use a clean, publication-ready style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette
COLORS = {
    "refusal": "#e63946",      # Red - for refusal
    "compliant": "#2a9d8f",    # Teal - for compliant  
    "neutral": "#457b9d",      # Blue - neutral
    "highlight": "#f4a261",    # Orange - highlights
    "background": "#1d3557",   # Dark blue - backgrounds
}


def plot_patching_heatmap(
    effect_matrix: np.ndarray,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    title: str = "Activation Patching Results",
    xlabel: str = "Token Position",
    ylabel: str = "Layer",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a heatmap of patching effects.
    
    Args:
        effect_matrix: 2D array of effect sizes [layers, positions]
        x_labels: Labels for x-axis (positions)
        y_labels: Labels for y-axis (layers)
        title: Plot title
        xlabel/ylabel: Axis labels
        cmap: Colormap (RdBu_r centers on 0)
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Center colormap on 0
    vmax = max(abs(effect_matrix.min()), abs(effect_matrix.max()))
    vmin = -vmax
    
    sns.heatmap(
        effect_matrix,
        ax=ax,
        cmap=cmap,
        center=0,
        vmin=vmin,
        vmax=vmax,
        xticklabels=x_labels if x_labels else True,
        yticklabels=y_labels if y_labels else True,
        cbar_kws={"label": "Effect Size (Δ refusal score)"},
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_layer_importance(
    layer_scores: List[float],
    title: str = "Layer Importance for Refusal",
    highlight_threshold: float = 0.1,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar plot showing importance of each layer.
    
    Args:
        layer_scores: Importance score for each layer
        title: Plot title
        highlight_threshold: Threshold for highlighting important layers
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_layers = len(layer_scores)
    x = np.arange(n_layers)
    
    # Color bars based on importance
    colors = [
        COLORS["refusal"] if abs(s) > highlight_threshold else COLORS["neutral"]
        for s in layer_scores
    ]
    
    bars = ax.bar(x, layer_scores, color=colors, edgecolor='black', linewidth=0.5)
    
    # Add threshold line
    ax.axhline(y=highlight_threshold, color=COLORS["highlight"], linestyle='--', 
               label=f'Threshold ({highlight_threshold})', alpha=0.7)
    ax.axhline(y=-highlight_threshold, color=COLORS["highlight"], linestyle='--', alpha=0.7)
    
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Importance Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_head_importance(
    head_scores: np.ndarray,  # [n_layers, n_heads]
    title: str = "Attention Head Importance",
    top_k: int = 10,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap showing importance of each attention head.
    
    Args:
        head_scores: 2D array [layers, heads] of importance scores
        title: Plot title
        top_k: Number of top heads to annotate
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_layers, n_heads = head_scores.shape
    
    # Heatmap
    vmax = max(abs(head_scores.min()), abs(head_scores.max()))
    im = ax.imshow(head_scores, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    
    # Annotate top-k heads
    flat_idx = np.argsort(np.abs(head_scores.flatten()))[-top_k:]
    for idx in flat_idx:
        layer = idx // n_heads
        head = idx % n_heads
        score = head_scores[layer, head]
        ax.add_patch(plt.Rectangle(
            (head - 0.5, layer - 0.5), 1, 1,
            fill=False, edgecolor=COLORS["highlight"], linewidth=2
        ))
    
    ax.set_xlabel("Attention Head", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label="Importance Score")
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_refusal_direction_separation(
    refusal_projections: np.ndarray,
    compliant_projections: np.ndarray,
    layer: int,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize separation between refusal and compliant activations.
    
    Shows how well the refusal direction separates the two classes.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create violin/box plots
    data = [refusal_projections, compliant_projections]
    positions = [0, 1]
    
    parts = ax.violinplot(data, positions, showmeans=True, showmedians=True)
    
    # Color the violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(COLORS["refusal"] if i == 0 else COLORS["compliant"])
        pc.set_alpha(0.7)
    
    # Add individual points with jitter
    for i, (proj, color) in enumerate([
        (refusal_projections, COLORS["refusal"]),
        (compliant_projections, COLORS["compliant"])
    ]):
        jitter = np.random.normal(0, 0.04, len(proj))
        ax.scatter(
            np.full_like(proj, i) + jitter,
            proj,
            c=color, alpha=0.5, s=30, edgecolors='white', linewidth=0.5
        )
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Refusal Prompts', 'Compliant Prompts'])
    ax.set_ylabel('Projection onto Refusal Direction', fontsize=12)
    
    if title is None:
        title = f'Refusal Direction Separation (Layer {layer})'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add separation score annotation
    mean_diff = refusal_projections.mean() - compliant_projections.mean()
    pooled_std = np.sqrt((refusal_projections.var() + compliant_projections.var()) / 2)
    sep_score = mean_diff / (pooled_std + 1e-8)
    
    ax.text(
        0.02, 0.98, f'Separation: {sep_score:.2f}σ',
        transform=ax.transAxes, fontsize=11,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_steering_effects(
    coefficients: List[float],
    refusal_rates: List[float],
    title: str = "Effect of Steering on Refusal Rate",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot how steering coefficient affects refusal rate.
    
    Args:
        coefficients: List of steering coefficients tested
        refusal_rates: Refusal rate at each coefficient
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(coefficients, refusal_rates, 'o-', 
            color=COLORS["refusal"], linewidth=2, markersize=8)
    
    # Highlight zero point
    zero_idx = coefficients.index(0.0) if 0.0 in coefficients else None
    if zero_idx is not None:
        ax.scatter([0], [refusal_rates[zero_idx]], 
                  color=COLORS["highlight"], s=150, zorder=5, 
                  label='No steering')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Steering Coefficient', fontsize=12)
    ax.set_ylabel('Refusal Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_circuit_diagram(
    critical_heads: List[Tuple[int, int]],
    critical_mlps: List[int],
    n_layers: int,
    n_heads: int,
    title: str = "Identified Refusal Circuit",
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a visual diagram of the identified circuit.
    
    Shows which attention heads and MLPs are part of the refusal circuit.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Setup grid
    head_width = 0.8 / n_heads
    layer_height = 0.8 / n_layers
    
    # Draw layers
    for layer in range(n_layers):
        y = 0.9 - layer * layer_height
        
        # Layer label
        ax.text(-0.02, y, f'L{layer}', fontsize=9, ha='right', va='center')
        
        # Draw attention heads
        for head in range(n_heads):
            x = 0.05 + head * head_width
            
            # Check if this head is critical
            is_critical = (layer, head) in critical_heads
            color = COLORS["refusal"] if is_critical else '#e0e0e0'
            
            rect = plt.Rectangle(
                (x, y - layer_height/4), head_width * 0.8, layer_height/2,
                facecolor=color, edgecolor='black', linewidth=0.5
            )
            ax.add_patch(rect)
            
            if is_critical:
                ax.text(x + head_width * 0.4, y, f'H{head}', 
                       fontsize=7, ha='center', va='center', color='white')
        
        # Draw MLP
        mlp_x = 0.05 + n_heads * head_width + 0.05
        is_critical_mlp = layer in critical_mlps
        mlp_color = COLORS["refusal"] if is_critical_mlp else '#e0e0e0'
        
        mlp_rect = plt.Rectangle(
            (mlp_x, y - layer_height/4), 0.08, layer_height/2,
            facecolor=mlp_color, edgecolor='black', linewidth=0.5
        )
        ax.add_patch(mlp_rect)
        ax.text(mlp_x + 0.04, y, 'MLP', fontsize=7, ha='center', va='center')
    
    # Legend
    legend_elements = [
        plt.Rectangle((0,0), 1, 1, facecolor=COLORS["refusal"], 
                      edgecolor='black', label='Critical Component'),
        plt.Rectangle((0,0), 1, 1, facecolor='#e0e0e0', 
                      edgecolor='black', label='Non-critical'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlim(-0.05, 1.0)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_all_figures(
    figures: Dict[str, plt.Figure],
    output_dir: str = "figures",
) -> None:
    """Save all figures to a directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, fig in figures.items():
        filepath = output_path / f"{name}.png"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved {filepath}")
