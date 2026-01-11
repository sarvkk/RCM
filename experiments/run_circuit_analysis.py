#!/usr/bin/env python3
"""
Main Experiment: Refusal Circuit Localization

This script runs the complete pipeline to identify refusal circuits:
1. Load model and prompt pairs
2. Run activation patching experiments  
3. Identify critical components
4. Compute refusal direction
5. Validate with clamping experiments
6. Generate visualizations

Usage:
    python run_circuit_analysis.py --model pythia-410m --n-pairs 10
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import load_model
from src.data import REFUSAL_PROMPT_PAIRS, get_prompt_pairs_by_category
from src.circuits import CircuitAnalyzer, find_refusal_circuit, compute_refusal_direction
from src.steering import ClampingExperiment, run_clamping_validation
from src.utils import (
    plot_patching_heatmap,
    plot_layer_importance,
    plot_head_importance,
    plot_refusal_direction_separation,
    create_circuit_diagram,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run refusal circuit analysis")
    parser.add_argument(
        "--model", 
        type=str, 
        default="pythia-410m",
        help="Model to analyze (e.g., pythia-410m, gpt2-medium)"
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=10,
        help="Number of prompt pairs to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--components",
        type=str,
        default="all",
        choices=["resid", "attn", "mlp", "all"],
        help="Which components to analyze"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.model}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("REFUSAL CIRCUIT ANALYSIS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Prompt pairs: {args.n_pairs}")
    print(f"Components: {args.components}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Step 1: Load model
    print("\n[1/6] Loading model...")
    model = load_model(args.model, device=args.device)
    
    # Step 2: Select prompt pairs
    print("\n[2/6] Preparing prompt pairs...")
    prompt_pairs = REFUSAL_PROMPT_PAIRS[:args.n_pairs]
    print(f"  Using {len(prompt_pairs)} prompt pairs")
    for pair in prompt_pairs[:3]:
        print(f"  - {pair.category}: {pair.refusal_prompt[:40]}...")
    
    # Step 3: Run circuit analysis
    print("\n[3/6] Running activation patching...")
    analyzer = CircuitAnalyzer(model)
    circuit = analyzer.analyze_multiple_pairs(
        prompt_pairs, 
        components=args.components,
        aggregate="mean"
    )
    
    print(f"\n  Found circuit:")
    print(f"    Critical layers: {circuit.critical_layers}")
    print(f"    Critical heads: {len(circuit.critical_heads)} heads")
    print(f"    Critical MLPs: {circuit.critical_mlps}")
    
    # Get top components
    top_components = circuit.top_k_components(10)
    print("\n  Top 10 components by importance:")
    for comp in top_components:
        print(f"    {comp.name}: {comp.importance_score:.4f}")
    
    # Step 4: Compute refusal direction
    print("\n[4/6] Computing refusal direction...")
    
    # Find best layer for refusal direction
    best_layer = circuit.critical_layers[0] if circuit.critical_layers else model.cfg.n_layers // 2
    
    refusal_direction = compute_refusal_direction(
        model,
        prompt_pairs,
        layer=best_layer,
        method="mean_diff"
    )
    
    print(f"  Best layer: {best_layer}")
    print(f"  Separation score: {refusal_direction.separation_score:.3f}")
    print(f"  Probe accuracy: {refusal_direction.probe_accuracy:.3f}")
    
    # Step 5: Validate with clamping
    print("\n[5/6] Running clamping validation...")
    
    # Test prompts for validation
    harmless_prompts = [
        "How do I bake a cake?",
        "What is the capital of France?",
        "Tell me a joke",
    ]
    harmful_prompts = [pair.refusal_prompt for pair in prompt_pairs[:3]]
    
    clamping_results = run_clamping_validation(
        model,
        refusal_direction,
        harmless_prompts,
        harmful_prompts,
        strength_values=[1.0, 2.0, 3.0],
    )
    
    # Step 6: Generate visualizations
    print("\n[6/6] Generating visualizations...")
    
    # Layer importance plot
    n_layers = model.cfg.n_layers
    layer_scores = []
    for layer in range(n_layers):
        layer_score = sum(
            abs(c.importance_score) for c in circuit.component_scores
            if c.layer == layer
        )
        layer_scores.append(layer_score)
    
    fig_layers = plot_layer_importance(
        layer_scores,
        title=f"Layer Importance for Refusal ({args.model})",
        save_path=str(figures_dir / "layer_importance.png")
    )
    
    # Head importance heatmap (if we analyzed attention)
    if args.components in ["attn", "all"]:
        n_heads = model.cfg.n_heads
        head_scores = np.zeros((n_layers, n_heads))
        
        for comp in circuit.component_scores:
            if comp.component_type == "attn_head" and comp.head is not None:
                head_scores[comp.layer, comp.head] = comp.importance_score
        
        fig_heads = plot_head_importance(
            head_scores,
            title=f"Attention Head Importance ({args.model})",
            save_path=str(figures_dir / "head_importance.png")
        )
    
    # Circuit diagram
    fig_circuit = create_circuit_diagram(
        circuit.critical_heads,
        circuit.critical_mlps,
        n_layers,
        model.cfg.n_heads,
        title=f"Refusal Circuit ({args.model})",
        save_path=str(figures_dir / "circuit_diagram.png")
    )
    
    # Save results
    results = {
        "model": args.model,
        "n_prompt_pairs": args.n_pairs,
        "components_analyzed": args.components,
        "critical_layers": circuit.critical_layers,
        "critical_heads": circuit.critical_heads,
        "critical_mlps": circuit.critical_mlps,
        "total_effect": circuit.total_effect,
        "refusal_direction": {
            "layer": refusal_direction.layer,
            "separation_score": refusal_direction.separation_score,
            "probe_accuracy": refusal_direction.probe_accuracy,
        },
        "top_10_components": [
            {"name": c.name, "score": c.importance_score}
            for c in top_components
        ],
        "clamping_validation": {
            "force_refusal_success_rate": sum(
                1 for r in clamping_results["force_refusal"] if r.behavior_changed
            ) / len(clamping_results["force_refusal"]),
            "suppress_refusal_success_rate": sum(
                1 for r in clamping_results["suppress_refusal"] if r.behavior_changed
            ) / len(clamping_results["suppress_refusal"]),
        },
    }
    
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"  - results.json: Quantitative results")
    print(f"  - figures/: Visualizations")
    print()
    print("Key findings:")
    print(f"  - Most important layers: {circuit.critical_layers[:5]}")
    print(f"  - Top attention heads: {circuit.critical_heads[:5]}")
    print(f"  - Refusal direction separation: {refusal_direction.separation_score:.2f}σ")
    
    return results


if __name__ == "__main__":
    main()
