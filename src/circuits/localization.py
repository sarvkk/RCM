"""
Circuit Localization for Refusal Behavior

Tools for identifying which components of a neural network are
causally responsible for refusal decisions.

The goal is to find a minimal "circuit" - a subset of model components
(attention heads, MLP layers, etc.) that are necessary and sufficient
for refusal behavior.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

from ..patching import (
    ActivationPatcher,
    PatchingResult,
    run_patching_experiment,
    cache_activations,
)
from ..data import RefusalPromptPair


@dataclass
class ComponentImportance:
    """Importance score for a model component."""
    component_type: str  # "resid", "attn_head", "mlp"
    layer: int
    head: Optional[int] = None  # Only for attention heads
    importance_score: float = 0.0
    
    @property
    def name(self) -> str:
        if self.component_type == "attn_head":
            return f"L{self.layer}H{self.head}"
        else:
            return f"L{self.layer}_{self.component_type}"


@dataclass 
class RefusalCircuit:
    """
    A identified circuit responsible for refusal behavior.
    
    Contains the minimal set of components that, when patched,
    eliminate or significantly reduce refusal behavior.
    """
    critical_layers: List[int]
    critical_heads: List[Tuple[int, int]]  # (layer, head) pairs
    critical_mlps: List[int]
    component_scores: List[ComponentImportance]
    total_effect: float
    prompt_pair: Optional[RefusalPromptPair] = None
    
    def __repr__(self):
        return (
            f"RefusalCircuit(\n"
            f"  critical_layers={self.critical_layers},\n"
            f"  critical_heads={self.critical_heads[:5]}...,\n"
            f"  total_effect={self.total_effect:.3f}\n"
            f")"
        )
    
    def top_k_components(self, k: int = 10) -> List[ComponentImportance]:
        """Get the k most important components."""
        sorted_components = sorted(
            self.component_scores, 
            key=lambda x: abs(x.importance_score), 
            reverse=True
        )
        return sorted_components[:k]


class CircuitAnalyzer:
    """
    Main class for analyzing refusal circuits.
    
    Performs systematic patching experiments to identify
    which model components are responsible for refusal.
    """
    
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.n_heads = model.cfg.n_heads
        self.patcher = ActivationPatcher(model)
        
    def analyze_prompt_pair(
        self,
        prompt_pair: RefusalPromptPair,
        components: str = "all",  # "resid", "attn", "mlp", "all"
        position: int = -1,
        verbose: bool = True,
    ) -> RefusalCircuit:
        """
        Analyze a single prompt pair to find the refusal circuit.
        
        Args:
            prompt_pair: Refusal/compliant prompt pair
            components: Which components to analyze
            position: Token position to analyze
            verbose: Print progress
            
        Returns:
            RefusalCircuit with identified critical components
        """
        if verbose:
            print(f"Analyzing prompt pair: {prompt_pair.category}")
            print(f"  Refusal: {prompt_pair.refusal_prompt[:50]}...")
            print(f"  Compliant: {prompt_pair.compliant_prompt[:50]}...")
        
        # Run patching experiment
        results = run_patching_experiment(
            self.model,
            prompt_pair.refusal_prompt,
            prompt_pair.compliant_prompt,
            patch_type=components,
            positions=[position],
        )
        
        # Convert results to ComponentImportance scores
        component_scores = []
        
        # Process residual stream results
        for i, result in enumerate(results.get("resid", [])):
            layer = i  # Assuming one result per layer
            component_scores.append(ComponentImportance(
                component_type="resid",
                layer=layer,
                importance_score=result.effect_size,
            ))
        
        # Process attention head results
        head_idx = 0
        for result in results.get("attn", []):
            layer = head_idx // self.n_heads
            head = head_idx % self.n_heads
            component_scores.append(ComponentImportance(
                component_type="attn_head",
                layer=layer,
                head=head,
                importance_score=result.effect_size,
            ))
            head_idx += 1
        
        # Process MLP results
        for i, result in enumerate(results.get("mlp", [])):
            layer = i
            component_scores.append(ComponentImportance(
                component_type="mlp",
                layer=layer,
                importance_score=result.effect_size,
            ))
        
        # Identify critical components (those with significant effect)
        threshold = 0.1  # Components with >10% effect are "critical"
        sorted_scores = sorted(component_scores, key=lambda x: abs(x.importance_score), reverse=True)
        
        critical_layers = []
        critical_heads = []
        critical_mlps = []
        
        for comp in sorted_scores:
            if abs(comp.importance_score) > threshold:
                if comp.component_type == "resid":
                    critical_layers.append(comp.layer)
                elif comp.component_type == "attn_head":
                    critical_heads.append((comp.layer, comp.head))
                elif comp.component_type == "mlp":
                    critical_mlps.append(comp.layer)
        
        total_effect = sum(abs(c.importance_score) for c in sorted_scores)
        
        circuit = RefusalCircuit(
            critical_layers=sorted(set(critical_layers)),
            critical_heads=critical_heads,
            critical_mlps=sorted(set(critical_mlps)),
            component_scores=component_scores,
            total_effect=total_effect,
            prompt_pair=prompt_pair,
        )
        
        if verbose:
            print(f"\nFound circuit with {len(circuit.critical_heads)} critical heads")
            print(f"Top 5 components:")
            for comp in circuit.top_k_components(5):
                print(f"  {comp.name}: {comp.importance_score:.4f}")
        
        return circuit
    
    def analyze_multiple_pairs(
        self,
        prompt_pairs: List[RefusalPromptPair],
        components: str = "all",
        aggregate: str = "mean",  # "mean", "max", "sum"
    ) -> RefusalCircuit:
        """
        Analyze multiple prompt pairs and aggregate results.
        
        This helps identify components that are consistently important
        across different types of refusal, not just prompt-specific.
        """
        all_circuits = []
        
        for pair in tqdm(prompt_pairs, desc="Analyzing pairs"):
            circuit = self.analyze_prompt_pair(pair, components, verbose=False)
            all_circuits.append(circuit)
        
        # Aggregate component scores across all pairs
        score_aggregator: Dict[str, List[float]] = {}
        
        for circuit in all_circuits:
            for comp in circuit.component_scores:
                key = comp.name
                if key not in score_aggregator:
                    score_aggregator[key] = []
                score_aggregator[key].append(comp.importance_score)
        
        # Compute aggregated scores
        aggregated_components = []
        for key, scores in score_aggregator.items():
            if aggregate == "mean":
                agg_score = np.mean(scores)
            elif aggregate == "max":
                agg_score = np.max(scores)
            else:  # sum
                agg_score = np.sum(scores)
            
            # Parse component type from key
            if "H" in key:  # Attention head
                parts = key.replace("L", "").split("H")
                layer, head = int(parts[0]), int(parts[1])
                aggregated_components.append(ComponentImportance(
                    component_type="attn_head",
                    layer=layer,
                    head=head,
                    importance_score=agg_score,
                ))
            else:
                parts = key.replace("L", "").split("_")
                layer = int(parts[0])
                comp_type = parts[1] if len(parts) > 1 else "resid"
                aggregated_components.append(ComponentImportance(
                    component_type=comp_type,
                    layer=layer,
                    importance_score=agg_score,
                ))
        
        # Build aggregated circuit
        threshold = 0.1
        sorted_scores = sorted(aggregated_components, key=lambda x: abs(x.importance_score), reverse=True)
        
        critical_layers = []
        critical_heads = []
        critical_mlps = []
        
        for comp in sorted_scores:
            if abs(comp.importance_score) > threshold:
                if comp.component_type == "resid":
                    critical_layers.append(comp.layer)
                elif comp.component_type == "attn_head":
                    critical_heads.append((comp.layer, comp.head))
                elif comp.component_type == "mlp":
                    critical_mlps.append(comp.layer)
        
        return RefusalCircuit(
            critical_layers=sorted(set(critical_layers)),
            critical_heads=critical_heads,
            critical_mlps=sorted(set(critical_mlps)),
            component_scores=aggregated_components,
            total_effect=sum(abs(c.importance_score) for c in aggregated_components),
        )


def find_refusal_circuit(
    model: HookedTransformer,
    prompt_pairs: List[RefusalPromptPair],
    components: str = "all",
) -> RefusalCircuit:
    """
    High-level function to find the refusal circuit.
    
    Convenience wrapper around CircuitAnalyzer.
    """
    analyzer = CircuitAnalyzer(model)
    return analyzer.analyze_multiple_pairs(prompt_pairs, components)


def rank_components_by_importance(
    circuit: RefusalCircuit,
    top_k: int = 20,
) -> List[ComponentImportance]:
    """
    Get ranked list of most important components for refusal.
    """
    return circuit.top_k_components(top_k)


def identify_critical_layers(
    circuit: RefusalCircuit,
    threshold: float = 0.15,
) -> List[int]:
    """
    Identify layers with the most refusal-relevant activity.
    
    Returns layers where total component importance exceeds threshold.
    """
    layer_importance = {}
    
    for comp in circuit.component_scores:
        layer = comp.layer
        if layer not in layer_importance:
            layer_importance[layer] = 0.0
        layer_importance[layer] += abs(comp.importance_score)
    
    critical = [
        layer for layer, importance in layer_importance.items()
        if importance > threshold
    ]
    
    return sorted(critical)
