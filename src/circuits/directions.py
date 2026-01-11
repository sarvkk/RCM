"""
Refusal Direction Analysis

Identifies the "refusal direction" in activation space - the vector
that distinguishes refusal activations from compliant activations.

This direction can be used for:
1. Probing: Project activations to predict refusal probability
2. Steering: Add/subtract the direction to control refusal behavior
3. Understanding: Analyze what the direction represents
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from ..patching import cache_activations, ActivationCache
from ..data import RefusalPromptPair


@dataclass
class RefusalDirection:
    """
    A direction in activation space associated with refusal.
    
    The direction is computed as the difference between mean
    refusal activations and mean compliant activations.
    """
    direction: torch.Tensor  # [d_model] normalized direction vector
    layer: int
    position: int  # Token position (-1 = last)
    
    # Statistics about the direction
    refusal_mean: torch.Tensor
    compliant_mean: torch.Tensor
    separation_score: float  # How well it separates refusal vs compliant
    
    # Optional: trained probe
    probe: Optional[LogisticRegression] = None
    probe_accuracy: Optional[float] = None
    
    def project(self, activation: torch.Tensor) -> torch.Tensor:
        """
        Project an activation onto this refusal direction.
        
        Returns a scalar - higher means more "refusal-like".
        """
        if activation.dim() == 3:  # [batch, seq, d_model]
            activation = activation[0, self.position, :]
        elif activation.dim() == 2:  # [seq, d_model]
            activation = activation[self.position, :]
        
        return torch.dot(activation, self.direction)
    
    def to(self, device: str) -> "RefusalDirection":
        """Move tensors to device."""
        self.direction = self.direction.to(device)
        self.refusal_mean = self.refusal_mean.to(device)
        self.compliant_mean = self.compliant_mean.to(device)
        return self


def compute_refusal_direction(
    model: HookedTransformer,
    prompt_pairs: List[RefusalPromptPair],
    layer: int,
    position: int = -1,
    method: str = "mean_diff",  # "mean_diff", "pca", "probe"
) -> RefusalDirection:
    """
    Compute the refusal direction at a specific layer.
    
    Args:
        model: HookedTransformer model
        prompt_pairs: List of refusal/compliant prompt pairs
        layer: Which layer to extract activations from
        position: Token position (-1 for last token)
        method: How to compute direction
            - "mean_diff": Difference of means (simple, interpretable)
            - "pca": First PC of the difference
            - "probe": Train a linear probe, use weights as direction
            
    Returns:
        RefusalDirection object
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    
    # Collect activations for all pairs
    refusal_activations = []
    compliant_activations = []
    
    for pair in prompt_pairs:
        # Get refusal activations
        ref_cache = cache_activations(
            model, pair.refusal_prompt, [hook_name], include_logits=False
        )
        ref_act = ref_cache[hook_name][0, position, :].detach().cpu()
        refusal_activations.append(ref_act)
        
        # Get compliant activations
        comp_cache = cache_activations(
            model, pair.compliant_prompt, [hook_name], include_logits=False
        )
        comp_act = comp_cache[hook_name][0, position, :].detach().cpu()
        compliant_activations.append(comp_act)
    
    # Stack into tensors
    refusal_acts = torch.stack(refusal_activations)  # [n_pairs, d_model]
    compliant_acts = torch.stack(compliant_activations)
    
    # Compute means
    refusal_mean = refusal_acts.mean(dim=0)
    compliant_mean = compliant_acts.mean(dim=0)
    
    # Compute direction based on method
    if method == "mean_diff":
        direction = refusal_mean - compliant_mean
        direction = direction / (direction.norm() + 1e-8)
        
    elif method == "pca":
        # PCA on the differences
        diffs = refusal_acts - compliant_acts
        pca = PCA(n_components=1)
        pca.fit(diffs.numpy())
        direction = torch.tensor(pca.components_[0], dtype=torch.float32)
        direction = direction / (direction.norm() + 1e-8)
        
    elif method == "probe":
        # Train a linear probe
        X = torch.cat([refusal_acts, compliant_acts], dim=0).numpy()
        y = np.array([1] * len(refusal_acts) + [0] * len(compliant_acts))
        
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)
        
        direction = torch.tensor(probe.coef_[0], dtype=torch.float32)
        direction = direction / (direction.norm() + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute separation score
    refusal_projections = (refusal_acts @ direction).numpy()
    compliant_projections = (compliant_acts @ direction).numpy()
    
    # Separation = difference in means / pooled std
    pooled_std = np.sqrt(
        (refusal_projections.var() + compliant_projections.var()) / 2
    )
    separation_score = (
        refusal_projections.mean() - compliant_projections.mean()
    ) / (pooled_std + 1e-8)
    
    # Optionally compute probe accuracy
    probe = None
    probe_accuracy = None
    if method != "probe":
        # Train probe for evaluation
        X = torch.cat([refusal_acts, compliant_acts], dim=0).numpy()
        y = np.array([1] * len(refusal_acts) + [0] * len(compliant_acts))
        probe = LogisticRegression(max_iter=1000)
        probe.fit(X, y)
        probe_accuracy = probe.score(X, y)
    else:
        probe_accuracy = probe.score(X, y)
    
    return RefusalDirection(
        direction=direction,
        layer=layer,
        position=position,
        refusal_mean=refusal_mean,
        compliant_mean=compliant_mean,
        separation_score=float(separation_score),
        probe=probe,
        probe_accuracy=probe_accuracy,
    )


def compute_refusal_directions_all_layers(
    model: HookedTransformer,
    prompt_pairs: List[RefusalPromptPair],
    position: int = -1,
    method: str = "mean_diff",
) -> List[RefusalDirection]:
    """
    Compute refusal direction at every layer.
    
    Returns list of RefusalDirection, one per layer.
    Useful for finding which layer has the clearest refusal signal.
    """
    directions = []
    
    for layer in range(model.cfg.n_layers):
        direction = compute_refusal_direction(
            model, prompt_pairs, layer, position, method
        )
        directions.append(direction)
    
    return directions


def find_best_layer_for_refusal_direction(
    model: HookedTransformer,
    prompt_pairs: List[RefusalPromptPair],
    metric: str = "separation",  # "separation" or "probe_accuracy"
) -> Tuple[int, RefusalDirection]:
    """
    Find the layer with the clearest refusal direction.
    
    Args:
        model: HookedTransformer model
        prompt_pairs: Prompt pairs for analysis
        metric: Which metric to optimize
        
    Returns:
        (best_layer, best_direction)
    """
    directions = compute_refusal_directions_all_layers(model, prompt_pairs)
    
    if metric == "separation":
        scores = [d.separation_score for d in directions]
    else:  # probe_accuracy
        scores = [d.probe_accuracy or 0.0 for d in directions]
    
    best_idx = np.argmax(scores)
    return best_idx, directions[best_idx]


def project_onto_refusal_direction(
    activation: torch.Tensor,
    direction: RefusalDirection,
) -> float:
    """
    Project an activation onto a refusal direction.
    
    Higher values = more refusal-like.
    """
    return direction.project(activation).item()


def get_refusal_probability(
    model: HookedTransformer,
    prompt: str,
    direction: RefusalDirection,
) -> float:
    """
    Get probability that a prompt will trigger refusal.
    
    Uses the trained probe from the RefusalDirection.
    """
    hook_name = f"blocks.{direction.layer}.hook_resid_post"
    cache = cache_activations(model, prompt, [hook_name], include_logits=False)
    
    activation = cache[hook_name][0, direction.position, :].detach().cpu().numpy()
    
    if direction.probe is not None:
        prob = direction.probe.predict_proba([activation])[0, 1]
        return float(prob)
    else:
        # Fall back to projection-based estimate
        proj = direction.project(torch.tensor(activation))
        # Sigmoid to convert to probability
        return float(torch.sigmoid(proj).item())
