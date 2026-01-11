"""
Activation Steering for Controlled Generation

Steering adds a direction vector to activations during generation,
allowing continuous control over model behavior without clamping.

Unlike clamping (which fixes values), steering adds/subtracts from
the natural activations, providing smoother control.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np

from ..circuits.directions import RefusalDirection
from ..patching import cache_activations


@dataclass
class SteeringVector:
    """
    A vector for steering model behavior.
    
    Can be derived from refusal direction or computed directly
    from activation differences.
    """
    vector: torch.Tensor  # [d_model]
    layer: int
    name: str = "steering"
    
    @classmethod
    def from_refusal_direction(
        cls,
        direction: RefusalDirection,
        name: str = "refusal",
    ) -> "SteeringVector":
        """Create steering vector from refusal direction."""
        return cls(
            vector=direction.direction.clone(),
            layer=direction.layer,
            name=name,
        )
    
    @classmethod
    def from_activation_diff(
        cls,
        model: HookedTransformer,
        positive_prompt: str,
        negative_prompt: str,
        layer: int,
        position: int = -1,
        name: str = "custom",
    ) -> "SteeringVector":
        """
        Create steering vector from activation difference.
        
        The vector points FROM negative_prompt activations
        TO positive_prompt activations.
        """
        hook_name = f"blocks.{layer}.hook_resid_post"
        
        pos_cache = cache_activations(model, positive_prompt, [hook_name])
        neg_cache = cache_activations(model, negative_prompt, [hook_name])
        
        pos_act = pos_cache[hook_name][0, position, :]
        neg_act = neg_cache[hook_name][0, position, :]
        
        vector = pos_act - neg_act
        vector = vector / (vector.norm() + 1e-8)
        
        return cls(vector=vector, layer=layer, name=name)
    
    def to(self, device: str) -> "SteeringVector":
        self.vector = self.vector.to(device)
        return self


@dataclass
class SteeringResult:
    """Results from a steering experiment."""
    prompt: str
    original_output: str
    steered_output: str
    coefficient: float
    layer: int
    steering_name: str


def steer_generation(
    model: HookedTransformer,
    prompt: str,
    steering_vector: SteeringVector,
    coefficient: float = 1.0,
    max_tokens: int = 50,
    layers: Optional[List[int]] = None,
    positions: str = "all",  # "all", "last", "first"
) -> SteeringResult:
    """
    Generate text with activation steering.
    
    Args:
        model: HookedTransformer model
        prompt: Input prompt
        steering_vector: Vector to add during generation
        coefficient: Scaling factor for the steering vector
            - Positive: steer in vector direction
            - Negative: steer opposite to vector
            - Zero: no steering
        max_tokens: Maximum tokens to generate
        layers: Which layers to apply steering (default: just vector's layer)
        positions: Which token positions to steer
        
    Returns:
        SteeringResult with original and steered outputs
    """
    device = model.cfg.device
    vector = steering_vector.vector.to(device)
    
    if layers is None:
        layers = [steering_vector.layer]
    
    def steering_hook(activation, hook):
        """Add steering vector to activations."""
        # activation: [batch, seq, d_model]
        
        if positions == "all":
            activation = activation + coefficient * vector
        elif positions == "last":
            activation[:, -1, :] = activation[:, -1, :] + coefficient * vector
        elif positions == "first":
            activation[:, 0, :] = activation[:, 0, :] + coefficient * vector
        
        return activation
    
    # Build hooks for all specified layers
    hooks = [
        (f"blocks.{layer}.hook_resid_post", steering_hook)
        for layer in layers
    ]
    
    # Generate original
    tokens = model.to_tokens(prompt)
    original_tokens = model.generate(
        tokens,
        max_new_tokens=max_tokens,
        do_sample=False,
    )
    original_output = model.to_string(original_tokens[0, tokens.shape[1]:])
    
    # Generate with steering
    with model.hooks(fwd_hooks=hooks):
        steered_tokens = model.generate(
            tokens,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    steered_output = model.to_string(steered_tokens[0, tokens.shape[1]:])
    
    return SteeringResult(
        prompt=prompt,
        original_output=original_output,
        steered_output=steered_output,
        coefficient=coefficient,
        layer=steering_vector.layer,
        steering_name=steering_vector.name,
    )


def compute_steering_effect(
    model: HookedTransformer,
    prompts: List[str],
    steering_vector: SteeringVector,
    coefficients: List[float] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
    max_tokens: int = 30,
) -> Dict[str, List[SteeringResult]]:
    """
    Systematically measure steering effect across coefficients.
    
    Returns dict mapping coefficient to list of results.
    """
    results = {str(c): [] for c in coefficients}
    
    for prompt in prompts:
        for coeff in coefficients:
            result = steer_generation(
                model, prompt, steering_vector, coeff, max_tokens
            )
            results[str(coeff)].append(result)
    
    return results


def find_optimal_steering_coefficient(
    model: HookedTransformer,
    prompt: str,
    steering_vector: SteeringVector,
    target_contains: Optional[str] = None,
    target_avoids: Optional[str] = None,
    search_range: Tuple[float, float] = (-3.0, 3.0),
    n_steps: int = 20,
    max_tokens: int = 30,
) -> Tuple[float, SteeringResult]:
    """
    Find the steering coefficient that achieves desired behavior.
    
    Args:
        model: HookedTransformer model
        prompt: Input prompt
        steering_vector: Steering vector to use
        target_contains: Output should contain this string
        target_avoids: Output should NOT contain this string
        search_range: Range of coefficients to search
        n_steps: Number of steps in search
        max_tokens: Max generation length
        
    Returns:
        (best_coefficient, best_result)
    """
    coefficients = np.linspace(search_range[0], search_range[1], n_steps)
    
    best_coeff = 0.0
    best_result = None
    
    for coeff in coefficients:
        result = steer_generation(
            model, prompt, steering_vector, coeff, max_tokens
        )
        
        # Check if this achieves target
        output_lower = result.steered_output.lower()
        
        meets_contains = (
            target_contains is None or 
            target_contains.lower() in output_lower
        )
        meets_avoids = (
            target_avoids is None or
            target_avoids.lower() not in output_lower
        )
        
        if meets_contains and meets_avoids:
            # Found a good coefficient - prefer smaller absolute value
            if best_result is None or abs(coeff) < abs(best_coeff):
                best_coeff = coeff
                best_result = result
    
    if best_result is None:
        # Return coefficient=0 result as fallback
        best_result = steer_generation(
            model, prompt, steering_vector, 0.0, max_tokens
        )
    
    return best_coeff, best_result


def create_refusal_steering_vector(
    model: HookedTransformer,
    refusal_direction: RefusalDirection,
) -> SteeringVector:
    """
    Create a steering vector from a refusal direction.
    
    Convenience function that wraps the RefusalDirection
    into a SteeringVector for use with steering functions.
    """
    return SteeringVector.from_refusal_direction(
        refusal_direction,
        name="refusal_steering",
    )
