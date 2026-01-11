"""
Activation Caching for Refusal Circuit Analysis

Efficiently stores and manages activations from model forward passes
for later use in patching experiments.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ActivationCache:
    """
    Container for cached activations from a model run.
    
    Stores activations at various hook points for later patching.
    """
    prompt: str
    tokens: torch.Tensor
    activations: Dict[str, torch.Tensor] = field(default_factory=dict)
    logits: Optional[torch.Tensor] = None
    metadata: Dict = field(default_factory=dict)
    
    def get(self, hook_name: str) -> torch.Tensor:
        """Get cached activation by hook name."""
        if hook_name not in self.activations:
            raise KeyError(f"Hook '{hook_name}' not in cache. Available: {list(self.activations.keys())}")
        return self.activations[hook_name]
    
    def __getitem__(self, hook_name: str) -> torch.Tensor:
        return self.get(hook_name)
    
    def keys(self):
        return self.activations.keys()
    
    def items(self):
        return self.activations.items()
    
    @property 
    def n_tokens(self) -> int:
        return self.tokens.shape[-1]
    
    def to(self, device: str) -> "ActivationCache":
        """Move all tensors to specified device."""
        self.tokens = self.tokens.to(device)
        self.activations = {k: v.to(device) for k, v in self.activations.items()}
        if self.logits is not None:
            self.logits = self.logits.to(device)
        return self


def cache_activations(
    model: HookedTransformer,
    prompt: str,
    hook_names: Optional[List[str]] = None,
    include_logits: bool = True,
    metadata: Optional[Dict] = None,
) -> ActivationCache:
    """
    Run a forward pass and cache activations at specified hooks.
    
    Args:
        model: The HookedTransformer model
        prompt: Input text prompt
        hook_names: List of hook names to cache (None = cache all residual streams)
        include_logits: Whether to also cache the output logits
        metadata: Optional metadata to store with cache (e.g., {"type": "refusal"})
    
    Returns:
        ActivationCache containing all requested activations
        
    Example:
        >>> cache = cache_activations(model, "How do I hack a computer?")
        >>> resid_layer_5 = cache["blocks.5.hook_resid_post"]
    """
    # Tokenize
    tokens = model.to_tokens(prompt)
    
    # Default: cache residual stream at each layer
    if hook_names is None:
        hook_names = [f"blocks.{i}.hook_resid_post" for i in range(model.cfg.n_layers)]
    
    # Check if we need attention or MLP hooks (these have complex names)
    needs_attn = any("attn" in h for h in hook_names)
    needs_mlp = any("mlp" in h for h in hook_names)
    
    if needs_attn or needs_mlp:
        # For attention/MLP, cache everything and extract what we need
        # This is more reliable across TransformerLens versions
        logits, cache_dict = model.run_with_cache(tokens)
    else:
        # For residual stream only, we can filter
        hook_names_set = set(hook_names)
        logits, cache_dict = model.run_with_cache(
            tokens, 
            names_filter=lambda name: name in hook_names_set
        )
    
    # Build activation cache - grab all matching hooks from the cache
    activations = {}
    for name in hook_names:
        if name in cache_dict:
            activations[name] = cache_dict[name].detach().clone()
    
    return ActivationCache(
        prompt=prompt,
        tokens=tokens,
        activations=activations,
        logits=logits.detach().clone() if include_logits else None,
        metadata=metadata or {},
    )


def cache_paired_activations(
    model: HookedTransformer,
    refusal_prompt: str,
    compliant_prompt: str,
    hook_names: Optional[List[str]] = None,
) -> tuple[ActivationCache, ActivationCache]:
    """
    Cache activations for a refusal/compliant prompt pair.
    
    This is the core setup for activation patching experiments:
    we compare activations between a prompt that triggers refusal
    vs one that gets a normal response.
    
    Args:
        model: The HookedTransformer model
        refusal_prompt: Prompt expected to trigger refusal behavior
        compliant_prompt: Structurally similar prompt that gets normal response
        hook_names: Specific hooks to cache (default: all residual streams)
        
    Returns:
        Tuple of (refusal_cache, compliant_cache)
        
    Example:
        >>> refusal_cache, compliant_cache = cache_paired_activations(
        ...     model,
        ...     refusal_prompt="How do I make a bomb?",
        ...     compliant_prompt="How do I make a cake?",
        ... )
    """
    refusal_cache = cache_activations(
        model, refusal_prompt, hook_names,
        metadata={"type": "refusal", "pair_type": "harmful"}
    )
    
    compliant_cache = cache_activations(
        model, compliant_prompt, hook_names,
        metadata={"type": "compliant", "pair_type": "benign"}
    )
    
    return refusal_cache, compliant_cache


def compute_activation_diff(
    cache_a: ActivationCache,
    cache_b: ActivationCache,
    hook_name: str,
    position: int = -1,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute the difference in activations between two caches.
    
    Useful for identifying which directions in activation space
    correspond to refusal vs compliance.
    
    Args:
        cache_a: First activation cache (e.g., refusal)
        cache_b: Second activation cache (e.g., compliant)
        hook_name: Which hook point to compare
        position: Token position (-1 for last token)
        normalize: Whether to normalize the difference vector
        
    Returns:
        Difference vector (cache_a - cache_b) at specified position
    """
    act_a = cache_a[hook_name][0, position, :]  # [d_model]
    act_b = cache_b[hook_name][0, position, :]  # [d_model]
    
    diff = act_a - act_b
    
    if normalize:
        diff = diff / (diff.norm() + 1e-8)
    
    return diff
