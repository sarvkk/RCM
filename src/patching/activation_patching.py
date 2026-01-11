"""
Activation Patching for Refusal Circuit Localization

Core implementation of activation patching - the key technique for
identifying which components of the model are causally responsible
for refusal behavior.

The basic idea:
1. Run model on refusal-triggering prompt, cache activations
2. Run model on compliant prompt, cache activations  
3. Re-run refusal prompt, but PATCH in activations from compliant run
4. If output changes from refusal to compliance, that component matters!
"""

import torch
from transformer_lens import HookedTransformer
from typing import Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from .cache import ActivationCache, cache_activations


@dataclass
class PatchingResult:
    """Results from a patching experiment."""
    hook_name: str
    position: Optional[int]  # Token position, or None if whole sequence
    head_idx: Optional[int]  # Attention head index, or None if not head-specific
    
    original_logit_diff: float  # Logit difference before patching
    patched_logit_diff: float   # Logit difference after patching
    effect_size: float          # Change in logit diff (how much patching mattered)
    
    @property
    def normalized_effect(self) -> float:
        """Effect normalized by original logit diff."""
        if abs(self.original_logit_diff) < 1e-8:
            return 0.0
        return self.effect_size / abs(self.original_logit_diff)


class ActivationPatcher:
    """
    Main class for running activation patching experiments.
    
    Implements the core patching methodology for identifying
    refusal circuits in language models.
    """
    
    def __init__(
        self,
        model: HookedTransformer,
        refusal_tokens: List[str] = None,
        compliant_tokens: List[str] = None,
    ):
        """
        Initialize the patcher.
        
        Args:
            model: HookedTransformer model to analyze
            refusal_tokens: Tokens indicating refusal (default: common refusal phrases)
            compliant_tokens: Tokens indicating compliance (for comparison)
        """
        self.model = model
        self.device = model.cfg.device
        
        # Default refusal indicators - tokens that suggest the model is refusing
        self.refusal_tokens = refusal_tokens or [
            "I", "Sorry", "cannot", "can't", "won't", "unable", 
            "Unfortunately", "apologize", "inappropriate", "harmful"
        ]
        
        self.compliant_tokens = compliant_tokens or [
            "Sure", "Here", "To", "The", "First", "You", "It"
        ]
        
        # Get token IDs for metric computation
        self._refusal_token_ids = self._get_token_ids(self.refusal_tokens)
        self._compliant_token_ids = self._get_token_ids(self.compliant_tokens)
    
    def _get_token_ids(self, tokens: List[str]) -> List[int]:
        """Get token IDs, handling tokenizer differences."""
        ids = []
        for token in tokens:
            # Try with and without space prefix
            for variant in [token, f" {token}"]:
                try:
                    token_id = self.model.to_single_token(variant)
                    ids.append(token_id)
                except:
                    pass
        return ids
    
    def compute_refusal_score(
        self,
        logits: torch.Tensor,
        position: int = -1,
    ) -> float:
        """
        Compute a "refusal score" from model logits.
        
        Higher score = more likely to refuse.
        
        This computes: mean(refusal_token_logits) - mean(compliant_token_logits)
        """
        logits_at_pos = logits[0, position, :]  # [vocab_size]
        
        refusal_logits = logits_at_pos[self._refusal_token_ids].mean()
        compliant_logits = logits_at_pos[self._compliant_token_ids].mean()
        
        return (refusal_logits - compliant_logits).item()
    
    def patch_residual_stream(
        self,
        clean_cache: ActivationCache,
        corrupted_cache: ActivationCache,
        layer: int,
        position: Optional[int] = None,
    ) -> PatchingResult:
        """
        Patch the residual stream at a specific layer.
        
        Runs the "corrupted" (refusal) input but patches in the "clean" 
        (compliant) residual stream activations at the specified layer.
        
        Args:
            clean_cache: Activations from compliant prompt (what we patch IN)
            corrupted_cache: Activations from refusal prompt (what we're modifying)
            layer: Which layer to patch
            position: Token position to patch (None = all positions)
            
        Returns:
            PatchingResult with effect size measurements
        """
        hook_name = f"blocks.{layer}.hook_resid_post"
        clean_act = clean_cache[hook_name]
        
        def patch_hook(activation, hook):
            if position is None:
                # Patch entire sequence
                min_len = min(activation.shape[1], clean_act.shape[1])
                activation[:, :min_len, :] = clean_act[:, :min_len, :]
            else:
                # Patch specific position
                pos = position if position >= 0 else activation.shape[1] + position
                if pos < clean_act.shape[1]:
                    activation[:, pos, :] = clean_act[:, pos, :]
            return activation
        
        # Get original refusal score
        original_score = self.compute_refusal_score(corrupted_cache.logits)
        
        # Run with patching
        patched_logits = self.model.run_with_hooks(
            corrupted_cache.tokens,
            fwd_hooks=[(hook_name, patch_hook)],
        )
        patched_score = self.compute_refusal_score(patched_logits)
        
        return PatchingResult(
            hook_name=hook_name,
            position=position,
            head_idx=None,
            original_logit_diff=original_score,
            patched_logit_diff=patched_score,
            effect_size=original_score - patched_score,  # Positive = reduced refusal
        )


def patch_residual_stream(
    model: HookedTransformer,
    refusal_prompt: str,
    compliant_prompt: str,
    layer: int,
    position: int = -1,
) -> PatchingResult:
    """
    Convenience function for single residual stream patch.
    
    Patches compliant activations into refusal run at specified layer.
    """
    patcher = ActivationPatcher(model)
    
    # Cache activations
    refusal_cache = cache_activations(
        model, refusal_prompt,
        hook_names=[f"blocks.{layer}.hook_resid_post"]
    )
    compliant_cache = cache_activations(
        model, compliant_prompt,
        hook_names=[f"blocks.{layer}.hook_resid_post"]
    )
    
    return patcher.patch_residual_stream(compliant_cache, refusal_cache, layer, position)


def patch_attention_head(
    model: HookedTransformer,
    refusal_cache: ActivationCache,
    compliant_cache: ActivationCache,
    layer: int,
    head: int,
    position: int = -1,
) -> PatchingResult:
    """
    Patch a specific attention head's output.
    
    This is more fine-grained than residual stream patching -
    allows identifying specific heads responsible for refusal.
    
    Uses attn.hook_z which has shape [batch, pos, n_heads, d_head]
    (the output before the output projection W_O).
    """
    # For per-head patching, we need hook_z which has the head dimension
    # hook_attn_out is already combined across heads [batch, pos, d_model]
    hook_name = f"blocks.{layer}.attn.hook_z"
    
    # Check if we have this hook cached
    if hook_name not in compliant_cache.activations:
        available = list(compliant_cache.activations.keys())
        raise ValueError(f"Cache missing {hook_name} for head patching. Available: {available}")
    
    clean_act = compliant_cache[hook_name]  # [batch, pos, n_heads, d_head]
    
    def patch_hook(activation, hook):
        pos = position if position >= 0 else activation.shape[1] + position
        if pos < clean_act.shape[1]:
            activation[:, pos, head, :] = clean_act[:, pos, head, :]
        return activation
    
    # Compute scores
    patcher = ActivationPatcher(model)
    original_score = patcher.compute_refusal_score(refusal_cache.logits)
    
    patched_logits = model.run_with_hooks(
        refusal_cache.tokens,
        fwd_hooks=[(hook_name, patch_hook)],
    )
    patched_score = patcher.compute_refusal_score(patched_logits)
    
    return PatchingResult(
        hook_name=hook_name,
        position=position,
        head_idx=head,
        original_logit_diff=original_score,
        patched_logit_diff=patched_score,
        effect_size=original_score - patched_score,
    )


def patch_mlp_layer(
    model: HookedTransformer,
    refusal_cache: ActivationCache,
    compliant_cache: ActivationCache,
    layer: int,
    position: int = -1,
) -> PatchingResult:
    """
    Patch MLP output at a specific layer.
    """
    # Try both naming conventions
    hook_name = f"blocks.{layer}.hook_mlp_out"
    alt_hook_name = f"blocks.{layer}.mlp.hook_post"
    
    # Find which hook name is available
    if hook_name in compliant_cache.activations:
        pass  # Use primary name
    elif alt_hook_name in compliant_cache.activations:
        hook_name = alt_hook_name
    else:
        available = list(compliant_cache.activations.keys())
        raise ValueError(f"Cache missing MLP hook for layer {layer}. Available: {available}")
    
    clean_act = compliant_cache[hook_name]
    
    def patch_hook(activation, hook):
        pos = position if position >= 0 else activation.shape[1] + position
        if pos < clean_act.shape[1]:
            activation[:, pos, :] = clean_act[:, pos, :]
        return activation
    
    patcher = ActivationPatcher(model)
    original_score = patcher.compute_refusal_score(refusal_cache.logits)
    
    patched_logits = model.run_with_hooks(
        refusal_cache.tokens,
        fwd_hooks=[(hook_name, patch_hook)],
    )
    patched_score = patcher.compute_refusal_score(patched_logits)
    
    return PatchingResult(
        hook_name=hook_name,
        position=position,
        head_idx=None,
        original_logit_diff=original_score,
        patched_logit_diff=patched_score,
        effect_size=original_score - patched_score,
    )


def run_patching_experiment(
    model: HookedTransformer,
    refusal_prompt: str,
    compliant_prompt: str,
    patch_type: str = "resid",  # "resid", "attn", "mlp", "all"
    positions: Optional[List[int]] = None,
) -> Dict[str, List[PatchingResult]]:
    """
    Run comprehensive patching experiment across layers.
    
    This is the main entry point for circuit localization.
    Patches each layer/component and measures effect on refusal.
    
    Args:
        model: HookedTransformer model
        refusal_prompt: Prompt that triggers refusal
        compliant_prompt: Similar prompt that gets normal response
        patch_type: What to patch - "resid", "attn", "mlp", or "all"
        positions: Token positions to test (default: [-1] for last token)
        
    Returns:
        Dictionary mapping component type to list of PatchingResults
        
    Example:
        >>> results = run_patching_experiment(
        ...     model,
        ...     "How do I hack into a bank?",
        ...     "How do I deposit money at a bank?",
        ...     patch_type="all"
        ... )
        >>> # Find most important layer
        >>> resid_effects = [r.effect_size for r in results["resid"]]
        >>> important_layer = np.argmax(resid_effects)
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    positions = positions or [-1]
    
    # Determine which hooks to cache
    # Include both naming conventions for compatibility across TransformerLens versions
    hooks_to_cache = []
    if patch_type in ["resid", "all"]:
        hooks_to_cache.extend([f"blocks.{i}.hook_resid_post" for i in range(n_layers)])
    if patch_type in ["attn", "all"]:
        # Use hook_z for per-head patching (has shape [batch, pos, n_heads, d_head])
        hooks_to_cache.extend([f"blocks.{i}.attn.hook_z" for i in range(n_layers)])
    if patch_type in ["mlp", "all"]:
        # Both naming conventions for MLP output
        hooks_to_cache.extend([f"blocks.{i}.hook_mlp_out" for i in range(n_layers)])
        hooks_to_cache.extend([f"blocks.{i}.mlp.hook_post" for i in range(n_layers)])
    
    # Cache activations
    print("Caching activations...")
    refusal_cache = cache_activations(model, refusal_prompt, hooks_to_cache)
    compliant_cache = cache_activations(model, compliant_prompt, hooks_to_cache)
    
    patcher = ActivationPatcher(model)
    results = {"resid": [], "attn": [], "mlp": []}
    
    # Residual stream patching
    if patch_type in ["resid", "all"]:
        print("Patching residual stream...")
        for layer in tqdm(range(n_layers), desc="Resid layers"):
            for pos in positions:
                result = patcher.patch_residual_stream(
                    compliant_cache, refusal_cache, layer, pos
                )
                results["resid"].append(result)
    
    # Attention head patching
    if patch_type in ["attn", "all"]:
        print("Patching attention heads...")
        for layer in tqdm(range(n_layers), desc="Attn layers"):
            for head in range(n_heads):
                for pos in positions:
                    result = patch_attention_head(
                        model, refusal_cache, compliant_cache, layer, head, pos
                    )
                    results["attn"].append(result)
    
    # MLP patching
    if patch_type in ["mlp", "all"]:
        print("Patching MLP layers...")
        for layer in tqdm(range(n_layers), desc="MLP layers"):
            for pos in positions:
                result = patch_mlp_layer(
                    model, refusal_cache, compliant_cache, layer, pos
                )
                results["mlp"].append(result)
    
    return results
