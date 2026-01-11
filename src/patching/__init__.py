"""Activation patching utilities for causal interventions."""

from .activation_patching import (
    ActivationPatcher,
    PatchingResult,
    patch_residual_stream,
    patch_attention_head,
    patch_mlp_layer,
    run_patching_experiment,
)
from .cache import ActivationCache, cache_activations

__all__ = [
    "ActivationPatcher",
    "PatchingResult",
    "patch_residual_stream", 
    "patch_attention_head",
    "patch_mlp_layer",
    "run_patching_experiment",
    "ActivationCache",
    "cache_activations",
]
