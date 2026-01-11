"""
Model Loading Utilities for Mechanistic Interpretability

Provides standardized loading of models compatible with TransformerLens
for activation analysis and circuit mapping.

Includes both BASE models (no safety training) and INSTRUCTION-TUNED models
(with refusal behavior) for comparative analysis.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Optional, Literal, Dict, Any
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Type of model - affects expected refusal behavior."""
    BASE = "base"  # No safety training, unlikely to refuse
    INSTRUCTION_TUNED = "instruction_tuned"  # Has refusal behavior
    CHAT = "chat"  # Chat-optimized, strong refusal


@dataclass
class ModelConfig:
    """Configuration for a supported model."""
    layers: int
    heads: int
    d_model: int
    model_type: ModelType
    hf_name: str  # HuggingFace model name for TransformerLens
    memory_gb: float  # Approximate GPU memory needed (fp16)
    description: str = ""


# BASE MODELS - No safety training, for comparison
BASE_MODELS: Dict[str, ModelConfig] = {
    # GPT-2 family (good baseline, well-understood)
    "gpt2": ModelConfig(
        layers=12, heads=12, d_model=768, 
        model_type=ModelType.BASE,
        hf_name="gpt2",
        memory_gb=0.5,
        description="Original GPT-2 small, no safety training"
    ),
    "gpt2-medium": ModelConfig(
        layers=24, heads=16, d_model=1024,
        model_type=ModelType.BASE,
        hf_name="gpt2-medium",
        memory_gb=1.4,
        description="GPT-2 medium, no safety training"
    ),
    "gpt2-large": ModelConfig(
        layers=36, heads=20, d_model=1280,
        model_type=ModelType.BASE,
        hf_name="gpt2-large",
        memory_gb=3.0,
        description="GPT-2 large, no safety training"
    ),
    
    # Pythia family (EleutherAI - great for interpretability)
    "pythia-70m": ModelConfig(
        layers=6, heads=8, d_model=512,
        model_type=ModelType.BASE,
        hf_name="EleutherAI/pythia-70m",
        memory_gb=0.15,
        description="Smallest Pythia, good for quick experiments"
    ),
    "pythia-160m": ModelConfig(
        layers=12, heads=12, d_model=768,
        model_type=ModelType.BASE,
        hf_name="EleutherAI/pythia-160m",
        memory_gb=0.35,
        description="Small Pythia, good balance of speed and capacity"
    ),
    "pythia-410m": ModelConfig(
        layers=24, heads=16, d_model=1024,
        model_type=ModelType.BASE,
        hf_name="EleutherAI/pythia-410m",
        memory_gb=0.85,
        description="Medium Pythia, recommended for detailed analysis"
    ),
    "pythia-1b": ModelConfig(
        layers=16, heads=8, d_model=2048,
        model_type=ModelType.BASE,
        hf_name="EleutherAI/pythia-1b",
        memory_gb=2.0,
        description="Large Pythia, more sophisticated representations"
    ),
    
    # OPT family (Meta) - base models
    "opt-125m": ModelConfig(
        layers=12, heads=12, d_model=768,
        model_type=ModelType.BASE,
        hf_name="facebook/opt-125m",
        memory_gb=0.25,
        description="Small OPT base model"
    ),
}

# INSTRUCTION-TUNED MODELS - Have refusal behavior
INSTRUCTION_TUNED_MODELS: Dict[str, ModelConfig] = {
    # Pythia instruction-tuned variants
    "pythia-70m-deduped": ModelConfig(
        layers=6, heads=8, d_model=512,
        model_type=ModelType.BASE,  # Deduped is still base, but cleaner
        hf_name="EleutherAI/pythia-70m-deduped",
        memory_gb=0.15,
        description="Pythia trained on deduplicated data"
    ),
    
    # StableLM models (have some safety training)
    "stablelm-base-alpha-3b": ModelConfig(
        layers=16, heads=32, d_model=2560,
        model_type=ModelType.INSTRUCTION_TUNED,
        hf_name="stabilityai/stablelm-base-alpha-3b",
        memory_gb=6.0,
        description="StableLM with safety training (needs more VRAM)"
    ),
    
    # GPT-Neo-X instruction tuned
    "gpt-neox-20b": ModelConfig(
        layers=44, heads=64, d_model=6144,
        model_type=ModelType.BASE,
        hf_name="EleutherAI/gpt-neox-20b",
        memory_gb=40.0,
        description="Large model, needs significant resources"
    ),
    
    # Phi models (Microsoft - small but capable)
    "phi-1": ModelConfig(
        layers=24, heads=32, d_model=2048,
        model_type=ModelType.INSTRUCTION_TUNED,
        hf_name="microsoft/phi-1",
        memory_gb=2.5,
        description="Microsoft Phi-1, small but aligned"
    ),
    "phi-1_5": ModelConfig(
        layers=24, heads=32, d_model=2048,
        model_type=ModelType.INSTRUCTION_TUNED,
        hf_name="microsoft/phi-1_5",
        memory_gb=2.5,
        description="Microsoft Phi-1.5, improved alignment"
    ),
    
    # Qwen models (small instruction-tuned)
    "qwen-1_8b": ModelConfig(
        layers=24, heads=16, d_model=2048,
        model_type=ModelType.INSTRUCTION_TUNED,
        hf_name="Qwen/Qwen-1_8B",
        memory_gb=3.6,
        description="Qwen 1.8B with instruction tuning"
    ),
}

# Combined dictionary for backward compatibility
SUPPORTED_MODELS = {
    **{k: {"layers": v.layers, "heads": v.heads, "d_model": v.d_model} 
       for k, v in BASE_MODELS.items()},
    **{k: {"layers": v.layers, "heads": v.heads, "d_model": v.d_model} 
       for k, v in INSTRUCTION_TUNED_MODELS.items()},
}

# All model configs combined
ALL_MODELS: Dict[str, ModelConfig] = {**BASE_MODELS, **INSTRUCTION_TUNED_MODELS}


def get_model_config(model_name: str) -> ModelConfig:
    """Get full configuration for a supported model."""
    if model_name in ALL_MODELS:
        return ALL_MODELS[model_name]
    raise ValueError(
        f"Model '{model_name}' not in supported models. "
        f"Available: {list(ALL_MODELS.keys())}"
    )


def get_models_by_type(model_type: ModelType) -> Dict[str, ModelConfig]:
    """Get all models of a specific type."""
    return {k: v for k, v in ALL_MODELS.items() if v.model_type == model_type}


def get_models_by_memory(max_memory_gb: float) -> Dict[str, ModelConfig]:
    """Get all models that fit within a memory budget."""
    return {k: v for k, v in ALL_MODELS.items() if v.memory_gb <= max_memory_gb}


def list_available_models(verbose: bool = False) -> None:
    """Print all available models with their configurations."""
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS FOR REFUSAL CIRCUIT ANALYSIS")
    print("=" * 70)
    
    print("\n📦 BASE MODELS (no safety training - for comparison):")
    print("-" * 50)
    for name, config in BASE_MODELS.items():
        mem_str = f"{config.memory_gb:.1f}GB"
        print(f"  {name:20s} | L{config.layers:2d} H{config.heads:2d} D{config.d_model:4d} | {mem_str:>6s}")
        if verbose and config.description:
            print(f"    └─ {config.description}")
    
    print("\n🛡️ INSTRUCTION-TUNED MODELS (have refusal behavior):")
    print("-" * 50)
    for name, config in INSTRUCTION_TUNED_MODELS.items():
        mem_str = f"{config.memory_gb:.1f}GB"
        print(f"  {name:20s} | L{config.layers:2d} H{config.heads:2d} D{config.d_model:4d} | {mem_str:>6s}")
        if verbose and config.description:
            print(f"    └─ {config.description}")
    print()


def load_model(
    model_name: str = "pythia-410m",
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    center_unembed: bool = True,
    center_writing_weights: bool = True,
    fold_ln: bool = True,
    refactor_factored_attn_matrices: bool = False,
) -> HookedTransformer:
    """
    Load a model with TransformerLens hooks for interpretability analysis.
    
    Args:
        model_name: Name of the model to load (see ALL_MODELS)
        device: Device to load model on ('cuda', 'cpu', or None for auto)
        dtype: Data type for model weights (default: float16 on CUDA, float32 on CPU)
        center_unembed: Whether to center the unembedding matrix
        center_writing_weights: Whether to center writing weights
        fold_ln: Whether to fold layer norms into weights
        refactor_factored_attn_matrices: Whether to refactor attention matrices
    
    Returns:
        HookedTransformer model ready for activation analysis
    
    Example:
        >>> model = load_model("pythia-410m")
        >>> print(f"Loaded {model.cfg.model_name} with {model.cfg.n_layers} layers")
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use float16 on CUDA to save memory, float32 on CPU
    if dtype is None:
        dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Get model config
    config = get_model_config(model_name) if model_name in ALL_MODELS else None
    
    # Determine HuggingFace model name
    if config:
        tl_model_name = config.hf_name
        model_type_str = f" [{config.model_type.value}]"
    else:
        # Fallback for custom models
        tl_model_name = model_name
        if model_name.startswith("pythia-"):
            tl_model_name = f"EleutherAI/{model_name}"
        model_type_str = ""
    
    print(f"Loading {model_name}{model_type_str} on {device} ({dtype})...")
    
    model = HookedTransformer.from_pretrained(
        tl_model_name,
        device=device,
        dtype=dtype,
        center_unembed=center_unembed,
        center_writing_weights=center_writing_weights,
        fold_ln=fold_ln,
        refactor_factored_attn_matrices=refactor_factored_attn_matrices,
    )
    
    print(f"✓ Loaded {model.cfg.model_name}")
    print(f"  Layers: {model.cfg.n_layers}, Heads: {model.cfg.n_heads}, d_model: {model.cfg.d_model}")
    print(f"  Vocab size: {model.cfg.d_vocab}, Context: {model.cfg.n_ctx}")
    if config:
        print(f"  Type: {config.model_type.value}, Memory: ~{config.memory_gb:.1f}GB")
    
    return model


def get_activation_names(model: HookedTransformer) -> dict:
    """
    Get standard activation point names for a model.
    
    Returns dict with keys for common intervention points:
    - 'resid_pre': Residual stream before each layer
    - 'resid_post': Residual stream after each layer  
    - 'attn_out': Attention output for each layer
    - 'mlp_out': MLP output for each layer
    - 'pattern': Attention patterns for each layer/head
    """
    n_layers = model.cfg.n_layers
    
    return {
        "resid_pre": [f"blocks.{i}.hook_resid_pre" for i in range(n_layers)],
        "resid_post": [f"blocks.{i}.hook_resid_post" for i in range(n_layers)],
        "attn_out": [f"blocks.{i}.attn.hook_result" for i in range(n_layers)],
        "mlp_out": [f"blocks.{i}.mlp.hook_post" for i in range(n_layers)],
        "pattern": [f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)],
        "q": [f"blocks.{i}.attn.hook_q" for i in range(n_layers)],
        "k": [f"blocks.{i}.attn.hook_k" for i in range(n_layers)],
        "v": [f"blocks.{i}.attn.hook_v" for i in range(n_layers)],
        "z": [f"blocks.{i}.attn.hook_z" for i in range(n_layers)],
    }
