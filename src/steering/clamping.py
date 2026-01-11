"""
Activation Clamping for Refusal Control

"Clamping" fixes activations at specific values during generation.
This is the key experimental technique to PROVE that we've found
the refusal circuit: if clamping these neurons changes behavior,
we've found a control mechanism.

Two main experiments:
1. Force Refusal: Clamp refusal neurons HIGH on harmless queries
   - If model refuses harmless queries, we found refusal control
   
2. Suppress Refusal: Clamp refusal neurons LOW on harmful queries
   - If model complies with harmful queries, we found refusal control
   
WARNING: These experiments are for RESEARCH to understand safety mechanisms.
They should inform how to build MORE robust refusal, not circumvent it.
"""

import torch
from transformer_lens import HookedTransformer
from typing import Optional, List, Callable, Dict, Tuple
from dataclasses import dataclass

from ..circuits.directions import RefusalDirection


@dataclass
class ClampingResult:
    """Results from a clamping experiment."""
    prompt: str
    original_output: str
    clamped_output: str
    clamp_value: float
    layer: int
    behavior_changed: bool  # Did clamping change refusal behavior?
    
    def __repr__(self):
        change_str = "CHANGED" if self.behavior_changed else "unchanged"
        return f"ClampingResult(behavior={change_str}, layer={self.layer})"


class ClampingExperiment:
    """
    Run clamping experiments to verify refusal circuit discovery.
    
    This class provides controlled experiments to validate that
    identified components actually control refusal behavior.
    """
    
    def __init__(
        self,
        model: HookedTransformer,
        refusal_direction: RefusalDirection,
    ):
        """
        Initialize clamping experiment.
        
        Args:
            model: The HookedTransformer model
            refusal_direction: Identified refusal direction from circuit analysis
        """
        self.model = model
        self.direction = refusal_direction
        self.device = model.cfg.device
        
        # Common refusal indicators
        self.refusal_phrases = [
            "I cannot", "I can't", "I won't", "I'm unable",
            "Sorry", "Apologies", "Unfortunately", 
            "not able to", "cannot help", "against my",
            "inappropriate", "harmful", "dangerous",
            "I don't think", "I shouldn't",
        ]
    
    def _detect_refusal(self, text: str) -> bool:
        """Check if text contains refusal indicators."""
        text_lower = text.lower()
        return any(phrase.lower() in text_lower for phrase in self.refusal_phrases)
    
    def _generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        hook_fn: Optional[Callable] = None,
        hook_name: Optional[str] = None,
    ) -> str:
        """Generate text with optional intervention hook."""
        tokens = self.model.to_tokens(prompt)
        
        if hook_fn is not None and hook_name is not None:
            with self.model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
                output_tokens = self.model.generate(
                    tokens,
                    max_new_tokens=max_tokens,
                    do_sample=False,  # Deterministic for reproducibility
                    temperature=0.0,
                )
        else:
            output_tokens = self.model.generate(
                tokens,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
            )
        
        # Decode only the new tokens
        output_text = self.model.to_string(output_tokens[0, tokens.shape[1]:])
        return output_text
    
    def clamp_refusal_direction(
        self,
        prompt: str,
        clamp_value: float,
        max_tokens: int = 50,
    ) -> ClampingResult:
        """
        Clamp the projection onto refusal direction to a fixed value.
        
        Args:
            prompt: Input prompt
            clamp_value: Value to clamp projection to
                - Positive: push toward refusal
                - Negative: push toward compliance
                - Zero: neutral
            max_tokens: Max tokens to generate
            
        Returns:
            ClampingResult with original and clamped outputs
        """
        layer = self.direction.layer
        hook_name = f"blocks.{layer}.hook_resid_post"
        direction = self.direction.direction.to(self.device)
        
        def clamp_hook(activation, hook):
            """Remove current projection and add clamped value."""
            # Get current projection onto refusal direction
            # activation shape: [batch, seq, d_model]
            
            # For each position, compute and modify projection
            for pos in range(activation.shape[1]):
                act = activation[0, pos, :]  # [d_model]
                current_proj = torch.dot(act, direction)
                
                # Remove current projection, add clamped value
                activation[0, pos, :] = (
                    act - current_proj * direction + clamp_value * direction
                )
            
            return activation
        
        # Get original output
        original_output = self._generate(prompt, max_tokens)
        
        # Get clamped output
        clamped_output = self._generate(
            prompt, max_tokens,
            hook_fn=clamp_hook,
            hook_name=hook_name,
        )
        
        # Detect behavior change
        original_refuses = self._detect_refusal(original_output)
        clamped_refuses = self._detect_refusal(clamped_output)
        behavior_changed = original_refuses != clamped_refuses
        
        return ClampingResult(
            prompt=prompt,
            original_output=original_output,
            clamped_output=clamped_output,
            clamp_value=clamp_value,
            layer=layer,
            behavior_changed=behavior_changed,
        )
    
    def force_refusal(
        self,
        harmless_prompt: str,
        strength: float = 2.0,
        max_tokens: int = 50,
    ) -> ClampingResult:
        """
        Attempt to force refusal on a harmless prompt.
        
        If successful (model refuses harmless query), this proves
        we've found the refusal control mechanism.
        
        Args:
            harmless_prompt: A benign prompt that normally gets answered
            strength: How strongly to push toward refusal
            max_tokens: Max generation length
        """
        return self.clamp_refusal_direction(
            harmless_prompt,
            clamp_value=strength,  # Positive = toward refusal
            max_tokens=max_tokens,
        )
    
    def suppress_refusal(
        self,
        harmful_prompt: str,
        strength: float = 2.0,
        max_tokens: int = 50,
    ) -> ClampingResult:
        """
        Attempt to suppress refusal on a harmful prompt.
        
        WARNING: This is for RESEARCH to understand safety mechanisms.
        If successful (model answers harmful query), this proves
        we've found the refusal control mechanism.
        
        Args:
            harmful_prompt: A prompt that normally triggers refusal
            strength: How strongly to push away from refusal
            max_tokens: Max generation length
        """
        return self.clamp_refusal_direction(
            harmful_prompt,
            clamp_value=-strength,  # Negative = away from refusal
            max_tokens=max_tokens,
        )


def clamp_refusal_direction(
    model: HookedTransformer,
    direction: RefusalDirection,
    prompt: str,
    clamp_value: float,
) -> ClampingResult:
    """Convenience function for single clamping experiment."""
    experiment = ClampingExperiment(model, direction)
    return experiment.clamp_refusal_direction(prompt, clamp_value)


def force_refusal(
    model: HookedTransformer,
    direction: RefusalDirection,
    harmless_prompt: str,
    strength: float = 2.0,
) -> ClampingResult:
    """Force model to refuse a harmless prompt."""
    experiment = ClampingExperiment(model, direction)
    return experiment.force_refusal(harmless_prompt, strength)


def suppress_refusal(
    model: HookedTransformer,
    direction: RefusalDirection,
    harmful_prompt: str,
    strength: float = 2.0,
) -> ClampingResult:
    """Attempt to suppress refusal on harmful prompt."""
    experiment = ClampingExperiment(model, direction)
    return experiment.suppress_refusal(harmful_prompt, strength)


def run_clamping_validation(
    model: HookedTransformer,
    direction: RefusalDirection,
    harmless_prompts: List[str],
    harmful_prompts: List[str],
    strength_values: List[float] = [0.5, 1.0, 2.0, 3.0],
) -> Dict[str, List[ClampingResult]]:
    """
    Run comprehensive clamping validation.
    
    Tests whether identified direction actually controls refusal
    by trying to force/suppress refusal at various strengths.
    
    Returns dict with "force_refusal" and "suppress_refusal" results.
    """
    experiment = ClampingExperiment(model, direction)
    results = {
        "force_refusal": [],
        "suppress_refusal": [],
    }
    
    # Test forcing refusal on harmless prompts
    for prompt in harmless_prompts:
        for strength in strength_values:
            result = experiment.force_refusal(prompt, strength)
            results["force_refusal"].append(result)
    
    # Test suppressing refusal on harmful prompts  
    for prompt in harmful_prompts:
        for strength in strength_values:
            result = experiment.suppress_refusal(prompt, strength)
            results["suppress_refusal"].append(result)
    
    # Summary statistics
    force_success = sum(1 for r in results["force_refusal"] if r.behavior_changed)
    suppress_success = sum(1 for r in results["suppress_refusal"] if r.behavior_changed)
    
    print(f"\n=== Clamping Validation Results ===")
    print(f"Force Refusal: {force_success}/{len(results['force_refusal'])} behavior changes")
    print(f"Suppress Refusal: {suppress_success}/{len(results['suppress_refusal'])} behavior changes")
    
    return results
