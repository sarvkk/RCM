"""Activation steering and clamping for refusal control."""

from .clamping import (
    ClampingExperiment,
    ClampingResult,
    clamp_refusal_direction,
    force_refusal,
    suppress_refusal,
    run_clamping_validation,
)
from .steering import (
    SteeringVector,
    SteeringResult,
    steer_generation,
    compute_steering_effect,
)

__all__ = [
    "ClampingExperiment",
    "ClampingResult",
    "clamp_refusal_direction",
    "force_refusal",
    "suppress_refusal",
    "run_clamping_validation",
    "SteeringVector",
    "SteeringResult",
    "steer_generation",
    "compute_steering_effect",
]
