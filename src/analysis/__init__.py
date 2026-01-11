"""Statistical analysis and comparison tools for refusal circuit research."""

from .statistics import (
    compute_cohens_d,
    compute_confidence_interval,
    significance_test,
    aggregate_results,
    EffectSize,
    StatisticalResult,
)
from .comparison import (
    ModelComparison,
    compare_models,
    compare_model_types,
    generate_comparison_table,
)

__all__ = [
    # Statistics
    "compute_cohens_d",
    "compute_confidence_interval",
    "significance_test",
    "aggregate_results",
    "EffectSize",
    "StatisticalResult",
    # Comparison
    "ModelComparison",
    "compare_models",
    "compare_model_types",
    "generate_comparison_table",
]
