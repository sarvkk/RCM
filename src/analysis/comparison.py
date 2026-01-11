"""
Cross-Model Comparison for Refusal Circuit Analysis

Provides tools to compare refusal circuits across:
- Different model sizes
- Base vs instruction-tuned models
- Different prompt categories
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .statistics import (
    compute_cohens_d,
    significance_test,
    compute_confidence_interval,
    EffectSize,
    StatisticalResult,
)


class ComparisonDimension(Enum):
    """Dimensions along which to compare models."""
    MODEL_TYPE = "model_type"  # Base vs instruction-tuned
    MODEL_SIZE = "model_size"  # Parameter count
    PROMPT_CATEGORY = "prompt_category"  # Harm type


@dataclass
class MetricComparison:
    """Comparison of a single metric between two groups."""
    metric_name: str
    group1_name: str
    group2_name: str
    group1_values: List[float]
    group2_values: List[float]
    statistical_result: StatisticalResult
    
    @property
    def difference(self) -> float:
        """Mean difference (group2 - group1)."""
        return np.mean(self.group2_values) - np.mean(self.group1_values)
    
    @property
    def percent_change(self) -> float:
        """Percent change from group1 to group2."""
        g1_mean = np.mean(self.group1_values)
        if abs(g1_mean) < 1e-10:
            return 0.0
        return (np.mean(self.group2_values) - g1_mean) / abs(g1_mean) * 100


@dataclass
class ModelComparison:
    """
    Comprehensive comparison between two groups of models.
    
    Includes statistical tests for all key metrics.
    """
    comparison_name: str
    dimension: ComparisonDimension
    group1_models: List[str]
    group2_models: List[str]
    
    # Metric comparisons
    metric_comparisons: Dict[str, MetricComparison] = field(default_factory=dict)
    
    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_metric_comparison(
        self,
        metric_name: str,
        group1_values: List[float],
        group2_values: List[float],
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
    ):
        """Add a metric comparison to this model comparison."""
        stat_result = significance_test(group1_values, group2_values)
        
        comparison = MetricComparison(
            metric_name=metric_name,
            group1_name=group1_name,
            group2_name=group2_name,
            group1_values=group1_values,
            group2_values=group2_values,
            statistical_result=stat_result,
        )
        
        self.metric_comparisons[metric_name] = comparison
    
    def get_significant_differences(self, alpha: float = 0.05) -> List[str]:
        """Get list of metrics with significant differences."""
        return [
            name for name, comp in self.metric_comparisons.items()
            if comp.statistical_result.p_value < alpha
        ]
    
    def generate_summary(self):
        """Generate summary of all comparisons."""
        self.summary = {
            "n_metrics_compared": len(self.metric_comparisons),
            "n_significant": len(self.get_significant_differences()),
            "group1_models": self.group1_models,
            "group2_models": self.group2_models,
            "metrics": {},
        }
        
        for name, comp in self.metric_comparisons.items():
            self.summary["metrics"][name] = {
                "group1_mean": float(np.mean(comp.group1_values)),
                "group2_mean": float(np.mean(comp.group2_values)),
                "difference": comp.difference,
                "percent_change": comp.percent_change,
                "cohens_d": comp.statistical_result.effect_size.cohens_d if comp.statistical_result.effect_size else None,
                "p_value": comp.statistical_result.p_value,
                "is_significant": comp.statistical_result.is_significant,
            }
        
        return self.summary
    
    def __repr__(self):
        sig = len(self.get_significant_differences())
        total = len(self.metric_comparisons)
        return f"ModelComparison({self.comparison_name}, {sig}/{total} significant)"


def compare_models(
    model_results: Dict[str, Dict[str, Any]],
    group1_models: List[str],
    group2_models: List[str],
    metrics: List[str] = None,
    group1_name: str = "Group 1",
    group2_name: str = "Group 2",
    comparison_name: str = "Model Comparison",
) -> ModelComparison:
    """
    Compare two groups of models across multiple metrics.
    
    Args:
        model_results: Dict mapping model names to their result dicts
        group1_models: List of model names in first group
        group2_models: List of model names in second group
        metrics: List of metric keys to compare (default: common metrics)
        group1_name: Display name for group 1
        group2_name: Display name for group 2
        comparison_name: Name for this comparison
        
    Returns:
        ModelComparison with all statistical tests
    """
    # Default metrics to compare
    if metrics is None:
        metrics = [
            "separation_score",
            "probe_accuracy",
            "force_refusal_success_rate",
            "suppress_refusal_success_rate",
        ]
    
    comparison = ModelComparison(
        comparison_name=comparison_name,
        dimension=ComparisonDimension.MODEL_TYPE,
        group1_models=group1_models,
        group2_models=group2_models,
    )
    
    # Extract values for each metric
    for metric in metrics:
        group1_values = []
        group2_values = []
        
        for model in group1_models:
            if model in model_results and metric in model_results[model]:
                group1_values.append(model_results[model][metric])
        
        for model in group2_models:
            if model in model_results and metric in model_results[model]:
                group2_values.append(model_results[model][metric])
        
        # Only compare if both groups have data
        if group1_values and group2_values:
            comparison.add_metric_comparison(
                metric_name=metric,
                group1_values=group1_values,
                group2_values=group2_values,
                group1_name=group1_name,
                group2_name=group2_name,
            )
    
    comparison.generate_summary()
    return comparison


def compare_model_types(
    model_results: Dict[str, Dict[str, Any]],
    base_models: List[str],
    instruction_tuned_models: List[str],
) -> ModelComparison:
    """
    Compare base models vs instruction-tuned models.
    
    This is the key comparison for understanding how refusal emerges.
    
    Args:
        model_results: Dict mapping model names to their result dicts
        base_models: List of base model names
        instruction_tuned_models: List of instruction-tuned model names
        
    Returns:
        ModelComparison specifically for base vs instruction-tuned
    """
    return compare_models(
        model_results=model_results,
        group1_models=base_models,
        group2_models=instruction_tuned_models,
        group1_name="Base Models",
        group2_name="Instruction-Tuned Models",
        comparison_name="Base vs Instruction-Tuned",
    )


def compare_by_model_size(
    model_results: Dict[str, Dict[str, Any]],
    model_sizes: Dict[str, int],  # model_name -> parameter count
    size_threshold: int = 200_000_000,  # 200M params
) -> ModelComparison:
    """
    Compare small vs large models.
    
    Args:
        model_results: Dict mapping model names to their result dicts
        model_sizes: Dict mapping model names to parameter counts
        size_threshold: Parameter count threshold for small/large split
        
    Returns:
        ModelComparison for small vs large models
    """
    small_models = [m for m, size in model_sizes.items() if size < size_threshold]
    large_models = [m for m, size in model_sizes.items() if size >= size_threshold]
    
    return compare_models(
        model_results=model_results,
        group1_models=small_models,
        group2_models=large_models,
        group1_name=f"Small (<{size_threshold/1e6:.0f}M)",
        group2_name=f"Large (>={size_threshold/1e6:.0f}M)",
        comparison_name="Small vs Large Models",
    )


def generate_comparison_table(
    comparison: ModelComparison,
    format: str = "markdown",
) -> str:
    """
    Generate a formatted comparison table.
    
    Args:
        comparison: ModelComparison to format
        format: Output format ("markdown", "latex", "text")
        
    Returns:
        Formatted table string
    """
    if format == "markdown":
        return _generate_markdown_table(comparison)
    elif format == "latex":
        return _generate_latex_table(comparison)
    else:
        return _generate_text_table(comparison)


def _generate_markdown_table(comparison: ModelComparison) -> str:
    """Generate Markdown formatted comparison table."""
    lines = [
        f"## {comparison.comparison_name}",
        "",
        f"| Metric | {comparison.metric_comparisons[list(comparison.metric_comparisons.keys())[0]].group1_name} | {comparison.metric_comparisons[list(comparison.metric_comparisons.keys())[0]].group2_name} | Δ | Cohen's d | p-value | Sig. |" if comparison.metric_comparisons else "| Metric | Group 1 | Group 2 | Δ | Cohen's d | p-value | Sig. |",
        "|--------|---------|---------|---|----------|---------|------|",
    ]
    
    for name, comp in comparison.metric_comparisons.items():
        g1_mean = np.mean(comp.group1_values)
        g2_mean = np.mean(comp.group2_values)
        diff = comp.difference
        d = comp.statistical_result.effect_size.cohens_d if comp.statistical_result.effect_size else 0
        p = comp.statistical_result.p_value
        sig = "✓" if comp.statistical_result.is_significant else ""
        
        lines.append(
            f"| {name} | {g1_mean:.3f} | {g2_mean:.3f} | {diff:+.3f} | {d:.2f} | {p:.4f} | {sig} |"
        )
    
    return "\n".join(lines)


def _generate_latex_table(comparison: ModelComparison) -> str:
    """Generate LaTeX formatted comparison table."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\caption{{{comparison.comparison_name}}}",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
    ]
    
    if comparison.metric_comparisons:
        first_comp = comparison.metric_comparisons[list(comparison.metric_comparisons.keys())[0]]
        lines.append(
            f"Metric & {first_comp.group1_name} & {first_comp.group2_name} & $\\Delta$ & Cohen's $d$ & $p$-value & Sig. \\\\"
        )
    else:
        lines.append("Metric & Group 1 & Group 2 & $\\Delta$ & Cohen's $d$ & $p$-value & Sig. \\\\")
    
    lines.append("\\midrule")
    
    for name, comp in comparison.metric_comparisons.items():
        g1_mean = np.mean(comp.group1_values)
        g2_mean = np.mean(comp.group2_values)
        diff = comp.difference
        d = comp.statistical_result.effect_size.cohens_d if comp.statistical_result.effect_size else 0
        p = comp.statistical_result.p_value
        sig = "$\\checkmark$" if comp.statistical_result.is_significant else ""
        
        # Escape underscores for LaTeX
        name_latex = name.replace("_", "\\_")
        
        lines.append(
            f"{name_latex} & {g1_mean:.3f} & {g2_mean:.3f} & {diff:+.3f} & {d:.2f} & {p:.4f} & {sig} \\\\"
        )
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    
    return "\n".join(lines)


def _generate_text_table(comparison: ModelComparison) -> str:
    """Generate plain text formatted comparison table."""
    lines = [
        "=" * 80,
        comparison.comparison_name.center(80),
        "=" * 80,
        "",
    ]
    
    for name, comp in comparison.metric_comparisons.items():
        g1_mean = np.mean(comp.group1_values)
        g2_mean = np.mean(comp.group2_values)
        diff = comp.difference
        d = comp.statistical_result.effect_size.cohens_d if comp.statistical_result.effect_size else 0
        p = comp.statistical_result.p_value
        sig = "*" if comp.statistical_result.is_significant else ""
        
        lines.extend([
            f"{name}:",
            f"  {comp.group1_name}: {g1_mean:.3f}",
            f"  {comp.group2_name}: {g2_mean:.3f}",
            f"  Difference: {diff:+.3f}",
            f"  Cohen's d: {d:.2f}",
            f"  p-value: {p:.4f} {sig}",
            "",
        ])
    
    sig_count = len(comparison.get_significant_differences())
    total = len(comparison.metric_comparisons)
    lines.append(f"Significant differences: {sig_count}/{total}")
    
    return "\n".join(lines)


def rank_models_by_metric(
    model_results: Dict[str, Dict[str, Any]],
    metric: str,
    ascending: bool = False,
) -> List[Tuple[str, float]]:
    """
    Rank models by a specific metric.
    
    Args:
        model_results: Dict mapping model names to their result dicts
        metric: Metric key to rank by
        ascending: If True, lower is better
        
    Returns:
        List of (model_name, metric_value) tuples, sorted by metric
    """
    rankings = []
    
    for model, results in model_results.items():
        if metric in results:
            rankings.append((model, results[metric]))
    
    rankings.sort(key=lambda x: x[1], reverse=not ascending)
    
    return rankings


def identify_best_models(
    model_results: Dict[str, Dict[str, Any]],
    metrics: Dict[str, bool] = None,  # metric -> higher_is_better
) -> Dict[str, str]:
    """
    Identify the best model for each metric.
    
    Args:
        model_results: Dict mapping model names to their result dicts
        metrics: Dict mapping metric names to whether higher is better
        
    Returns:
        Dict mapping metric names to best model names
    """
    if metrics is None:
        metrics = {
            "separation_score": True,
            "probe_accuracy": True,
            "force_refusal_success_rate": True,
        }
    
    best_models = {}
    
    for metric, higher_is_better in metrics.items():
        rankings = rank_models_by_metric(
            model_results, 
            metric, 
            ascending=not higher_is_better
        )
        if rankings:
            best_models[metric] = rankings[0][0]
    
    return best_models
