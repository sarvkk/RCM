"""
Statistical Analysis for Refusal Circuit Research

Provides rigorous statistical analysis including:
- Effect size metrics (Cohen's d, Glass's delta)
- Confidence intervals
- Statistical significance tests
- Aggregation across multiple experiments
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum


class EffectSizeInterpretation(Enum):
    """Standard interpretation of Cohen's d effect size."""
    NEGLIGIBLE = "negligible"  # |d| < 0.2
    SMALL = "small"            # 0.2 <= |d| < 0.5
    MEDIUM = "medium"          # 0.5 <= |d| < 0.8
    LARGE = "large"            # |d| >= 0.8


@dataclass
class EffectSize:
    """
    Effect size measurement with interpretation.
    
    Cohen's d is the standardized difference between two means:
    d = (M1 - M2) / pooled_std
    """
    cohens_d: float
    interpretation: EffectSizeInterpretation
    confidence_interval: Tuple[float, float]
    
    @classmethod
    def from_groups(
        cls,
        group1: np.ndarray,
        group2: np.ndarray,
        confidence: float = 0.95,
    ) -> "EffectSize":
        """Compute effect size from two groups of observations."""
        d = compute_cohens_d(group1, group2)
        interp = interpret_cohens_d(d)
        ci = cohens_d_confidence_interval(d, len(group1), len(group2), confidence)
        return cls(cohens_d=d, interpretation=interp, confidence_interval=ci)
    
    def __repr__(self):
        return f"EffectSize(d={self.cohens_d:.3f}, {self.interpretation.value})"


@dataclass
class StatisticalResult:
    """
    Result of a statistical significance test.
    """
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool  # At alpha=0.05
    effect_size: Optional[EffectSize] = None
    
    # Additional context
    sample_sizes: Optional[Tuple[int, int]] = None
    means: Optional[Tuple[float, float]] = None
    stds: Optional[Tuple[float, float]] = None
    
    def __repr__(self):
        sig_str = "significant" if self.is_significant else "not significant"
        return f"StatisticalResult({self.test_name}, p={self.p_value:.4f}, {sig_str})"


def compute_cohens_d(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
) -> float:
    """
    Compute Cohen's d effect size between two groups.
    
    Cohen's d = (M1 - M2) / pooled_std
    
    Args:
        group1: First group of observations
        group2: Second group of observations
        
    Returns:
        Cohen's d value (positive if group1 > group2)
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    
    if n1 < 2 or n2 < 2:
        return 0.0
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-10:
        return 0.0
    
    return (mean1 - mean2) / pooled_std


def interpret_cohens_d(d: float) -> EffectSizeInterpretation:
    """
    Interpret Cohen's d using standard thresholds.
    
    - |d| < 0.2: Negligible
    - 0.2 <= |d| < 0.5: Small
    - 0.5 <= |d| < 0.8: Medium
    - |d| >= 0.8: Large
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return EffectSizeInterpretation.NEGLIGIBLE
    elif abs_d < 0.5:
        return EffectSizeInterpretation.SMALL
    elif abs_d < 0.8:
        return EffectSizeInterpretation.MEDIUM
    else:
        return EffectSizeInterpretation.LARGE


def cohens_d_confidence_interval(
    d: float,
    n1: int,
    n2: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute confidence interval for Cohen's d.
    
    Uses the non-central t-distribution approximation.
    """
    # Variance of d
    se = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
    
    # Critical value
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)
    
    return (d - z * se, d + z * se)


def compute_confidence_interval(
    data: Union[List[float], np.ndarray],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute confidence interval for the mean of a sample.
    
    Args:
        data: Sample observations
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        (lower_bound, upper_bound) tuple
    """
    data = np.asarray(data)
    n = len(data)
    
    if n < 2:
        mean = np.mean(data)
        return (mean, mean)
    
    mean = np.mean(data)
    se = stats.sem(data)
    
    # t-distribution critical value
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    
    margin = t_crit * se
    return (mean - margin, mean + margin)


def significance_test(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    test: str = "auto",
    alpha: float = 0.05,
) -> StatisticalResult:
    """
    Perform statistical significance test between two groups.
    
    Args:
        group1: First group of observations
        group2: Second group of observations
        test: Test to use - "t-test", "mann-whitney", or "auto" (chooses based on data)
        alpha: Significance level
        
    Returns:
        StatisticalResult with test outcomes
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    n1, n2 = len(group1), len(group2)
    
    # Choose test
    if test == "auto":
        # Use Mann-Whitney if samples are small or non-normal
        if n1 < 20 or n2 < 20:
            test = "mann-whitney"
        else:
            # Shapiro-Wilk test for normality
            _, p1 = stats.shapiro(group1) if n1 >= 3 else (0, 1)
            _, p2 = stats.shapiro(group2) if n2 >= 3 else (0, 1)
            if p1 < 0.05 or p2 < 0.05:
                test = "mann-whitney"
            else:
                test = "t-test"
    
    # Perform test
    if test == "t-test":
        statistic, p_value = stats.ttest_ind(group1, group2)
        test_name = "Independent t-test"
    elif test == "mann-whitney":
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        test_name = "Mann-Whitney U"
    else:
        raise ValueError(f"Unknown test: {test}")
    
    # Compute effect size
    effect_size = EffectSize.from_groups(group1, group2)
    
    return StatisticalResult(
        test_name=test_name,
        statistic=float(statistic),
        p_value=float(p_value),
        is_significant=p_value < alpha,
        effect_size=effect_size,
        sample_sizes=(n1, n2),
        means=(float(np.mean(group1)), float(np.mean(group2))),
        stds=(float(np.std(group1)), float(np.std(group2))),
    )


def aggregate_results(
    results: List[Dict[str, float]],
    metric_key: str,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Aggregate results across multiple experiments.
    
    Args:
        results: List of result dictionaries
        metric_key: Key for the metric to aggregate
        confidence: Confidence level for interval
        
    Returns:
        Dictionary with mean, std, median, CI, etc.
    """
    values = [r[metric_key] for r in results if metric_key in r]
    
    if not values:
        return {"error": f"No values found for key '{metric_key}'"}
    
    values = np.asarray(values)
    ci = compute_confidence_interval(values, confidence)
    
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        f"ci_{int(confidence*100)}_lower": ci[0],
        f"ci_{int(confidence*100)}_upper": ci[1],
    }


def bootstrap_confidence_interval(
    data: Union[List[float], np.ndarray],
    statistic_func=np.mean,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval.
    
    Useful when distributional assumptions may not hold.
    
    Args:
        data: Sample data
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        random_state: Random seed for reproducibility
        
    Returns:
        (lower, upper) confidence interval bounds
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    data = np.asarray(data)
    n = len(data)
    
    # Bootstrap sampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha / 2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)
    
    return (float(lower), float(upper))


def compute_correlation(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    method: str = "pearson",
) -> Dict[str, float]:
    """
    Compute correlation between two variables.
    
    Args:
        x: First variable
        y: Second variable
        method: "pearson" or "spearman"
        
    Returns:
        Dictionary with correlation coefficient and p-value
    """
    x, y = np.asarray(x), np.asarray(y)
    
    if method == "pearson":
        r, p = stats.pearsonr(x, y)
    elif method == "spearman":
        r, p = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        "method": method,
        "correlation": float(r),
        "p_value": float(p),
        "is_significant": p < 0.05,
    }
