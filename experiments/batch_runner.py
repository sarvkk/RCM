#!/usr/bin/env python3
"""
Batch Experiment Runner for Refusal Circuit Analysis

Runs comprehensive experiments across multiple models to:
1. Compare base models vs instruction-tuned models
2. Identify refusal circuits using activation patching
3. Validate findings with steering experiments
4. Generate statistical analysis and reports

Usage:
    python experiments/batch_runner.py --config experiments/config.yaml
    python experiments/batch_runner.py --models pythia-70m pythia-160m --n-pairs 10
    python experiments/batch_runner.py --full-analysis
"""

import argparse
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm
import logging
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    load_model, 
    get_model_config, 
    get_models_by_memory,
    ALL_MODELS,
    BASE_MODELS,
    INSTRUCTION_TUNED_MODELS,
    ModelType,
)
from src.data import REFUSAL_PROMPT_PAIRS, get_prompt_pairs_by_category
from src.circuits import CircuitAnalyzer, compute_refusal_direction
from src.steering import ClampingExperiment, run_clamping_validation
from src.patching import run_patching_experiment


@dataclass
class ExperimentResult:
    """Results from a single model experiment."""
    model_name: str
    model_type: str
    timestamp: str
    
    # Circuit analysis results
    critical_layers: List[int]
    critical_heads: List[tuple]
    critical_mlps: List[int]
    component_scores: Dict[str, float]
    
    # Refusal direction results
    refusal_direction_layer: int
    separation_score: float
    probe_accuracy: float
    
    # Steering validation results
    force_refusal_success_rate: float
    suppress_refusal_success_rate: float
    
    # Metadata
    n_prompt_pairs: int
    device: str
    duration_seconds: float


@dataclass
class BatchExperimentResults:
    """Results from a batch of experiments across models."""
    experiment_name: str
    timestamp: str
    config: Dict[str, Any]
    
    # Per-model results
    model_results: Dict[str, ExperimentResult]
    
    # Aggregate statistics
    base_model_stats: Dict[str, Any]
    instruction_tuned_stats: Dict[str, Any]
    
    # Comparison metrics
    comparison_metrics: Dict[str, Any]


class ExperimentRunner:
    """
    Runs refusal circuit experiments across multiple models.
    
    Handles:
    - Model loading with memory management
    - Automatic CPU fallback for large models
    - Checkpointing for long experiments
    - Statistical aggregation across models
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = "results",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Results storage
        self.results: Dict[str, ExperimentResult] = {}
        
        # Timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_device_for_model(self, model_name: str) -> str:
        """Determine best device for a model based on memory."""
        hw_config = self.config.get("hardware", {})
        max_gpu_mem = hw_config.get("max_gpu_memory", 3.0)
        
        if not torch.cuda.is_available():
            return "cpu"
        
        if model_name in ALL_MODELS:
            model_config = ALL_MODELS[model_name]
            if model_config.memory_gb <= max_gpu_mem:
                return "cuda"
            elif hw_config.get("cpu_fallback", True):
                self.logger.info(f"Model {model_name} ({model_config.memory_gb}GB) exceeds GPU limit, using CPU")
                return "cpu"
            else:
                raise RuntimeError(f"Model {model_name} too large for GPU and CPU fallback disabled")
        
        return "cuda" if hw_config.get("prefer_gpu", True) else "cpu"
    
    def _get_dtype(self, device: str) -> torch.dtype:
        """Get appropriate dtype for device."""
        hw_config = self.config.get("hardware", {})
        if device == "cuda":
            dtype_str = hw_config.get("gpu_dtype", "float16")
        else:
            dtype_str = hw_config.get("cpu_dtype", "float32")
        
        return torch.float16 if dtype_str == "float16" else torch.float32
    
    def run_single_model(
        self,
        model_name: str,
        prompt_pairs: List,
        save_checkpoint: bool = True,
    ) -> ExperimentResult:
        """
        Run complete analysis on a single model.
        
        Returns:
            ExperimentResult with all analysis results
        """
        start_time = datetime.now()
        self.logger.info(f"=" * 60)
        self.logger.info(f"Analyzing model: {model_name}")
        self.logger.info(f"=" * 60)
        
        # Determine device and load model
        device = self._get_device_for_model(model_name)
        dtype = self._get_dtype(device)
        
        try:
            model = load_model(model_name, device=device, dtype=dtype)
        except Exception as e:
            self.logger.error(f"Failed to load {model_name}: {e}")
            raise
        
        # Get model type
        if model_name in ALL_MODELS:
            model_type = ALL_MODELS[model_name].model_type.value
        else:
            model_type = "unknown"
        
        # Run circuit analysis
        self.logger.info("Running circuit analysis...")
        analysis_config = self.config.get("analysis", {})
        
        analyzer = CircuitAnalyzer(model)
        circuit = analyzer.analyze_multiple_pairs(
            prompt_pairs,
            components=analysis_config.get("components", "all"),
            aggregate="mean",
        )
        
        # Compute component scores dict
        component_scores = {
            comp.name: comp.importance_score 
            for comp in circuit.top_k_components(20)
        }
        
        # Compute refusal direction
        self.logger.info("Computing refusal direction...")
        best_layer = circuit.critical_layers[0] if circuit.critical_layers else model.cfg.n_layers // 2
        
        refusal_direction = compute_refusal_direction(
            model,
            prompt_pairs,
            layer=best_layer,
            method="mean_diff",
        )
        
        # Run steering validation
        self.logger.info("Running steering validation...")
        steering_config = self.config.get("steering", {})
        
        harmless_prompts = steering_config.get("harmless_test_prompts", [
            "How do I bake a cake?",
            "What is the capital of France?",
            "Tell me a joke",
        ])
        harmful_prompts = [p.refusal_prompt for p in prompt_pairs[:3]]
        
        clamping_results = run_clamping_validation(
            model,
            refusal_direction,
            harmless_prompts,
            harmful_prompts,
            strength_values=steering_config.get("strengths", [1.0, 2.0, 3.0]),
        )
        
        # Calculate success rates
        force_success = sum(1 for r in clamping_results["force_refusal"] if r.behavior_changed)
        suppress_success = sum(1 for r in clamping_results["suppress_refusal"] if r.behavior_changed)
        
        force_rate = force_success / max(len(clamping_results["force_refusal"]), 1)
        suppress_rate = suppress_success / max(len(clamping_results["suppress_refusal"]), 1)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        
        # Create result
        result = ExperimentResult(
            model_name=model_name,
            model_type=model_type,
            timestamp=datetime.now().isoformat(),
            critical_layers=circuit.critical_layers,
            critical_heads=circuit.critical_heads,
            critical_mlps=circuit.critical_mlps,
            component_scores=component_scores,
            refusal_direction_layer=best_layer,
            separation_score=refusal_direction.separation_score,
            probe_accuracy=refusal_direction.probe_accuracy or 0.0,
            force_refusal_success_rate=force_rate,
            suppress_refusal_success_rate=suppress_rate,
            n_prompt_pairs=len(prompt_pairs),
            device=device,
            duration_seconds=duration,
        )
        
        # Store result
        self.results[model_name] = result
        
        # Save checkpoint
        if save_checkpoint:
            self._save_checkpoint(model_name, result)
        
        # Clean up GPU memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        self.logger.info(f"Completed {model_name} in {duration:.1f}s")
        self.logger.info(f"  Separation score: {result.separation_score:.3f}")
        self.logger.info(f"  Force refusal rate: {force_rate:.1%}")
        self.logger.info(f"  Suppress refusal rate: {suppress_rate:.1%}")
        
        return result
    
    def _save_checkpoint(self, model_name: str, result: ExperimentResult):
        """Save checkpoint for a completed model."""
        checkpoint_dir = self.output_dir / "checkpoints" / self.timestamp
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"{model_name}.json"
        with open(checkpoint_path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        
        self.logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
    def run_batch(
        self,
        model_names: List[str],
        n_prompt_pairs: int = 15,
    ) -> BatchExperimentResults:
        """
        Run experiments on multiple models.
        
        Args:
            model_names: List of model names to analyze
            n_prompt_pairs: Number of prompt pairs to use
            
        Returns:
            BatchExperimentResults with all results and statistics
        """
        self.logger.info("=" * 70)
        self.logger.info("BATCH REFUSAL CIRCUIT ANALYSIS")
        self.logger.info("=" * 70)
        self.logger.info(f"Models: {model_names}")
        self.logger.info(f"Prompt pairs: {n_prompt_pairs}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("=" * 70)
        
        # Get prompt pairs
        prompt_pairs = REFUSAL_PROMPT_PAIRS[:n_prompt_pairs]
        
        # Run each model
        for model_name in tqdm(model_names, desc="Models"):
            try:
                self.run_single_model(model_name, prompt_pairs)
            except Exception as e:
                self.logger.error(f"Failed on {model_name}: {e}")
                continue
        
        # Compute aggregate statistics
        base_results = {k: v for k, v in self.results.items() if v.model_type == "base"}
        inst_results = {k: v for k, v in self.results.items() if v.model_type == "instruction_tuned"}
        
        base_stats = self._compute_aggregate_stats(base_results)
        inst_stats = self._compute_aggregate_stats(inst_results)
        
        # Compute comparison metrics
        comparison = self._compute_comparison_metrics(base_results, inst_results)
        
        # Create batch results
        batch_results = BatchExperimentResults(
            experiment_name=self.config.get("experiment", {}).get("name", "refusal_analysis"),
            timestamp=self.timestamp,
            config=self.config,
            model_results=self.results,
            base_model_stats=base_stats,
            instruction_tuned_stats=inst_stats,
            comparison_metrics=comparison,
        )
        
        # Save final results
        self._save_batch_results(batch_results)
        
        return batch_results
    
    def _compute_aggregate_stats(self, results: Dict[str, ExperimentResult]) -> Dict[str, Any]:
        """Compute aggregate statistics for a group of models."""
        if not results:
            return {}
        
        sep_scores = [r.separation_score for r in results.values()]
        probe_accs = [r.probe_accuracy for r in results.values()]
        force_rates = [r.force_refusal_success_rate for r in results.values()]
        suppress_rates = [r.suppress_refusal_success_rate for r in results.values()]
        
        return {
            "n_models": len(results),
            "separation_score": {
                "mean": float(np.mean(sep_scores)),
                "std": float(np.std(sep_scores)),
                "min": float(np.min(sep_scores)),
                "max": float(np.max(sep_scores)),
            },
            "probe_accuracy": {
                "mean": float(np.mean(probe_accs)),
                "std": float(np.std(probe_accs)),
            },
            "force_refusal_rate": {
                "mean": float(np.mean(force_rates)),
                "std": float(np.std(force_rates)),
            },
            "suppress_refusal_rate": {
                "mean": float(np.mean(suppress_rates)),
                "std": float(np.std(suppress_rates)),
            },
        }
    
    def _compute_comparison_metrics(
        self,
        base_results: Dict[str, ExperimentResult],
        inst_results: Dict[str, ExperimentResult],
    ) -> Dict[str, Any]:
        """Compute comparison metrics between base and instruction-tuned models."""
        if not base_results or not inst_results:
            return {"note": "Insufficient data for comparison"}
        
        base_sep = [r.separation_score for r in base_results.values()]
        inst_sep = [r.separation_score for r in inst_results.values()]
        
        base_force = [r.force_refusal_success_rate for r in base_results.values()]
        inst_force = [r.force_refusal_success_rate for r in inst_results.values()]
        
        return {
            "separation_score_diff": float(np.mean(inst_sep) - np.mean(base_sep)),
            "force_refusal_diff": float(np.mean(inst_force) - np.mean(base_force)),
            "base_models_analyzed": list(base_results.keys()),
            "instruction_tuned_analyzed": list(inst_results.keys()),
        }
    
    def _save_batch_results(self, results: BatchExperimentResults):
        """Save batch results to disk."""
        results_dir = self.output_dir / self.timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results JSON
        results_path = results_dir / "batch_results.json"
        results_dict = {
            "experiment_name": results.experiment_name,
            "timestamp": results.timestamp,
            "config": results.config,
            "model_results": {k: asdict(v) for k, v in results.model_results.items()},
            "base_model_stats": results.base_model_stats,
            "instruction_tuned_stats": results.instruction_tuned_stats,
            "comparison_metrics": results.comparison_metrics,
        }
        
        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        self.logger.info(f"Saved results to: {results_path}")
        
        # Save summary
        self._save_summary(results, results_dir)
    
    def _save_summary(self, results: BatchExperimentResults, output_dir: Path):
        """Save human-readable summary."""
        summary_path = output_dir / "summary.txt"
        
        with open(summary_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("REFUSAL CIRCUIT ANALYSIS - SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Experiment: {results.experiment_name}\n")
            f.write(f"Timestamp: {results.timestamp}\n")
            f.write(f"Models analyzed: {len(results.model_results)}\n\n")
            
            f.write("-" * 50 + "\n")
            f.write("PER-MODEL RESULTS\n")
            f.write("-" * 50 + "\n")
            
            for name, result in results.model_results.items():
                f.write(f"\n{name} [{result.model_type}]\n")
                f.write(f"  Separation score: {result.separation_score:.3f}\n")
                f.write(f"  Probe accuracy: {result.probe_accuracy:.1%}\n")
                f.write(f"  Force refusal rate: {result.force_refusal_success_rate:.1%}\n")
                f.write(f"  Suppress refusal rate: {result.suppress_refusal_success_rate:.1%}\n")
                f.write(f"  Critical layers: {result.critical_layers}\n")
            
            f.write("\n" + "-" * 50 + "\n")
            f.write("AGGREGATE STATISTICS\n")
            f.write("-" * 50 + "\n")
            
            if results.base_model_stats:
                f.write("\nBase Models:\n")
                stats = results.base_model_stats
                f.write(f"  Separation: {stats['separation_score']['mean']:.3f} ± {stats['separation_score']['std']:.3f}\n")
                f.write(f"  Force refusal: {stats['force_refusal_rate']['mean']:.1%}\n")
            
            if results.instruction_tuned_stats:
                f.write("\nInstruction-Tuned Models:\n")
                stats = results.instruction_tuned_stats
                f.write(f"  Separation: {stats['separation_score']['mean']:.3f} ± {stats['separation_score']['std']:.3f}\n")
                f.write(f"  Force refusal: {stats['force_refusal_rate']['mean']:.1%}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        self.logger.info(f"Saved summary to: {summary_path}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch refusal circuit analysis experiments"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to configuration YAML file",
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to analyze (overrides config)",
    )
    
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=None,
        help="Number of prompt pairs to use (overrides config)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Run full analysis with reports generation",
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # List models if requested
    if args.list_models:
        from src.models import list_available_models
        list_available_models(verbose=True)
        return
    
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"Config not found at {config_path}, using defaults")
        config = {}
    
    # Determine models to run
    if args.models:
        model_names = args.models
    else:
        models_config = config.get("models", {})
        model_names = models_config.get("small_gpu", ["pythia-70m", "pythia-160m"])
        if config.get("hardware", {}).get("cpu_fallback", True):
            model_names.extend(models_config.get("medium_cpu", []))
    
    # Determine number of prompt pairs
    n_pairs = args.n_pairs or config.get("analysis", {}).get("n_prompt_pairs", 15)
    
    # Create runner and execute
    runner = ExperimentRunner(config, output_dir=args.output_dir)
    
    results = runner.run_batch(model_names, n_prompt_pairs=n_pairs)
    
    # Generate reports if full analysis
    if args.full_analysis:
        print("\nGenerating reports...")
        try:
            from src.reports import ReportGenerator
            report_gen = ReportGenerator(results, output_dir=args.output_dir)
            report_gen.generate_all()
        except ImportError as e:
            print(f"Could not import report generator: {e}")
            print("Hint: Make sure you're running from the project root directory")
    
    print("\n" + "=" * 60)
    print("BATCH ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {runner.output_dir / runner.timestamp}")
    print(f"Models analyzed: {len(results.model_results)}")


if __name__ == "__main__":
    main()
