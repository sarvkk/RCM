"""
Report Generator for Refusal Circuit Analysis

Generates comprehensive reports in multiple formats:
- Markdown (for GitHub/documentation)
- LaTeX (for academic papers)
- Jupyter Notebook (for interactive exploration)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import shutil

# Try to import nbformat for notebook generation
try:
    import nbformat
    from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Refusal Circuit Analysis Report"
    author: str = "SaycuredAI Research"
    include_code: bool = True
    include_figures: bool = True
    figure_format: str = "png"
    figure_dpi: int = 300


class ReportGenerator:
    """
    Generates comprehensive reports from experiment results.
    
    Supports multiple output formats:
    - Markdown: GitHub-friendly README style
    - LaTeX: Academic paper format
    - Jupyter: Interactive notebook with reproducible code
    """
    
    def __init__(
        self,
        results: Dict[str, Any],
        output_dir: str = "reports",
        config: Optional[ReportConfig] = None,
    ):
        """
        Initialize report generator.
        
        Args:
            results: Experiment results dictionary or BatchExperimentResults
            output_dir: Directory to save reports
            config: Report configuration
        """
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ReportConfig()
        
        # Create subdirectories
        (self.output_dir / "figures").mkdir(exist_ok=True)
    
    def generate_all(self):
        """Generate all report formats."""
        print("Generating reports...")
        
        # Generate Markdown
        md_path = self.generate_markdown()
        print(f"  ✓ Markdown: {md_path}")
        
        # Generate LaTeX
        latex_path = self.generate_latex()
        print(f"  ✓ LaTeX: {latex_path}")
        
        # Generate Jupyter notebook
        if NBFORMAT_AVAILABLE:
            nb_path = self.generate_jupyter()
            print(f"  ✓ Jupyter: {nb_path}")
        else:
            print("  ⚠ Jupyter notebook skipped (nbformat not installed)")
        
        print(f"\nReports saved to: {self.output_dir}")
    
    def generate_markdown(self) -> Path:
        """Generate Markdown report."""
        content = self._build_markdown_content()
        
        output_path = self.output_dir / "refusal_circuit_analysis.md"
        with open(output_path, "w") as f:
            f.write(content)
        
        return output_path
    
    def generate_latex(self) -> Path:
        """Generate LaTeX report."""
        content = self._build_latex_content()
        
        output_path = self.output_dir / "refusal_circuit_analysis.tex"
        with open(output_path, "w") as f:
            f.write(content)
        
        return output_path
    
    def generate_jupyter(self) -> Path:
        """Generate Jupyter notebook report."""
        if not NBFORMAT_AVAILABLE:
            raise ImportError("nbformat is required for Jupyter notebook generation")
        
        nb = self._build_notebook()
        
        output_path = self.output_dir / "refusal_circuit_analysis.ipynb"
        with open(output_path, "w") as f:
            nbformat.write(nb, f)
        
        return output_path
    
    def _get_model_results(self) -> Dict[str, Any]:
        """Extract model results from various result formats."""
        if hasattr(self.results, 'model_results'):
            return self.results.model_results
        elif isinstance(self.results, dict) and 'model_results' in self.results:
            return self.results['model_results']
        elif isinstance(self.results, dict):
            return self.results
        return {}
    
    def _build_markdown_content(self) -> str:
        """Build Markdown report content."""
        model_results = self._get_model_results()
        
        lines = [
            f"# {self.config.title}",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Author:** {self.config.author}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            "This report presents the results of mechanistic interpretability experiments",
            "aimed at identifying the neural circuit responsible for refusal behavior in",
            "large language models.",
            "",
            "### Key Findings",
            "",
        ]
        
        # Add key findings based on results
        if model_results:
            n_models = len(model_results)
            lines.append(f"- Analyzed **{n_models}** models")
            
            # Average separation score
            sep_scores = []
            for r in model_results.values():
                if r is None:
                    continue
                if hasattr(r, 'separation_score'):
                    sep_scores.append(r.separation_score)
                elif isinstance(r, dict):
                    sep_scores.append(r.get('separation_score', 0))
                else:
                    sep_scores.append(0)
            if sep_scores:
                avg_sep = np.mean([s for s in sep_scores if s])
                lines.append(f"- Average refusal direction separation: **{avg_sep:.2f}σ**")
        
        lines.extend([
            "",
            "---",
            "",
            "## Methodology",
            "",
            "### Activation Patching",
            "",
            "We use activation patching to identify which model components are causally",
            "responsible for refusal behavior. The procedure:",
            "",
            "1. Run model on a refusal-triggering prompt, cache activations",
            "2. Run model on a structurally similar compliant prompt, cache activations",
            "3. Re-run the refusal prompt, but patch in activations from the compliant run",
            "4. Measure how much the patching changes the refusal behavior",
            "",
            "Components that, when patched, cause large behavior changes are identified as",
            "part of the \"refusal circuit.\"",
            "",
            "### Refusal Direction",
            "",
            "We compute a \"refusal direction\" in activation space by:",
            "",
            "1. Collecting activations for refusal prompts and compliant prompts",
            "2. Computing the difference between mean activations (refusal - compliant)",
            "3. Normalizing to get a unit direction vector",
            "",
            "This direction can be used for steering experiments.",
            "",
            "### Steering Validation",
            "",
            "To validate that we found the refusal mechanism, we:",
            "",
            "1. **Force refusal**: Add the refusal direction to harmless prompts",
            "2. **Suppress refusal**: Subtract the refusal direction from harmful prompts",
            "",
            "If behavior changes as expected, this proves causal identification.",
            "",
            "---",
            "",
            "## Results",
            "",
        ])
        
        # Add per-model results
        if model_results:
            lines.append("### Per-Model Results")
            lines.append("")
            lines.append("| Model | Type | Separation | Probe Acc. | Force Refusal | Suppress Refusal |")
            lines.append("|-------|------|------------|------------|---------------|------------------|")
            
            for name, result in model_results.items():
                if hasattr(result, 'model_type'):
                    model_type = result.model_type
                    sep = result.separation_score
                    probe = result.probe_accuracy
                    force = result.force_refusal_success_rate
                    suppress = result.suppress_refusal_success_rate
                elif isinstance(result, dict):
                    model_type = result.get('model_type', 'unknown')
                    sep = result.get('separation_score', 0)
                    probe = result.get('probe_accuracy', 0)
                    force = result.get('force_refusal_success_rate', 0)
                    suppress = result.get('suppress_refusal_success_rate', 0)
                else:
                    continue
                
                lines.append(
                    f"| {name} | {model_type} | {sep:.3f}σ | {probe:.1%} | {force:.1%} | {suppress:.1%} |"
                )
            
            lines.append("")
        
        # Add critical components section
        lines.extend([
            "### Critical Components",
            "",
            "The following components were identified as critical for refusal behavior:",
            "",
        ])
        
        if model_results:
            # Aggregate critical layers across models
            all_critical_layers = []
            for result in model_results.values():
                if hasattr(result, 'critical_layers'):
                    all_critical_layers.extend(result.critical_layers)
                elif isinstance(result, dict) and 'critical_layers' in result:
                    all_critical_layers.extend(result['critical_layers'])
            
            if all_critical_layers:
                layer_counts = {}
                for layer in all_critical_layers:
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1
                
                lines.append("**Most frequently critical layers:**")
                lines.append("")
                for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1])[:5]:
                    lines.append(f"- Layer {layer}: appeared in {count} models")
                lines.append("")
        
        # Discussion
        lines.extend([
            "---",
            "",
            "## Discussion",
            "",
            "### Interpretation",
            "",
            "The results suggest that refusal behavior in language models is:",
            "",
            "1. **Localized**: Specific layers and attention heads are disproportionately responsible",
            "2. **Steerable**: We can manipulate refusal by intervening on the identified direction",
            "3. **Consistent**: Similar circuits appear across different model sizes",
            "",
            "### Implications for AI Safety",
            "",
            "Understanding refusal circuits helps us:",
            "",
            "- Build more robust safety mechanisms",
            "- Understand how alignment techniques affect model internals",
            "- Develop better interpretability tools",
            "",
            "### Limitations",
            "",
            "- Base models (without safety training) show weak refusal signals",
            "- Steering effects may not generalize to all prompt types",
            "- Analysis limited by computational resources to smaller models",
            "",
            "---",
            "",
            "## Reproducibility",
            "",
            "To reproduce these results:",
            "",
            "```bash",
            "# Install dependencies",
            "pip install -r requirements.txt",
            "",
            "# Run experiments",
            "python experiments/batch_runner.py --config experiments/config.yaml",
            "",
            "# Generate reports",
            "python -m src.reports.generator --input results/ --output reports/",
            "```",
            "",
            "---",
            "",
            f"*Report generated by SaycuredAI Refusal Circuit Analysis Framework*",
        ])
        
        return "\n".join(lines)
    
    def _build_latex_content(self) -> str:
        """Build LaTeX report content."""
        model_results = self._get_model_results()
        
        content = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{geometry}

\geometry{margin=1in}

\title{Mapping the Refusal Circuit in Large Language Models:\\
A Mechanistic Interpretability Study}
\author{""" + self.config.author + r"""}
\date{""" + datetime.now().strftime('%B %d, %Y') + r"""}

\begin{document}

\maketitle

\begin{abstract}
We present a systematic analysis of the neural circuits responsible for refusal behavior
in large language models. Using activation patching and steering experiments, we identify
specific layers and attention heads that causally contribute to refusal decisions.
Our findings suggest that refusal behavior is relatively localized and can be manipulated
through targeted interventions on the identified ``refusal direction'' in activation space.
These results have implications for understanding AI safety mechanisms and developing
more robust alignment techniques.
\end{abstract}

\section{Introduction}

When a language model refuses a request (e.g., ``I cannot help with that''), the computational
process underlying this decision remains poorly understood. We ask: is there a specific 
``refusal circuit'' -- a subset of model components that are necessary and sufficient
for refusal behavior?

This question is central to AI safety for several reasons:
\begin{itemize}
    \item Understanding refusal mechanisms helps build more robust safety measures
    \item Localized circuits may be vulnerable to targeted attacks
    \item Interpretability of safety-relevant behavior enables better alignment
\end{itemize}

\section{Methodology}

\subsection{Activation Patching}

We employ activation patching \cite{meng2022locating} to identify causally important components.
Given a refusal-triggering prompt $p_r$ and a structurally similar compliant prompt $p_c$:

\begin{enumerate}
    \item Run the model on $p_r$, caching activations $A_r$ at each hook point
    \item Run the model on $p_c$, caching activations $A_c$
    \item Re-run on $p_r$, but replace $A_r^{(l)}$ with $A_c^{(l)}$ at layer $l$
    \item Measure the change in output probability of refusal tokens
\end{enumerate}

Components where patching causes large behavior changes are identified as part of the
refusal circuit.

\subsection{Refusal Direction}

We compute a refusal direction $d$ in the residual stream:
\begin{equation}
    d = \frac{\bar{A}_r - \bar{A}_c}{\|\bar{A}_r - \bar{A}_c\|}
\end{equation}
where $\bar{A}_r$ and $\bar{A}_c$ are mean activations over refusal and compliant prompts.

\subsection{Steering Validation}

To validate causal identification, we perform steering experiments:
\begin{itemize}
    \item \textbf{Force refusal}: $A' = A + \alpha \cdot d$ on harmless prompts
    \item \textbf{Suppress refusal}: $A' = A - \alpha \cdot d$ on harmful prompts
\end{itemize}
If behavior changes as predicted, this confirms we identified the control mechanism.

\section{Results}

"""
        # Add results table
        if model_results:
            content += r"""
\subsection{Per-Model Analysis}

\begin{table}[h]
\centering
\caption{Refusal circuit analysis results across models}
\begin{tabular}{lcccc}
\toprule
Model & Separation ($\sigma$) & Probe Acc. & Force Rate & Suppress Rate \\
\midrule
"""
            for name, result in model_results.items():
                if hasattr(result, 'separation_score'):
                    sep = result.separation_score
                    probe = result.probe_accuracy
                    force = result.force_refusal_success_rate
                    suppress = result.suppress_refusal_success_rate
                elif isinstance(result, dict):
                    sep = result.get('separation_score', 0)
                    probe = result.get('probe_accuracy', 0)
                    force = result.get('force_refusal_success_rate', 0)
                    suppress = result.get('suppress_refusal_success_rate', 0)
                else:
                    continue
                
                name_escaped = name.replace("_", r"\_")
                content += f"{name_escaped} & {sep:.3f} & {probe:.1%} & {force:.1%} & {suppress:.1%} \\\\\n"
            
            content += r"""\bottomrule
\end{tabular}
\end{table}

"""
        
        content += r"""
\section{Discussion}

Our results indicate that refusal behavior in LLMs is mediated by identifiable
neural circuits. The refusal direction we extract shows clear separation between
refusal and compliant prompts, and steering experiments demonstrate causal control.

\subsection{Implications for AI Safety}

These findings suggest that current safety training creates relatively localized
modifications to model behavior. This has both positive implications (interpretability)
and concerning ones (potential for targeted manipulation).

\subsection{Limitations}

Our analysis is limited to smaller open-source models due to computational constraints.
Base models without safety training show weak refusal signals, making circuit identification
more challenging.

\section{Conclusion}

We have demonstrated that activation patching and steering can identify and manipulate
refusal circuits in language models. This work contributes to the mechanistic interpretability
of AI safety mechanisms and suggests directions for more robust alignment techniques.

\bibliographystyle{plain}
\begin{thebibliography}{9}

\bibitem{meng2022locating}
Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov.
Locating and editing factual associations in GPT.
\textit{Advances in Neural Information Processing Systems}, 2022.

\bibitem{elhage2022circuits}
Nelson Elhage et al.
A mathematical framework for transformer circuits.
\textit{Transformer Circuits Thread}, 2021.

\end{thebibliography}

\end{document}
"""
        return content
    
    def _build_notebook(self):
        """Build Jupyter notebook with analysis code."""
        nb = new_notebook()
        
        # Title cell
        nb.cells.append(new_markdown_cell(
            f"# {self.config.title}\n\n"
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "This notebook contains the complete analysis pipeline and results."
        ))
        
        # Setup cell
        nb.cells.append(new_code_cell(
            "# Setup\n"
            "import sys\n"
            "sys.path.insert(0, '..')\n\n"
            "import json\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "from src.models import load_model, list_available_models\n"
            "from src.data import REFUSAL_PROMPT_PAIRS\n"
            "from src.circuits import CircuitAnalyzer, compute_refusal_direction\n"
            "from src.steering import ClampingExperiment\n"
            "from src.analysis import significance_test, compare_models\n\n"
            "# Style\n"
            "plt.style.use('seaborn-v0_8-whitegrid')\n"
            "%matplotlib inline"
        ))
        
        # Load results cell
        nb.cells.append(new_markdown_cell(
            "## Load Experiment Results\n\n"
            "Load the results from batch experiments."
        ))
        
        nb.cells.append(new_code_cell(
            "# Load results\n"
            "results_path = 'batch_results.json'  # Adjust path as needed\n"
            "try:\n"
            "    with open(results_path, 'r') as f:\n"
            "        results = json.load(f)\n"
            "    print(f'Loaded results for {len(results.get(\"model_results\", {}))} models')\n"
            "except FileNotFoundError:\n"
            "    print('Results file not found. Run batch_runner.py first.')\n"
            "    results = {}"
        ))
        
        # Summary statistics
        nb.cells.append(new_markdown_cell(
            "## Summary Statistics\n\n"
            "Overview of results across all models."
        ))
        
        nb.cells.append(new_code_cell(
            "# Extract metrics\n"
            "if 'model_results' in results:\n"
            "    model_results = results['model_results']\n"
            "    \n"
            "    sep_scores = [r['separation_score'] for r in model_results.values()]\n"
            "    probe_accs = [r['probe_accuracy'] for r in model_results.values()]\n"
            "    \n"
            "    print('Separation Scores:')\n"
            "    print(f'  Mean: {np.mean(sep_scores):.3f}')\n"
            "    print(f'  Std:  {np.std(sep_scores):.3f}')\n"
            "    print(f'  Min:  {np.min(sep_scores):.3f}')\n"
            "    print(f'  Max:  {np.max(sep_scores):.3f}')\n"
            "    \n"
            "    print('\\nProbe Accuracy:')\n"
            "    print(f'  Mean: {np.mean(probe_accs):.1%}')\n"
            "    print(f'  Std:  {np.std(probe_accs):.1%}')"
        ))
        
        # Visualization
        nb.cells.append(new_markdown_cell(
            "## Visualizations\n\n"
            "Key plots from the analysis."
        ))
        
        nb.cells.append(new_code_cell(
            "# Bar plot of separation scores by model\n"
            "if 'model_results' in results:\n"
            "    models = list(model_results.keys())\n"
            "    sep_scores = [model_results[m]['separation_score'] for m in models]\n"
            "    \n"
            "    fig, ax = plt.subplots(figsize=(10, 5))\n"
            "    colors = ['#e63946' if model_results[m].get('model_type') == 'instruction_tuned' \n"
            "              else '#457b9d' for m in models]\n"
            "    ax.bar(models, sep_scores, color=colors)\n"
            "    ax.set_ylabel('Separation Score (σ)')\n"
            "    ax.set_xlabel('Model')\n"
            "    ax.set_title('Refusal Direction Separation by Model')\n"
            "    plt.xticks(rotation=45, ha='right')\n"
            "    plt.tight_layout()\n"
            "    plt.show()"
        ))
        
        # Interactive analysis
        nb.cells.append(new_markdown_cell(
            "## Interactive Analysis\n\n"
            "Run custom analyses on specific models."
        ))
        
        nb.cells.append(new_code_cell(
            "# Analyze a specific model (modify as needed)\n"
            "MODEL_NAME = 'pythia-70m'\n"
            "N_PAIRS = 5\n\n"
            "try:\n"
            "    # Load model\n"
            "    model = load_model(MODEL_NAME, device='cuda')\n"
            "    \n"
            "    # Get prompt pairs\n"
            "    pairs = REFUSAL_PROMPT_PAIRS[:N_PAIRS]\n"
            "    \n"
            "    # Run analysis\n"
            "    analyzer = CircuitAnalyzer(model)\n"
            "    circuit = analyzer.analyze_prompt_pair(pairs[0], components='resid')\n"
            "    \n"
            "    print('Top 5 components:')\n"
            "    for comp in circuit.top_k_components(5):\n"
            "        print(f'  {comp.name}: {comp.importance_score:.4f}')\n"
            "except Exception as e:\n"
            "    print(f'Error: {e}')"
        ))
        
        # Conclusions
        nb.cells.append(new_markdown_cell(
            "## Conclusions\n\n"
            "Key takeaways from this analysis:\n\n"
            "1. Refusal behavior is localized to specific layers and attention heads\n"
            "2. The refusal direction provides clear separation between prompt types\n"
            "3. Steering experiments validate causal identification\n\n"
            "---\n\n"
            "*Generated by SaycuredAI Refusal Circuit Analysis Framework*"
        ))
        
        return nb


def generate_full_report(
    results_path: str,
    output_dir: str = "reports",
    formats: List[str] = None,
) -> Dict[str, Path]:
    """
    Convenience function to generate all reports from a results file.
    
    Args:
        results_path: Path to results JSON file
        output_dir: Output directory for reports
        formats: List of formats to generate (default: all)
        
    Returns:
        Dict mapping format names to output paths
    """
    formats = formats or ["markdown", "latex", "jupyter"]
    
    # Load results
    with open(results_path, "r") as f:
        results = json.load(f)
    
    generator = ReportGenerator(results, output_dir=output_dir)
    
    outputs = {}
    
    if "markdown" in formats:
        outputs["markdown"] = generator.generate_markdown()
    
    if "latex" in formats:
        outputs["latex"] = generator.generate_latex()
    
    if "jupyter" in formats and NBFORMAT_AVAILABLE:
        outputs["jupyter"] = generator.generate_jupyter()
    
    return outputs


# CLI interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reports from experiment results")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output", "-o", type=str, default="reports", help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["markdown", "latex", "jupyter"],
                       help="Formats to generate")
    
    args = parser.parse_args()
    
    outputs = generate_full_report(args.input, args.output, args.formats)
    
    print("Generated reports:")
    for fmt, path in outputs.items():
        print(f"  {fmt}: {path}")
