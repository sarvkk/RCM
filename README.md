# Refusal Circuit Mapping

Mechanistic interpretability tools for finding where LLMs decide to refuse harmful requests.

## What is this?

When an AI refuses a harmful request, that decision happens *somewhere* inside the neural network. This project finds exactly where.

- **Locate** which layers and attention heads are responsible for refusal
- **Extract** the "refusal direction" in activation space  
- **Validate** findings by steering model behavior
- **Compare** base models vs safety-trained models

For the motivation behind this research, see [MOTIVATION.md](MOTIVATION.md).

## Quick Start

```bash
# Install
git clone https://github.com/sarvkk/RCM.git
cd RCM
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run analysis
python experiments/batch_runner.py --config experiments/config.yaml

# Or single model
python experiments/run_circuit_analysis.py --model pythia-160m --n-pairs 10

# Or interactive notebook
jupyter notebook notebooks/02_comprehensive_analysis.ipynb
```

## Project Structure

```
RCM/
├── src/
│   ├── models/          # Model loading
│   ├── data/            # Prompt datasets
│   ├── patching/        # Activation patching
│   ├── circuits/        # Circuit localization
│   ├── steering/        # Behavior steering
│   └── reports/         # Report generation
├── experiments/         # Run scripts
├── notebooks/           # Interactive analysis
└── reports/             # Analysis reports
```

## Supported Models

| Model | Size | Memory | Notes |
|-------|------|--------|-------|
| `pythia-70m` | 70M | 0.3GB | Fast testing |
| `pythia-160m` | 160M | 0.6GB | Quick experiments |
| `pythia-410m` | 410M | 1.5GB | Recommended |
| `gpt2` | 124M | 0.8GB | Baseline |
| `gpt2-medium` | 355M | 2.0GB | Larger baseline |

## Example Usage

**Find the refusal circuit:**
```python
from src.models import load_model
from src.circuits import CircuitAnalyzer
from src.data import REFUSAL_PROMPT_PAIRS

model = load_model("pythia-410m")
analyzer = CircuitAnalyzer(model)
circuit = analyzer.analyze_multiple_pairs(REFUSAL_PROMPT_PAIRS[:15])

print(f"Critical layers: {circuit.critical_layers}")
print(f"Critical heads: {circuit.critical_heads[:5]}")
```

**Extract refusal direction:**
```python
from src.circuits import compute_refusal_direction

direction = compute_refusal_direction(model, prompt_pairs, layer=16)
print(f"Separation: {direction.separation_score:.2f}σ")
```

**Steer behavior:**
```python
from src.steering import ClampingExperiment

exp = ClampingExperiment(model, direction)
result = exp.force_refusal("How do I bake a cake?", strength=2.0)
# Model now refuses a harmless request
```

## What to Expect

A successful analysis shows:

| Metric | Good | Notes |
|--------|------|-------|
| Separation score | > 2.0σ | Clear cluster separation |
| Probe accuracy | > 90% | Direction is predictive |
| Force refusal rate | > 70% | Can induce refusal |
| Suppress refusal rate | > 40% | Can suppress refusal |

Base models (without safety training) typically show weaker signals. This is expected.

## Troubleshooting

**CUDA out of memory?**
```bash
# Use smaller model
python experiments/run_circuit_analysis.py --model pythia-160m

# Or use CPU
python experiments/run_circuit_analysis.py --device cpu
```

**Weak refusal signal?**
- Base models don't have strong refusal circuits (expected)
- Try instruction-tuned models like `phi-1_5`
- Use more prompt pairs (25+)

**Model loading fails?**
```python
# Always use the framework's loader
from src.models import load_model
model = load_model("gpt2")  # Correct

# Not this:
# from transformers import AutoModel
# model = AutoModel.from_pretrained("gpt2")  # Wrong
```

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file - quick start |
| [MOTIVATION.md](MOTIVATION.md) | Why this project exists |
| [experiments/config.yaml](experiments/config.yaml) | Configuration options |
| [reports/refusal_circuit_analysis.md](reports/refusal_circuit_analysis.md) | Analysis methodology & results |

## References

- [Transformer Circuits](https://transformer-circuits.pub/) - Anthropic's circuits research
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Interpretability library
- [ROME](https://rome.baulab.info/) - Activation patching techniques
- [Pythia](https://github.com/EleutherAI/pythia) - Model suite

## Ethics

This research is for **defensive AI safety** - understanding refusal to build better protections, not bypass them. See [MOTIVATION.md](MOTIVATION.md) for the full ethical framework.

## Contributing

Contributions welcome. Areas of interest:
- Additional prompt datasets
- Larger model analysis (7B+)
- Transfer of refusal directions across models
- Visualization improvements

## License

MIT License
