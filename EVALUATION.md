# RE-ARC Evaluation Guide

## Quick Start

### 1. Download a model from Hugging Face

```bash
# Just download
python pull_from_hf.py username/re-arc-model --output ./models/my_model

# Download specific checkpoint
python pull_from_hf.py username/re-arc-model --checkpoint checkpoint_epoch_1000.pt

# Download and load into memory to verify
python pull_from_hf.py username/re-arc-model --checkpoint checkpoint_epoch_1000.pt --load
```

### 2. Evaluate on training tasks

```bash
# Greedy decoding on 100 synthetic training tasks
python evaluate.py --checkpoint checkpoints/model.pt --mode train --num-tasks 100

# With beam search
python evaluate.py --checkpoint checkpoints/model.pt --mode train --beam-width 10

# Measure entropy to guide search strategy
python evaluate.py --checkpoint checkpoints/model.pt --mode train --measure-entropy
```

### 3. Evaluate on real ARC test tasks

```bash
# Greedy decoding on all ARC evaluation tasks
uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval

# With beam search
uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval --beam-width 10

# Quick sanity check on 10 evaluation tasks
uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval --num-tasks 10

# Use training set instead (challenges + solutions)
uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval \
  --challenges data/arc-agi_training_challenges.json \
  --solutions data/arc-agi_training_solutions.json
```

## Evaluation Modes

### Training Mode (`--mode train`)
- Evaluates on **newly generated synthetic examples** using `generators.py`
- Each evaluation generates a **brand new** random example per task
- Examples are **validated** with verifiers (same as training)
- Defaults to **100 random tasks**; override with `--num-tasks`
- Good for quick validation during training
- Metrics: syntax validity, runtime success, correctness

**What it tests:** Generalization on new examples from the same task distribution

### Evaluation Mode (`--mode eval`)
- Evaluates on **real ARC test examples** from official challenge files
- Uses `data/arc-agi_evaluation_challenges.json` and `data/arc-agi_evaluation_solutions.json`
- This is the **actual held-out test set** - not synthetic!
- Each task has 1-2 test examples
- Uses **all 400 tasks by default**; pass `--num-tasks` to limit (useful for debugging)
- Metrics: tasks solved, examples correct, syntax validity, runtime success

**What it tests:** Real benchmark performance on the official ARC evaluation set

## Search Strategies

### Greedy Decoding (default)
```bash
python evaluate.py --checkpoint model.pt --mode eval
```
- Fastest
- Always picks most probable token
- No exploration of alternatives

### Beam Search
```bash
python evaluate.py --checkpoint model.pt --mode eval --beam-width 10
```
- Explores top-K alternatives at each step
- Better accuracy, slower inference
- Recommended K values:
  - K=5: Fast, moderate improvement
  - K=10: Sweet spot for most cases
  - K=20: High accuracy, diminishing returns

## Entropy Measurement

Use `--measure-entropy` to analyze model confidence:

```bash
python evaluate.py --checkpoint model.pt --mode train --measure-entropy
```

This measures the average entropy (uncertainty) of the model's predictions.

**Interpreting entropy:**
- **Low entropy (<10% of max)**: Model is very confident
  - Beam search offers limited benefit
  - Greedy decoding usually sufficient

- **Medium entropy (10-30%)**: Moderate uncertainty
  - Beam search K=5-10 recommended
  - Moderate improvement expected

- **High entropy (>30%)**: High uncertainty
  - Beam search K=10-20 or sampling recommended
  - Significant improvement possible

## Metrics Explained

### Syntax Valid
Percentage of generated programs that parse as valid Python AST.
If low, model hasn't learned basic syntax.

### Runtime Success
Percentage of syntactically valid programs that execute without errors.
If low relative to syntax valid, model generates valid but semantically broken code.

### Correct
Percentage of programs that produce the exact target output.
The ultimate metric - solving the task correctly.

### Tasks Solved (eval mode only)
Percentage of tasks where at least one test example was solved correctly.
In real ARC evaluation, solving any example counts as "solving" the task.

## Examples

### Quick validation after training
```bash
python evaluate.py \
  --checkpoint checkpoints/checkpoint_epoch_5000.pt \
  --mode train \
  --num-tasks 50 \
  --measure-entropy
```

### Full evaluation with beam search
```bash
python evaluate.py \
  --checkpoint checkpoints/final_model.pt \
  --mode eval \
  --beam-width 10 \
  --measure-entropy
```

### Compare greedy vs beam
```bash
# Greedy
python evaluate.py --checkpoint model.pt --mode eval > results_greedy.txt

# Beam
python evaluate.py --checkpoint model.pt --mode eval --beam-width 10 > results_beam.txt
```

## Device Selection

Scripts auto-detect the best available device:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (fallback)

Override with `--device`:
```bash
python evaluate.py --checkpoint model.pt --mode eval --device cpu
```

## Tips

1. **Start with entropy measurement** to guide search strategy choice
2. **Use training mode for quick iterations** during development
3. **Use eval mode for final benchmarks** on real ARC tasks
4. **Monitor all metrics**, not just correctness:
   - Syntax % → basic language learning
   - Runtime % → semantic understanding
   - Correct % → task solving ability
5. **Beam search trades off speed for accuracy** - choose K based on your needs
