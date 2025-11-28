# Neuro-DSL Project Plan
[Runpod one-click affiliate template](https://console.runpod.io/deploy?template=6w7lls0jx4&ref=jmfkcdio)

## Current Status
- [x] Basic Tokenizer (`tokenizer.py`)
- [x] Infinite Dataset Generator (`dataset.py`)
- [x] Recursive Transformer Model (`train.py`)
- [x] Environment Setup (`pyproject.toml`)
- [x] **Execution Metrics**: Implement `try_execute(code)` to measure:
    - **% Syntax Valid**: Valid Python AST.
    - **% Runtime Success**: Runs without crashing.
    - **% Solved**: Produces exact target grid.

Last work:
- Added in guided decoding but score is now down at 2.4$ instead of what was 3.5%. Syntax valid is now 97.4%, which isn't higher than without guidance...

============================================================
EVALUATION RESULTS (eval mode)
============================================================
Total tasks:     400
Tasks solved:      14 /  400 (3.5%)
Total examples:  419
Syntax valid:     409 /  419 (97.6%)
Runtime success:  295 /  419 (70.4%)
Correct:           14 /  419 (3.3%)
============================================================

and with guidance:
============================================================
EVALUATION RESULTS (eval mode)
============================================================
Total tasks:     400
Tasks solved:      10 /  400 (2.5%)
Total examples:  419
Syntax valid:     408 /  419 (97.4%)
Runtime success:  279 /  419 (66.6%)
Correct:           10 /  419 (2.4%)
============================================================

## TODO
- [x] **Review code overall**
    - [x] Review tokenizer
    - [x] Review data prep
    - [x] Review training code
- [ ] **Review architecture** Ensure choices make sense for the application at hand.
- [ ] **Guided Decoding**: Constrain generation.
- [ ] **Grammar-Constrained Training**: apply the same token-masking rules during teacher forcing so the model learns the DSL grammar, not just enforced at inference.
- [ ] **Static Analysis Optimization**: Filter programs with `ast.parse` and symbol checks before expensive execution.
- [ ] Do predicted programs include upper and lower bound parameters? if so, that could be powerful for data augmentation.
- [ ] **BPE Tokenization**: Implement Byte Pair Encoding to compound frequent DSL idioms.
- [ ] **Evaluation Script**: Create `evaluate.py` to test on `arc-agi_rearc_challenges.json.gz`.
- [ ] **Search Strategy**: Implement Beam Search or Temperature Sampling.

## Performance Notes
20k model on AA1 eval (naive): 3.5%
+ constrained de-coding: 
+ beam search: 2.4% (doesn't make a lot of sense)...