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

## TODO
- [x] **Review code overall**
    - [x] Review tokenizer
    - [x] Review data prep
    - [x] Review training code
- [ ] **Review architecture** Ensure choices make sense for the application at hand.
- [x] **Guided Decoding**: Constrain generation. UNCLEAR IF THIS HELPS.
- [ ] **Grammar-Constrained Training**: apply the same token-masking rules during teacher forcing so the model learns the DSL grammar, not just enforced at inference.
- [ ] **Static Analysis Optimization**: Filter programs with `ast.parse` and symbol checks before expensive execution.
- [ ] Do predicted programs include upper and lower bound parameters? if so, that could be powerful for data augmentation.
- [ ] **BPE Tokenization**: Implement Byte Pair Encoding to compound frequent DSL idioms. Or not Byte Pair but merging tokens for performance speed-ups.
- [ ] **Evaluation Script**: Create `evaluate.py` to test on `arc-agi_rearc_challenges.json.gz`.
- [ ] **Search Strategy**: Implement Beam Search or Temperature Sampling.

## Performance Notes
============================================================
BASELINE TRANSFER RESULTS - i.e. applying train verifiers to eval tasks
============================================================
Total eval tasks tested: 400
Solved by train verifiers: 36
Success rate: 9.0%

Successful transfers:
  070dd51e ← 40853293
  0c9aba6e ← 1b2d62fb
  1a2e2828 ← 44f52bb0
  27f8ce4f ← c3e719e8
  2c0b0aff ← 8efcae92
  358ba94e ← 1cf80156
  47996f11 ← 3631a71a
  506d28a5 ← ce4f8723
  50a16a69 ← caa06a1f
  5b6cbef5 ← 007bbfb7
  60c09cac ← b91ae062
  7039b2d7 ← 1190e5a7
  73182012 ← 2013d3e2
  67b4a34d ← dc0a314f
  73ccf9c2 ← be94b721
  7bb29440 ← 6ecd11f4
  8597cfd7 ← 995c5fa3
  981571dc ← 3631a71a
  9a4bb226 ← 8efcae92
  9ddd00f0 ← b8825c91
  aa18de87 ← a699fb00
  af22c60d ← 3631a71a
  bbb1b8b6 ← e6721834
  bf699163 ← e50d258f
  c663677b ← 0dfd9992
  c7d4e6ad ← c9f8e694
  cd3c21df ← 0b148d64
  d56f2372 ← 72ca375d
  e1baa8a4 ← 90c28cc7
  e66aafb8 ← 9ecd008a
  e7a25a18 ← 6b9890af
  e95e3d8e ← 0dfd9992
  ea9794b1 ← 75b8110e
  ea959feb ← c3f564a4
  f823c43c ← ff805c23
  f4081712 ← dc0a314f
============================================================

Using a neurally-guided approach to writing DSLs:
  Note that the sampling approach below assumes oracle correctness (which in practise could be checked on the train examples).

20k model on AA1 eval (naive): 3.5%
+ constrained de-coding + 64 samples: 5.5%
============================================================
EVALUATION RESULTS (eval mode)
============================================================
Total tasks:     400
Tasks solved:      23 /  400 (5.8%)
Total examples:  419
Syntax valid:     405 /  419 (96.7%)
Runtime success:  275 /  419 (65.6%)
Correct:           23 /  419 (5.5%)
============================================================