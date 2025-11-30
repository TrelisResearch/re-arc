# Abstraction Learning for RE-ARC

## Background: DreamCoder Approach

This document tracks our work on implementing DreamCoder-style abstraction learning for the RE-ARC DSL program synthesis system.

### DreamCoder's Three-Phase Learning

DreamCoder (Ellis et al., 2020) learns through wake-sleep cycles:

1. **Wake Phase**: Solve tasks using current library and neural recognition model
2. **Sleep: Abstraction Phase**: Find common patterns in solutions, compress them into new library primitives
3. **Sleep: Dreaming Phase**: Train neural network on replays + fantasies (programs sampled from library)

**Key insight**: The library and neural model bootstrap each other:
- Better library → richer dreams → better neural model
- Better neural model → solves more tasks → more patterns to abstract

### Mapping to RE-ARC

**Wake Phase (already implemented)**:
- Current: `train.py` and `evaluate.py` - transformer generates DSL code from grids
- The transformer IS the recognition model Q(ρ|x)

**Abstraction Phase (this document)**:
- Goal: Find frequent DSL patterns in `verifiers.py` (400 ground-truth programs)
- Compress patterns into new DSL primitives
- Expand vocabulary and retrain

**Dreaming Phase (partially implemented)**:
- Current: Train on 400 hand-coded generators (fixed distribution)
- Missing: Sample programs from learned library to create novel task combinations
- Missing: "Fantasies" that explore compositional space beyond the 400 fixed task types

---

## What We Built

### Scripts in `utils/`

#### 1. `mine_abstractions.py`
**Purpose**: Mine common patterns from verifier programs

**What it does**:
- Extracts all 400 verifier function bodies from `verifiers.py`
- Normalizes code lines (replaces variable names: `x0` → `VAR`, `I` → `INPUT`)
- Finds frequent sequential patterns using sliding window (2-line or 3-line)
- Scores patterns by tokens saved: `frequency × (pattern_length - 1)`
- Outputs ranked patterns to JSON

**Usage**:
```bash
uv run python utils/mine_abstractions.py \
    --min-freq 5 \
    --window-size 2 \
    --max-patterns 30 \
    --output-json utils/patterns_2line.json
```

**Results**:
- Window size 2: Found 244 patterns (min freq ≥5, tokens saved ≥10)
- Window size 3: Found 61 patterns
- Total potential savings: ~13,355 tokens across all programs

#### 2. `apply_abstractions.py`
**Purpose**: Apply abstractions to verifiers WITH VALIDATION

**What it does**:
- Loads patterns from JSON
- **Filters by specificity**: Calculates semantic constraints in normalized pattern
  - Specificity score = count of `INPUT`, function names, constants
  - Higher specificity → safer to abstract
- For each selected pattern:
  - Substitutes pattern in verifier programs
  - **Validates** by running on random test cases
  - Compares original vs substituted program outputs
  - Rolls back if outputs differ
- Accepts abstraction only if success rate ≥ 80% across tasks

**Usage**:
```bash
uv run python utils/apply_abstractions.py \
    --patterns utils/patterns_2line.json \
    --max-abstractions 10
```

**Current status**: Framework built but **validation catches real bugs** (see below)

---

## Key Findings

### 1. Most Patterns Are Too Variable to Abstract

**Example: `learned_mapply_paint`**
- Frequency: 41 occurrences
- Unique forms: 37 (almost every use is different!)
- Normalized pattern:
  ```python
  VAR = mapply(VAR, VAR)  # ← ALL generic VARs
  VAR = paint(VAR, VAR)   # ← Could paint to anything
  ```
- **Problem**: Too context-dependent
  - Sometimes paints to `I` (input grid)
  - Sometimes paints to intermediate result `x20`
  - Variables reference different upstream computations
- **Cannot make a single reusable function**

### 2. Rock-Solid Patterns Exist (High Consistency)

**Example: `learned_leastcolor_ofcolor`**
- Frequency: 18 occurrences
- Unique forms: 3 (ratio 6.0 - very consistent!)
- Normalized pattern:
  ```python
  VAR = leastcolor(INPUT)      # ← INPUT is specific!
  VAR = ofcolor(INPUT, VAR)    # ← Always operates on input grid
  ```
- **Semantic meaning**: "Get pixels of rarest color"
- 16/18 uses are IDENTICAL (just different line numbers)

**Other rock-solid candidates**:
- `frontiers_merge`: Extract borders → merge them (18 occurrences, 5 forms, ratio 3.6)
- `fgpartition_merge`: Partition foreground → merge (17 occurrences, 3 forms, ratio 5.7)
- `asindices_box_toobject`: Create bounding box (10 occurrences, 4 forms, ratio 2.5)

### 3. Specificity Filtering Works

Decision rule we implemented:
```python
specificity_score = count(INPUT) + count(function_names) + count(constants)

if specificity >= 4:
    return True  # High specificity → safe
elif specificity >= 2 and (frequency/unique_forms) >= 2.0:
    return True  # Medium specificity + high consistency → try it
else:
    return False  # Too generic
```

Filters 244 patterns → ~20-30 candidates automatically.

### 4. Validation is CRITICAL

We attempted to apply abstractions and validation caught:

**Bug 1: Hardcoded variable names**
```
❌ 2281f1f4: Validation failed: Execution error: name 'x0' is not defined
```
The primitive function generation uses hardcoded `x0`, `x1` from examples instead of proper parameters.

**Bug 2: Wrong number of arguments**
```
❌ learned_compose_sfilter() takes 1 positional argument but 2 were given
```
Pattern needs more parameters than just `I` because it operates on intermediate variables.

**Without validation**, these bugs would silently corrupt the verifiers!

---

## What Works vs What's Broken

### ✅ Working:
1. Pattern mining - finds meaningful patterns
2. Specificity scoring - filters generic patterns
3. Consistency detection - identifies rock-solid candidates
4. Validation framework - catches bugs before corruption

### ❌ Broken/Missing:

#### Critical: Primitive Function Generation
**Problem**: Creates functions with wrong variable scope

Current buggy code:
```python
def learned_leastcolor_ofcolor(I):
    x0 = leastcolor(I)      # ← Uses hardcoded x0, x1
    x1 = ofcolor(I, x0)     # ← from example task
    return x1               # ← Won't work in other contexts
```

Should generate with fresh variables or understand data flow.

#### Missing: Versioned Output Generation
No code yet to:
- Create `verifiers_v1.py` with substitutions applied
- Create `dsl_learned.py` with new primitive implementations
- Expand `tokenizer_v1.pkl` with new tokens
- Save metadata about what was abstracted

#### Missing: Vocabulary Expansion
When we add `learned_leastcolor_ofcolor` as new token:
1. Add to tokenizer vocabulary (384 → 385 tokens)
2. Expand model embedding matrix
3. Initialize new embedding from component embeddings:
   ```python
   new_embedding = average([emb("leastcolor"), emb("ofcolor")])
   ```
4. Continue training with expanded model

#### Missing: Training Integration
No script to:
- Load model checkpoint
- Expand embedding/output layers
- Train on v1 verifiers
- Compare v0 vs v1 performance

---

## Next Steps to Complete Automation

### Phase 1: Fix Primitive Function Generation (Critical)

**Issue**: Generated functions use wrong variable scope

**Fix needed**:
1. Parse pattern to understand data dependencies
2. Generate functions with proper parameters
3. Handle both INPUT-only patterns and variable-dependent patterns

**Example fix**:
```python
# For INPUT-only pattern:
def learned_leastcolor_ofcolor(I):
    result_0 = leastcolor(I)
    result_1 = ofcolor(I, result_0)
    return result_1

# For variable-dependent pattern (harder):
def learned_lbind_mapply(obj, indices):  # Need to infer parameters!
    temp_0 = lbind(shift, obj)
    temp_1 = mapply(temp_0, indices)
    return temp_1
```

**Simpler alternative**: Only abstract INPUT-only patterns for now (much easier).

### Phase 2: Generate Versioned Files

**2a. Create `verifiers_v1.py`**:
```python
# For each verifier in verifiers.py:
#   - Apply successful substitutions
#   - Write to verifiers_v1.py
with open('verifiers_v1.py', 'w') as f:
    f.write("from dsl import *\n")
    f.write("from dsl_learned import *\n\n")
    for task_id, modified_code in task_modifications.items():
        f.write(f"def verify_{task_id}(I: Grid) -> Grid:\n")
        f.write(modified_code)
```

**2b. Create `dsl_learned.py`**:
```python
# Implementations of abstracted patterns
from dsl import *

def learned_leastcolor_ofcolor(I):
    """Get pixels of rarest color"""
    x0 = leastcolor(I)
    x1 = ofcolor(I, x0)
    return x1

def learned_frontiers_merge(I):
    """Extract and merge all borders"""
    x0 = frontiers(I)
    x1 = merge(x0)
    return x1
```

**2c. Expand tokenizer**:
```python
# Add new primitives to vocabulary
tokenizer = DSLTokenizer()
new_primitives = ['learned_leastcolor_ofcolor', 'learned_frontiers_merge', ...]
for prim in new_primitives:
    tokenizer.add_token(prim)
tokenizer.save('tokenizer_v1.pkl')
```

### Phase 3: Model Expansion and Training

**3a. Expand embedding matrix**:
```python
# Load old model
checkpoint = torch.load('checkpoints/model_best.pt')
old_embeddings = checkpoint['model_state_dict']['embedding.weight']
old_vocab_size = 384

# Create new embedding matrix
new_vocab_size = 384 + len(new_primitives)
new_embeddings = torch.zeros(new_vocab_size, d_model)

# Copy old embeddings
new_embeddings[:old_vocab_size] = old_embeddings

# Initialize new embeddings from components
for i, prim in enumerate(new_primitives):
    # Get component token IDs from pattern
    component_ids = get_component_token_ids(prim, old_tokenizer)
    # Average their embeddings
    new_embeddings[old_vocab_size + i] = old_embeddings[component_ids].mean(dim=0)

# Update model
model.embedding = nn.Embedding.from_pretrained(new_embeddings, freeze=False)
model.fc_out = nn.Linear(d_model, new_vocab_size)
```

**3b. Train on v1 verifiers**:
```python
# Update dataset to use verifiers_v1.py
dataset_v1 = ARCDataset(verifier_file='verifiers_v1.py')

# Train for additional epochs
train(model, dataset_v1, epochs=5000)
```

**3c. Compare performance**:
```python
# Evaluate v0 vs v1 on held-out tasks
results_v0 = evaluate(model_v0, eval_tasks)
results_v1 = evaluate(model_v1, eval_tasks)

# Metrics:
# - Task solve rate
# - Average program length
# - Code readability (use abstractions?)
```

---

## Alternative: Manual Abstraction

Instead of full automation, manually add 3-5 high-value abstractions:

**Candidate abstractions** (validated by inspection):
1. `learned_leastcolor_ofcolor` - Get minority color pixels
2. `learned_frontiers_merge` - Extract and merge borders
3. `learned_fgpartition_merge` - Partition and merge foreground

**Steps**:
1. Manually write functions in `dsl_learned.py`
2. Manually add to tokenizer vocabulary
3. Manually update select verifiers to use them
4. Train and measure impact

**Pros**:
- Much faster (1-2 hours vs 6-8 hours)
- Guaranteed correct implementations
- Tests if abstractions help at all

**Cons**:
- Doesn't test full automation
- Manual maintenance required

---

## Open Questions

### 1. Do Abstractions Actually Help?
We don't know if learned abstractions improve:
- Task solve rate on held-out tasks
- Sample efficiency (solve with less data)
- Generalization (composing learned primitives)

Need experiments to find out!

### 2. When to Abstract?
DreamCoder abstracts after each wake cycle. For us:
- After N training epochs?
- After solving M new tasks?
- Periodically (every 1000 epochs)?

### 3. Fantasy Generation Strategy
How to generate fantasies (programs sampled from library)?

**Option A**: Sample from DSL grammar
```python
program = random.choice(dsl_functions)(random.choice(variables))
```
Problem: Most programs crash or produce nonsense.

**Option B**: Sample from model prior
```python
# Sample without task conditioning
program_tokens = model.sample(max_length=50)  # No input grids
```
Problem: Model might just replay training distribution.

**Option C**: Type-directed sampling
```python
# Sample programs that type-check
program = sample_typed_program(return_type=Grid)
```
More likely to produce valid programs.

### 4. How Many Abstractions?
- DreamCoder adds 5-10 per cycle
- We have 400 training tasks, found 244 patterns
- Start conservatively (3-5 abstractions)?
- Grow library slowly to avoid vocabulary explosion?

---

## References

**DreamCoder Paper**:
- Ellis et al. (2020). "DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning"
- Key sections:
  - Figure 2: Wake-sleep cycle diagram
  - Figure 3: Refactoring and abstraction example (discovering `map`)
  - Section on "Abstraction phase": MDL-based compression

**Our Implementation Notes**:
- Simplified: No refactoring (DreamCoder's version spaces)
- Simplified: Pattern matching instead of semantic equivalence
- Added: Validation via execution testing
- Added: Specificity-based filtering

---

## Current Status

**Built**: Pattern mining + validation framework
**Working**: Can identify rock-solid abstraction candidates
**Blocked**: Primitive function generation has bugs
**Decision needed**: Fix automation vs manual abstraction vs defer?

**Files**:
- `utils/mine_abstractions.py` - ✅ Working
- `utils/apply_abstractions.py` - ⚠️ Broken (validation works, generation doesn't)
- `utils/patterns_2line.json` - ✅ Contains 244 mined patterns
- `abstraction-todo.md` - This file

**Next immediate step**: Fix primitive function generation or switch to manual approach.
