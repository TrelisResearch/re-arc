"""
Evaluation script for RE-ARC trained models.

Supports two modes:
1. Training tasks: Evaluate on synthetic examples from generators
2. Evaluation tasks: Evaluate on real ARC test examples from JSON files

Usage:
    # Evaluate on training tasks (greedy decoding)
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode train --num-tasks 50

    # Evaluate on evaluation tasks
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval

    # Use parallel sampling (generates N diverse solutions with top-k=50 by default)
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval --num-samples 10 --temperature 0.8

    # Use top-p (nucleus) sampling instead of top-k
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval --num-samples 10 --top-p 0.95 --top-k 0

    # Adjust top-k value
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval --num-samples 10 --top-k 100

    # Use constrained decoding (grammar-aware token masking)
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode train --constrained-decoding

    # Combine sampling with constraints (uses top-k=50 by default)
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval --num-samples 10 --constrained-decoding

    # Measure entropy
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode train --measure-entropy

    # Limit number of tasks (works for both modes)
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval --num-tasks 10
"""

import argparse
import torch
import torch.nn.functional as F
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time

from train import DecoderOnlyDSLTransformer, generate_square_subsequent_mask, execute_and_score
from tokenizer import DSLTokenizer
from dataset import ARCDataset
import generators
import verifiers
import dsl
from utils import DSLConstrainedDecoder


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = DSLTokenizer()

    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', tokenizer.vocab_size)
    d_model = config.get('d_model', 512)
    n_head = config.get('n_head', 8)
    num_layers = config.get('num_layers', 1)
    num_recursions = config.get('num_recursions', 16)

    model = DecoderOnlyDSLTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        num_recursions=num_recursions
    ).to(device)

    # Handle torch.compile prefix (_orig_mod.)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    epoch = checkpoint.get('epoch', -1)
    step = checkpoint.get('global_step', -1)
    print(f"Loaded model from epoch {epoch}, step {step}")

    return model, tokenizer


def greedy_generate(model, src_tokens, tokenizer, device, max_len=2048,
                    constraint: DSLConstrainedDecoder = None, debug=False):
    """Greedy decoding (existing method)."""
    model.eval()
    constraint_state = constraint.new_state() if constraint else None

    start_time = time.time() if debug else None

    with torch.no_grad():
        curr_tokens = src_tokens.unsqueeze(0).to(device)
        past_key_values = None
        generated = []

        for step in range(max_len):
            step_start = time.time() if debug and constraint else None

            # Debug: Print progress every 50 tokens when constraint is enabled
            if debug and constraint and step > 0 and step % 50 == 0:
                print(f"  Step {step}: generated {len(generated)} tokens so far...")
            if past_key_values is None:
                seq_len = curr_tokens.size(1)
                causal_mask = generate_square_subsequent_mask(seq_len).to(device)
                padding_mask = (curr_tokens == tokenizer.pad_token_id)
                logits, past_key_values = model(
                    curr_tokens,
                    mask=causal_mask,
                    padding_mask=padding_mask,
                    use_cache=True
                )
            else:
                past_seq_len = past_key_values[0][0].size(2)
                position_ids = torch.full(
                    (1, curr_tokens.size(1)),
                    past_seq_len,
                    dtype=torch.long,
                    device=device
                )
                logits, past_key_values = model(
                    curr_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=position_ids
                )

            step_logits = logits[:, -1, :]

            # Debug: Check if model wants EOS but constraint blocks it
            if constraint_state and constraint:
                original_logits = step_logits.clone()

                # Profile constraint application
                constraint_start = time.time() if debug else None
                step_logits = constraint.apply(step_logits, constraint_state)
                if constraint_start:
                    constraint_time = time.time() - constraint_start
                    if constraint_time > 0.5:
                        print(f"  WARNING: Constraint.apply() at step {step} took {constraint_time:.2f}s!")
                        print(f"    State: line_start={constraint_state.line_start}, paren_depth={constraint_state.paren_depth}")

                # Check if EOS was the top choice before constraining
                top_unconstrained = torch.argmax(original_logits, dim=-1)
                if top_unconstrained.item() == tokenizer.eos_token_id:
                    if step_logits[0, tokenizer.eos_token_id].item() == float('-inf'):
                        print(f"WARNING: Model wants EOS at step {step} but constraint blocks it!")
                        print(f"  State: line_start={constraint_state.line_start}, "
                              f"paren_depth={constraint_state.paren_depth}, "
                              f"need_value={constraint_state.need_value}, "
                              f"can_terminate={constraint_state.can_terminate()}")

            next_token = torch.argmax(step_logits, dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated.append(next_token.item())
            curr_tokens = next_token.unsqueeze(1)  # [batch] -> [batch, 1]
            if constraint_state:
                constraint_state.update(next_token.item())

            # Debug: Detect slow steps
            if step_start and step > 0:
                step_time = time.time() - step_start
                if step_time > 0.5:  # Individual step taking >0.5s
                    print(f"  WARNING: Step {step} took {step_time:.2f}s! (context_len={len(generated)+src_tokens.size(0)})")

        # Debug: Check if we hit max_len
        if len(generated) >= max_len - 1:
            print(f"WARNING: Hit max_len ({max_len}) without generating EOS!")
            if constraint_state:
                print(f"  Final state: line_start={constraint_state.line_start}, "
                      f"paren_depth={constraint_state.paren_depth}, "
                      f"need_value={constraint_state.need_value}, "
                      f"can_terminate={constraint_state.can_terminate()}")
            print(f"  Generated tokens: {len(generated)}")

        if debug and start_time:
            elapsed = time.time() - start_time
            if elapsed > 2.0:  # Only report slow generations
                print(f"SLOW GENERATION: {elapsed:.1f}s for {len(generated)} tokens ({len(generated)/elapsed:.1f} tok/s)")

        return tokenizer.decode(generated)


def parallel_sample_generate(model, src_tokens, tokenizer, device, num_samples=10,
                            max_len=2048, temperature=0.8, top_k=50, top_p=1.0,
                            constraint: DSLConstrainedDecoder = None, debug=False):
    """
    Parallel sampling with temperature, top-k, and top-p filtering.

    Generates num_samples programs in parallel using batched inference.
    Much more efficient than beam search and provides better diversity.

    Args:
        top_k: If > 0, only sample from top k tokens (0 = disabled)
        top_p: If < 1.0, nucleus sampling - sample from smallest set with cumulative prob >= p

    Returns list of decoded strings (one per sample).
    """
    model.eval()
    start_time = time.time() if debug else None

    with torch.no_grad():
        # Batch the source tokens - all samples start with same prefix
        src_batch = src_tokens.unsqueeze(0).repeat(num_samples, 1).to(device)  # [num_samples, seq_len]

        # Each sample gets its own constraint state if using constraints
        constraint_states = [constraint.new_state() for _ in range(num_samples)] if constraint else None

        past_key_values = None
        generated = [[] for _ in range(num_samples)]  # Track generated tokens per sample
        finished = [False] * num_samples

        for step in range(max_len):
            if all(finished):
                break

            step_start = time.time() if debug else None

            # Debug: Print progress every 50 tokens
            if debug and step > 0 and step % 50 == 0:
                avg_generated = sum(len(g) for g in generated) / num_samples
                print(f"  Step {step}: avg {avg_generated:.1f} tokens generated across {num_samples} samples...")

            # Forward pass for all samples in parallel
            fwd_start = time.time() if debug else None
            if past_key_values is None:
                seq_len = src_batch.size(1)
                causal_mask = generate_square_subsequent_mask(seq_len).to(device)
                padding_mask = (src_batch == tokenizer.pad_token_id)
                logits, past_key_values = model(
                    src_batch,
                    mask=causal_mask,
                    padding_mask=padding_mask,
                    use_cache=True
                )
                step_logits = logits[:, -1, :]  # [num_samples, vocab_size]
            else:
                past_seq_len = past_key_values[0][0].size(2)
                position_ids = torch.full(
                    (num_samples, 1),
                    past_seq_len,
                    dtype=torch.long,
                    device=device
                )
                logits, past_key_values = model(
                    src_batch,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=position_ids
                )
                step_logits = logits[:, -1, :]  # [num_samples, vocab_size]

            if fwd_start and debug:
                fwd_time = time.time() - fwd_start
                if fwd_time > 0.3:
                    print(f"  WARNING: Forward pass at step {step} took {fwd_time:.2f}s!")

            # Apply constraints per sample if needed
            if constraint_states:
                constraint_start = time.time() if debug else None
                for i in range(num_samples):
                    if not finished[i]:
                        step_logits[i] = constraint.apply(step_logits[i].unsqueeze(0), constraint_states[i]).squeeze(0)
                if constraint_start and debug:
                    constraint_time = time.time() - constraint_start
                    if constraint_time > 0.3:
                        print(f"  WARNING: Constraint application at step {step} took {constraint_time:.2f}s across {sum(1 for f in finished if not f)} active samples!")

            # Apply temperature
            step_logits = step_logits / temperature

            # Top-k filtering
            if top_k > 0:
                filter_start = time.time() if debug else None
                top_k_vals, top_k_indices = torch.topk(step_logits, min(top_k, step_logits.size(-1)), dim=-1)
                # Set all non-top-k logits to -inf
                mask = torch.full_like(step_logits, float('-inf'))
                mask.scatter_(-1, top_k_indices, top_k_vals)
                step_logits = mask
                if filter_start and debug:
                    filter_time = time.time() - filter_start
                    if filter_time > 0.3:
                        print(f"  WARNING: Top-k filtering at step {step} took {filter_time:.2f}s!")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                filter_start = time.time() if debug else None
                sorted_logits, sorted_indices = torch.sort(step_logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter back to original indexing
                for i in range(num_samples):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    step_logits[i, indices_to_remove] = float('-inf')

                if filter_start and debug:
                    filter_time = time.time() - filter_start
                    if filter_time > 0.3:
                        print(f"  WARNING: Top-p filtering at step {step} took {filter_time:.2f}s!")

            # Sample from filtered distribution
            sample_start = time.time() if debug else None
            probs = F.softmax(step_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # [num_samples]
            if sample_start and debug:
                sample_time = time.time() - sample_start
                if sample_time > 0.3:
                    print(f"  WARNING: Sampling (softmax + multinomial) at step {step} took {sample_time:.2f}s!")

            # Update generated tokens and states
            for i in range(num_samples):
                if not finished[i]:
                    token_id = next_tokens[i].item()

                    if token_id == tokenizer.eos_token_id:
                        finished[i] = True
                    else:
                        generated[i].append(token_id)
                        if constraint_states:
                            constraint_states[i].update(token_id)

            # Prepare next input
            src_batch = next_tokens.unsqueeze(1)  # [num_samples, 1]

            # Debug: Detect slow steps
            if step_start and step > 0:
                step_time = time.time() - step_start
                if step_time > 0.5:  # Individual step taking >0.5s
                    avg_context = src_tokens.size(0) + sum(len(g) for g in generated) / num_samples
                    print(f"  WARNING: Step {step} took {step_time:.2f}s! (avg_context_len={avg_context:.0f})")

        # Decode all samples
        results = []
        for gen_tokens in generated:
            decoded = tokenizer.decode(gen_tokens)
            results.append(decoded)

        if debug and start_time:
            elapsed = time.time() - start_time
            avg_tokens = sum(len(g) for g in generated) / num_samples
            if elapsed > 2.0:  # Only report slow generations
                print(f"SLOW GENERATION: {elapsed:.1f}s for avg {avg_tokens:.1f} tokens across {num_samples} samples ({avg_tokens*num_samples/elapsed:.1f} tok/s)")

        return results


def measure_entropy(model, src_tokens, tokenizer, device, num_steps=10,
                    constraint: DSLConstrainedDecoder = None):
    """
    Measure average per-token entropy during generation.

    Returns:
        avg_entropy: Average entropy in bits
        normalized_entropy: Entropy normalized by max possible (uniform distribution)
    """
    model.eval()
    entropies = []

    constraint_state = constraint.new_state() if constraint else None

    with torch.no_grad():
        curr_tokens = src_tokens.unsqueeze(0).to(device)
        past_key_values = None

        for step in range(num_steps):
            if past_key_values is None:
                seq_len = curr_tokens.size(1)
                causal_mask = generate_square_subsequent_mask(seq_len).to(device)
                padding_mask = (curr_tokens == tokenizer.pad_token_id)
                logits, past_key_values = model(
                    curr_tokens,
                    mask=causal_mask,
                    padding_mask=padding_mask,
                    use_cache=True
                )
            else:
                past_seq_len = past_key_values[0][0].size(2)
                position_ids = torch.full(
                    (1, curr_tokens.size(1)),
                    past_seq_len,
                    dtype=torch.long,
                    device=device
                )
                logits, past_key_values = model(
                    curr_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=position_ids
                )

            # Compute entropy: H = -sum(p * log(p))
            step_logits = logits[0, -1, :]
            if constraint_state and constraint:
                step_logits = constraint.apply(step_logits, constraint_state)

            probs = F.softmax(step_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies.append(entropy)

            # Sample next token for continued generation
            next_token = torch.argmax(step_logits, dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

            curr_tokens = next_token.unsqueeze(1)  # [batch] -> [batch, 1]
            if constraint_state:
                constraint_state.update(next_token.item())

    avg_entropy = np.mean(entropies) if entropies else 0.0
    max_entropy = np.log(tokenizer.vocab_size)
    normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.0

    return avg_entropy, normalized_entropy


def evaluate_on_training_tasks(model, tokenizer, device, num_tasks=400,
                               num_samples=None, temperature=0.8, top_k=50, top_p=1.0,
                               measure_entropy_flag=False,
                               constraint: DSLConstrainedDecoder = None,
                               max_gen_len=2048):
    """Evaluate on synthetic training tasks."""
    dataset = ARCDataset(diff_lb=0.0, diff_ub=0.5)

    results = {
        'total': 0,
        'syntax_valid': 0,
        'runtime_success': 0,
        'correct': 0,
        'entropy_samples': []
    }

    # Sample random tasks
    task_indices = np.random.choice(len(dataset), size=min(num_tasks, len(dataset)), replace=False)

    for idx in tqdm(task_indices, desc="Evaluating training tasks"):
        task_start = time.time()
        task_id = dataset.tasks[idx]
        generator = getattr(generators, f'generate_{task_id}')
        verifier = getattr(verifiers, f'verify_{task_id}', None)

        if verifier is None:
            continue

        # Generate fresh example and validate like training does
        input_grid = None
        output_grid = None
        for attempt in range(5):
            try:
                example = generator(0.0, 0.5)
                input_grid = example['input']
                output_grid = example['output']

                # Validate like training does
                if input_grid == output_grid:
                    continue
                if verifier(input_grid) != output_grid:
                    continue

                break  # Valid example found
            except:
                continue

        if input_grid is None or output_grid is None:
            continue

        # Encode input
        input_tokens_list = tokenizer.encode_grid(input_grid)
        output_tokens_list = tokenizer.encode_grid(output_grid)
        src_ids = [tokenizer.bos_token_id] + input_tokens_list + \
                  [tokenizer.sep_token_id] + output_tokens_list + \
                  [tokenizer.eos_token_id]
        src_tensor = torch.tensor(src_ids)

        # Measure entropy if requested
        if measure_entropy_flag and len(results['entropy_samples']) < 20:
            avg_ent, norm_ent = measure_entropy(
                model, src_tensor, tokenizer, device, constraint=constraint
            )
            results['entropy_samples'].append({
                'task_id': task_id,
                'avg_entropy': avg_ent,
                'normalized_entropy': norm_ent
            })

        # Generate
        gen_start = time.time()
        if num_samples and num_samples > 1:
            # Parallel sampling - generate multiple diverse solutions
            sample_results = parallel_sample_generate(
                model, src_tensor, tokenizer, device,
                num_samples=num_samples,
                max_len=max_gen_len,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                constraint=constraint,
                debug=True
            )
            gen_time = time.time() - gen_start

            # Try each sample until one succeeds
            exec_start = time.time()
            for generated_code in sample_results:
                syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
                if correct:
                    results['correct'] += 1
                    results['runtime_success'] += 1
                    results['syntax_valid'] += 1
                    results['total'] += 1
                    break
            else:
                # No sample succeeded - count first sample's metrics
                generated_code = sample_results[0]
                syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
                if syntax_ok:
                    results['syntax_valid'] += 1
                if runs_ok:
                    results['runtime_success'] += 1
                results['total'] += 1
            exec_time = time.time() - exec_start
        else:
            generated_code = greedy_generate(
                model, src_tensor, tokenizer, device, max_len=max_gen_len,
                constraint=constraint, debug=(constraint is not None)
            )
            gen_time = time.time() - gen_start

            exec_start = time.time()
            if constraint:
                print(f"  Executing generated code ({len(generated_code)} chars)...")
            syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
            exec_time = time.time() - exec_start
            if constraint and exec_time > 0.5:
                print(f"  Execution took {exec_time:.2f}s")

            if syntax_ok:
                results['syntax_valid'] += 1
            if runs_ok:
                results['runtime_success'] += 1
            if correct:
                results['correct'] += 1
            results['total'] += 1

        task_time = time.time() - task_start
        if task_time > 5.0:  # Report slow tasks
            print(f"\nSLOW TASK {task_id}: {task_time:.1f}s total (gen: {gen_time:.1f}s, exec: {exec_time:.1f}s)")

    return results


def evaluate_on_eval_tasks(model, tokenizer, device,
                           challenges_path='data/arc-agi_evaluation_challenges.json',
                           solutions_path='data/arc-agi_evaluation_solutions.json',
                           num_tasks=None, num_samples=None, temperature=0.8,
                           top_k=50, top_p=1.0,
                           measure_entropy_flag=False,
                           constraint: DSLConstrainedDecoder = None,
                           max_gen_len=2048):
    """Evaluate on real ARC test examples using challenges and solutions files."""

    # Load challenges and solutions
    try:
        with open(challenges_path, 'r') as f:
            challenges = json.load(f)
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None

    results = {
        'total_tasks': 0,
        'total_examples': 0,
        'syntax_valid': 0,
        'runtime_success': 0,
        'correct': 0,
        'tasks_solved': 0,
        'entropy_samples': []
    }

    task_ids = list(challenges.keys())
    if num_tasks is not None:
        task_ids = task_ids[:min(num_tasks, len(task_ids))]

    for task_id in tqdm(task_ids, desc="Evaluating test tasks"):
        task_start = time.time()
        if task_id not in solutions:
            continue

        task_solved = False
        test_examples = challenges[task_id]['test']
        test_solutions = solutions[task_id]

        # Evaluate each test example
        for example, solution in zip(test_examples, test_solutions):
            input_grid = tuple(tuple(row) for row in example['input'])
            output_grid = tuple(tuple(row) for row in solution)

            # Encode
            input_tokens_list = tokenizer.encode_grid(input_grid)
            output_tokens_list = tokenizer.encode_grid(output_grid)
            src_ids = [tokenizer.bos_token_id] + input_tokens_list + \
                      [tokenizer.sep_token_id] + output_tokens_list + \
                      [tokenizer.eos_token_id]
            src_tensor = torch.tensor(src_ids)

            # Measure entropy
            if measure_entropy_flag and len(results['entropy_samples']) < 20:
                avg_ent, norm_ent = measure_entropy(
                    model, src_tensor, tokenizer, device, constraint=constraint
                )
                results['entropy_samples'].append({
                    'task_id': task_id,
                    'avg_entropy': avg_ent,
                    'normalized_entropy': norm_ent
                })

            # Generate
            gen_start = time.time()
            if num_samples and num_samples > 1:
                # Parallel sampling - generate multiple diverse solutions
                sample_results = parallel_sample_generate(
                    model, src_tensor, tokenizer, device,
                    num_samples=num_samples,
                    max_len=max_gen_len,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    constraint=constraint,
                    debug=True
                )
                gen_time = time.time() - gen_start

                exec_start = time.time()
                for generated_code in sample_results:
                    syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
                    if correct:
                        results['correct'] += 1
                        results['runtime_success'] += 1
                        results['syntax_valid'] += 1
                        task_solved = True
                        break
                else:
                    # No sample succeeded - count first sample's metrics
                    generated_code = sample_results[0]
                    syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
                    if syntax_ok:
                        results['syntax_valid'] += 1
                    if runs_ok:
                        results['runtime_success'] += 1
                exec_time = time.time() - exec_start
            else:
                generated_code = greedy_generate(
                    model, src_tensor, tokenizer, device, max_len=max_gen_len,
                    constraint=constraint, debug=(constraint is not None)
                )
                gen_time = time.time() - gen_start

                exec_start = time.time()
                if constraint:
                    print(f"  Executing generated code ({len(generated_code)} chars)...")
                syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
                exec_time = time.time() - exec_start
                if constraint and exec_time > 0.5:
                    print(f"  Execution took {exec_time:.2f}s")

                if syntax_ok:
                    results['syntax_valid'] += 1
                if runs_ok:
                    results['runtime_success'] += 1
                if correct:
                    results['correct'] += 1
                    task_solved = True

            results['total_examples'] += 1

        task_time = time.time() - task_start
        if task_time > 5.0:  # Report slow tasks
            print(f"\nSLOW TASK {task_id}: {task_time:.1f}s total (gen: {gen_time:.1f}s, exec: {exec_time:.1f}s)")

        if task_solved:
            results['tasks_solved'] += 1
        results['total_tasks'] += 1

    return results


def print_results(results, mode):
    """Print evaluation results."""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS ({mode} mode)")
    print("="*60)

    if mode == 'train':
        total = results['total']
        if total > 0:
            print(f"Total examples: {total}")
            print(f"Syntax valid:   {results['syntax_valid']:4d} / {total:4d} ({100*results['syntax_valid']/total:.1f}%)")
            print(f"Runtime success:{results['runtime_success']:4d} / {total:4d} ({100*results['runtime_success']/total:.1f}%)")
            print(f"Correct:        {results['correct']:4d} / {total:4d} ({100*results['correct']/total:.1f}%)")
    else:
        total_ex = results['total_examples']
        total_tasks = results['total_tasks']
        if total_ex > 0:
            print(f"Total tasks:     {total_tasks}")
            print(f"Tasks solved:    {results['tasks_solved']:4d} / {total_tasks:4d} ({100*results['tasks_solved']/total_tasks:.1f}%)")
            print(f"Total examples:  {total_ex}")
            print(f"Syntax valid:    {results['syntax_valid']:4d} / {total_ex:4d} ({100*results['syntax_valid']/total_ex:.1f}%)")
            print(f"Runtime success: {results['runtime_success']:4d} / {total_ex:4d} ({100*results['runtime_success']/total_ex:.1f}%)")
            print(f"Correct:         {results['correct']:4d} / {total_ex:4d} ({100*results['correct']/total_ex:.1f}%)")

    if results['entropy_samples']:
        print("\n" + "-"*60)
        print("ENTROPY ANALYSIS (sampled examples)")
        print("-"*60)
        avg_ent = np.mean([s['avg_entropy'] for s in results['entropy_samples']])
        norm_ent = np.mean([s['normalized_entropy'] for s in results['entropy_samples']])
        print(f"Average entropy:    {avg_ent:.3f} bits/token")
        print(f"Normalized entropy: {100*norm_ent:.1f}% of maximum")
        print(f"Samples analyzed:   {len(results['entropy_samples'])}")

        if norm_ent < 0.1:
            print("\n→ Model is VERY confident (low entropy)")
            print("  Beam search may offer limited benefit over greedy decoding")
        elif norm_ent < 0.3:
            print("\n→ Model is moderately confident")
            print("  Beam search K=5-10 recommended")
        else:
            print("\n→ Model has HIGH uncertainty")
            print("  Beam search K=10-20 or sampling recommended")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RE-ARC model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint (.pt file)")
    parser.add_argument("--mode", type=str, choices=['train', 'eval'], required=True,
                       help="Evaluation mode: 'train' (synthetic) or 'eval' (real ARC tasks)")
    parser.add_argument("--challenges", type=str, default="data/arc-agi_evaluation_challenges.json",
                       help="Path to challenges JSON file (for eval mode)")
    parser.add_argument("--solutions", type=str, default="data/arc-agi_evaluation_solutions.json",
                       help="Path to solutions JSON file (for eval mode)")
    parser.add_argument("--num-tasks", type=int, default=None,
                        help="Limit number of tasks (train defaults to 100; eval uses all)")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to generate in parallel (default: greedy decoding)")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature (only used with --num-samples)")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k filtering: only sample from top k tokens (default: 50, use 0 to disable)")
    parser.add_argument("--top-p", type=float, default=1.0,
                       help="Top-p (nucleus) filtering: sample from smallest set with cumulative prob >= p (default: disabled, use 0.95 for nucleus sampling)")
    parser.add_argument("--measure-entropy", action="store_true",
                       help="Measure model entropy on sample of examples")
    parser.add_argument("--constrained-decoding", action="store_true",
                        help="Enable grammar-aware token masking during decoding")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (auto-detects if not specified)")
    parser.add_argument("--max-gen-len", type=int, default=2048,
                       help="Maximum generation length")

    args = parser.parse_args()

    # Setup
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")

    # Load model
    model, tokenizer = load_model(args.checkpoint, device)
    constraint = DSLConstrainedDecoder(tokenizer) if args.constrained_decoding else None

    # Evaluate
    if args.mode == 'train':
        num_tasks = args.num_tasks if args.num_tasks is not None else 100
        results = evaluate_on_training_tasks(
            model, tokenizer, device,
            num_tasks=num_tasks,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            measure_entropy_flag=args.measure_entropy,
            constraint=constraint,
            max_gen_len=args.max_gen_len
        )
    else:
        results = evaluate_on_eval_tasks(
            model, tokenizer, device,
            challenges_path=args.challenges,
            solutions_path=args.solutions,
            num_tasks=args.num_tasks,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            measure_entropy_flag=args.measure_entropy,
            constraint=constraint,
            max_gen_len=args.max_gen_len
        )

    # Print results
    if results:
        print_results(results, args.mode)
    else:
        print("Evaluation failed or no results.")


if __name__ == "__main__":
    main()
