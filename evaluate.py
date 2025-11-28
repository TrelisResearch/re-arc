"""
Evaluation script for RE-ARC trained models.

Supports two modes:
1. Training tasks: Evaluate on synthetic examples from generators
2. Evaluation tasks: Evaluate on real ARC test examples from JSON files

Usage:
    # Evaluate on training tasks
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode train --num-tasks 50

    # Evaluate on evaluation tasks
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval

    # Use beam search
    uv run evaluate.py --checkpoint checkpoints/model.pt --mode eval --beam-width 10

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

from train import DecoderOnlyDSLTransformer, generate_square_subsequent_mask, execute_and_score
from tokenizer import DSLTokenizer
from dataset import ARCDataset
import generators
import verifiers
import dsl


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


def greedy_generate(model, src_tokens, tokenizer, device, max_len=2048):
    """Greedy decoding (existing method)."""
    model.eval()

    with torch.no_grad():
        curr_tokens = src_tokens.unsqueeze(0).to(device)
        past_key_values = None
        generated = []

        for _ in range(max_len):
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

            next_token = torch.argmax(logits[:, -1, :], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated.append(next_token.item())
            curr_tokens = next_token.unsqueeze(1)  # [batch] -> [batch, 1]

        return tokenizer.decode(generated)


def beam_search_generate(model, src_tokens, tokenizer, device, beam_width=10, max_len=2048):
    """
    Beam search decoding.

    Returns list of (decoded_string, log_probability) tuples, sorted by probability.
    """
    model.eval()

    with torch.no_grad():
        # Initialize beam: (prefix_tokens, log_prob, past_kv, finished)
        src_batch = src_tokens.unsqueeze(0).to(device)
        beams = [(src_batch, 0.0, None, False)]

        for step in range(max_len):
            if all(finished for _, _, _, finished in beams):
                break

            candidates = []

            for prefix, log_prob, past_kv, finished in beams:
                if finished:
                    candidates.append((prefix, log_prob, past_kv, True))
                    continue

                # Forward pass
                if past_kv is None:
                    seq_len = prefix.size(1)
                    causal_mask = generate_square_subsequent_mask(seq_len).to(device)
                    padding_mask = (prefix == tokenizer.pad_token_id)
                    logits, new_kv = model(
                        prefix,
                        mask=causal_mask,
                        padding_mask=padding_mask,
                        use_cache=True
                    )
                else:
                    past_seq_len = past_kv[0][0].size(2)
                    position_ids = torch.full(
                        (1, prefix.size(1)),
                        past_seq_len,
                        dtype=torch.long,
                        device=device
                    )
                    logits, new_kv = model(
                        prefix,
                        past_key_values=past_kv,
                        use_cache=True,
                        position_ids=position_ids
                    )

                # Get top-K tokens (fixing for beam search too)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                top_log_probs, top_indices = torch.topk(log_probs, k=beam_width)

                for token_log_prob, token_id in zip(top_log_probs, top_indices):
                    new_token = token_id.unsqueeze(0).unsqueeze(1)  # [1, 1]
                    new_prefix = torch.cat([prefix, new_token], dim=1) if past_kv is None else new_token
                    new_log_prob = log_prob + token_log_prob.item()
                    is_finished = (token_id.item() == tokenizer.eos_token_id)

                    candidates.append((new_prefix, new_log_prob, new_kv, is_finished))

            # Keep top beam_width candidates
            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

        # Decode all beams
        results = []
        for prefix, log_prob, _, _ in beams:
            # Extract generated tokens (skip source prefix)
            gen_tokens = prefix[0, src_tokens.size(0):].cpu().tolist()
            decoded = tokenizer.decode(gen_tokens)
            results.append((decoded, log_prob))

        return results


def measure_entropy(model, src_tokens, tokenizer, device, num_steps=10):
    """
    Measure average per-token entropy during generation.

    Returns:
        avg_entropy: Average entropy in bits
        normalized_entropy: Entropy normalized by max possible (uniform distribution)
    """
    model.eval()
    entropies = []

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
            probs = F.softmax(logits[0, -1, :], dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            entropies.append(entropy)

            # Sample next token for continued generation
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

            curr_tokens = next_token.unsqueeze(1)  # [batch] -> [batch, 1]

    avg_entropy = np.mean(entropies) if entropies else 0.0
    max_entropy = np.log(tokenizer.vocab_size)
    normalized_entropy = avg_entropy / max_entropy if max_entropy > 0 else 0.0

    return avg_entropy, normalized_entropy


def evaluate_on_training_tasks(model, tokenizer, device, num_tasks=400,
                               beam_width=None, measure_entropy_flag=False):
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
            avg_ent, norm_ent = measure_entropy(model, src_tensor, tokenizer, device)
            results['entropy_samples'].append({
                'task_id': task_id,
                'avg_entropy': avg_ent,
                'normalized_entropy': norm_ent
            })

        # Generate
        if beam_width and beam_width > 1:
            beam_results = beam_search_generate(model, src_tensor, tokenizer, device, beam_width)
            # Try each beam candidate
            for generated_code, log_prob in beam_results:
                syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
                if correct:
                    results['correct'] += 1
                    results['runtime_success'] += 1
                    results['syntax_valid'] += 1
                    results['total'] += 1
                    break
            else:
                # No beam succeeded - count best beam's metrics
                generated_code, _ = beam_results[0]
                syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
                if syntax_ok:
                    results['syntax_valid'] += 1
                if runs_ok:
                    results['runtime_success'] += 1
                results['total'] += 1
        else:
            generated_code = greedy_generate(model, src_tensor, tokenizer, device)
            syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)

            if syntax_ok:
                results['syntax_valid'] += 1
            if runs_ok:
                results['runtime_success'] += 1
            if correct:
                results['correct'] += 1
            results['total'] += 1

    return results


def evaluate_on_eval_tasks(model, tokenizer, device,
                           challenges_path='data/arc-agi_evaluation_challenges.json',
                           solutions_path='data/arc-agi_evaluation_solutions.json',
                           num_tasks=None, beam_width=None,
                           measure_entropy_flag=False):
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
                avg_ent, norm_ent = measure_entropy(model, src_tensor, tokenizer, device)
                results['entropy_samples'].append({
                    'task_id': task_id,
                    'avg_entropy': avg_ent,
                    'normalized_entropy': norm_ent
                })

            # Generate
            if beam_width and beam_width > 1:
                beam_results = beam_search_generate(model, src_tensor, tokenizer, device, beam_width)
                for generated_code, log_prob in beam_results:
                    syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
                    if correct:
                        results['correct'] += 1
                        results['runtime_success'] += 1
                        results['syntax_valid'] += 1
                        task_solved = True
                        break
                else:
                    generated_code, _ = beam_results[0]
                    syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)
                    if syntax_ok:
                        results['syntax_valid'] += 1
                    if runs_ok:
                        results['runtime_success'] += 1
            else:
                generated_code = greedy_generate(model, src_tensor, tokenizer, device)
                syntax_ok, runs_ok, correct = execute_and_score(generated_code, input_grid, output_grid)

                if syntax_ok:
                    results['syntax_valid'] += 1
                if runs_ok:
                    results['runtime_success'] += 1
                if correct:
                    results['correct'] += 1
                    task_solved = True

            results['total_examples'] += 1

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
    parser.add_argument("--beam-width", type=int, default=None,
                       help="Beam width for beam search (default: greedy decoding)")
    parser.add_argument("--measure-entropy", action="store_true",
                       help="Measure model entropy on sample of examples")
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

    # Evaluate
    if args.mode == 'train':
        num_tasks = args.num_tasks if args.num_tasks is not None else 100
        results = evaluate_on_training_tasks(
            model, tokenizer, device,
            num_tasks=num_tasks,
            beam_width=args.beam_width,
            measure_entropy_flag=args.measure_entropy
        )
    else:
        results = evaluate_on_eval_tasks(
            model, tokenizer, device,
            challenges_path=args.challenges,
            solutions_path=args.solutions,
            num_tasks=args.num_tasks,
            beam_width=args.beam_width,
            measure_entropy_flag=args.measure_entropy
        )

    # Print results
    if results:
        print_results(results, args.mode)
    else:
        print("Evaluation failed or no results.")


if __name__ == "__main__":
    main()
