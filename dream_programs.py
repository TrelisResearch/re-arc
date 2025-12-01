"""
Dream phase: Sample programs unconditionally and test on grids.

This implements DreamCoder's "fantasy" generation:
1. Sample programs from model prior (no task conditioning)
2. Execute on input grids (random or from tasks)
3. Collect valid (input, output, program) tuples for training

Usage:
    # Sample and test on training task inputs
    uv run python dream_programs.py --checkpoint checkpoints/model_best.pt --num-dreams 100 --mode train

    # Sample and test on evaluation task inputs
    uv run python dream_programs.py --checkpoint checkpoints/model_best.pt --num-dreams 100 --mode eval

    # Sample and test on random grids
    uv run python dream_programs.py --checkpoint checkpoints/model_best.pt --num-dreams 100 --mode random
"""

import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
import random

from train import DecoderOnlyDSLTransformer
from tokenizer import DSLTokenizer
from utils import DSLConstrainedDecoder
import dsl
from dsl import *
import generators


def sample_program_from_prior(model, tokenizer, max_length=200, temperature=1.0,
                               use_constrained=True, device='cpu'):
    """
    Sample a program unconditionally from the model (no input grid).

    This samples from P[ρ|L] instead of P[ρ|x,L].
    """
    model.eval()

    constraint = None
    constraint_state = None
    if use_constrained:
        constraint = DSLConstrainedDecoder(tokenizer)
        constraint_state = constraint.new_state()

    # Start with just BOS token (no input grid encoding!)
    tokens = [tokenizer.bos_token_id]

    with torch.no_grad():
        for _ in range(max_length):
            # Convert to tensor
            input_ids = torch.tensor([tokens], device=device)

            # Get logits
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature

            # Apply constraints if enabled
            if constraint_state and constraint:
                next_logits = constraint.apply(next_logits, constraint_state)

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            tokens.append(next_token)

            # Update constraint state
            if constraint_state:
                constraint_state.update(next_token)

            if next_token == tokenizer.eos_token_id:
                break

    # Decode to code
    code = tokenizer.decode(tokens)
    return code, tokens


def execute_program_on_grid(code, input_grid):
    """
    Execute a DSL program on an input grid.

    Returns:
        (success, output_grid, error_message)
    """
    try:
        # Create function from code
        func_code = f"def dream_program(I):\n"
        for line in code.split('\n'):
            if line.strip() and not line.strip().startswith('return'):
                func_code += f"    {line.strip()}\n"

        # Extract return statement
        import re
        return_match = re.search(r'return\s+(.+)', code)
        if return_match:
            func_code += f"    return {return_match.group(1)}\n"
        else:
            return False, None, "No return statement"

        # Execute in DSL namespace
        namespace = {name: getattr(dsl, name) for name in dir(dsl) if not name.startswith('__')}
        exec(func_code, namespace)
        dream_program = namespace['dream_program']

        # Execute on input
        output_grid = dream_program(input_grid)

        # Validate output
        if not isinstance(output_grid, tuple):
            return False, None, "Output is not a grid (tuple)"

        if len(output_grid) == 0:
            return False, None, "Output grid is empty"

        if any(len(row) == 0 for row in output_grid):
            return False, None, "Output grid has empty rows"

        # Check grid is reasonable size
        if len(output_grid) > 30 or any(len(row) > 30 for row in output_grid):
            return False, None, "Output grid too large"

        return True, output_grid, "Success"

    except Exception as e:
        return False, None, f"Execution error: {str(e)}"


def get_input_grids(mode, num_grids, task_ids=None):
    """
    Get input grids to test programs on.

    Args:
        mode: 'train', 'eval', or 'random'
        num_grids: How many grids to get
        task_ids: Optional list of specific task IDs

    Returns:
        List of (task_id, input_grid) tuples
    """
    grids = []

    if mode == 'random':
        for i in range(num_grids):
            # Generate random grid
            h = random.randint(3, 15)
            w = random.randint(3, 15)
            grid = tuple(
                tuple(random.randint(0, 9) for _ in range(w))
                for _ in range(h)
            )
            grids.append((f"random_{i}", grid))

    elif mode == 'train':
        # Get grids from training task generators
        if task_ids is None:
            # Get all available task IDs
            task_ids = []
            for name in dir(generators):
                if name.startswith('generate_'):
                    task_ids.append(name.replace('generate_', ''))

        for task_id in task_ids[:num_grids]:
            try:
                generator = getattr(generators, f'generate_{task_id}')
                example = generator(0.0, 0.5)
                grids.append((task_id, example['input']))
            except:
                continue

    elif mode == 'eval':
        # Load evaluation tasks
        import json
        with open('data/arc-agi_evaluation_challenges.json', 'r') as f:
            eval_tasks = json.load(f)

        task_ids_list = list(eval_tasks.keys())[:num_grids]
        for task_id in task_ids_list:
            task = eval_tasks[task_id]
            # Use first train example as input
            if task['train']:
                input_grid = tuple(tuple(row) for row in task['train'][0]['input'])
                grids.append((task_id, input_grid))

    return grids


def main():
    parser = argparse.ArgumentParser(description='Sample programs from model prior')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-dreams', type=int, default=100,
                       help='Number of programs to sample')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'random'],
                       help='Where to get input grids from')
    parser.add_argument('--grids-per-program', type=int, default=5,
                       help='Test each program on N grids')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--use-constrained', action='store_true', default=True,
                       help='Use constrained decoding')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Maximum program length')
    parser.add_argument('--output', type=str, default='dreams.json',
                       help='Output file for successful dreams')
    parser.add_argument('--config', type=str, default=None,
                       help='Model config file (if not provided, infers from checkpoint)')

    args = parser.parse_args()

    # Load checkpoint first to infer config if needed
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint_cpu = torch.load(args.checkpoint, map_location='cpu')

    # Infer model dimensions from checkpoint
    emb_shape = checkpoint_cpu['model_state_dict']['embedding.weight'].shape
    vocab_size_ckpt = emb_shape[0]
    d_model_ckpt = emb_shape[1]

    # Load config or create from checkpoint
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Infer config from checkpoint
        print(f"Inferring config from checkpoint (d_model={d_model_ckpt})...")
        config = {
            'model': {
                'd_model': d_model_ckpt,
                'n_head': 8,
                'num_layers': 1,
                'num_recursions': 8,
                'dropout': 0.1
            }
        }

    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = DSLTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load model
    model = DecoderOnlyDSLTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config['model']['d_model'],
        n_head=config['model']['n_head'],
        num_layers=config['model']['num_layers'],
        num_recursions=config['model']['num_recursions'],
        dropout=config['model']['dropout']
    ).to(device)

    # Move checkpoint to device and load
    checkpoint = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in checkpoint_cpu.items()}
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {args.checkpoint}")
    print(f"Model config: {config['model']}")

    # Get input grids
    print(f"\nGetting input grids (mode={args.mode})...")
    input_grids = get_input_grids(args.mode, args.grids_per_program * args.num_dreams)
    print(f"Got {len(input_grids)} input grids")

    # Sample programs and test
    print(f"\nSampling {args.num_dreams} programs from prior...")
    print(f"Testing each on {args.grids_per_program} grids")
    print("=" * 80)

    successful_dreams = []
    all_sampled_programs = []  # For debugging
    stats = {
        'total_sampled': 0,
        'syntax_valid': 0,
        'execution_success': 0,
        'total_tests': 0,
        'successful_tests': 0
    }

    for dream_idx in tqdm(range(args.num_dreams), desc="Dreaming"):
        stats['total_sampled'] += 1

        # Sample program
        code, tokens = sample_program_from_prior(
            model, tokenizer,
            max_length=args.max_length,
            temperature=args.temperature,
            use_constrained=args.use_constrained,
            device=device
        )

        # Save for debugging
        all_sampled_programs.append(code)

        # Check if syntactically valid
        is_syntax_valid = False
        try:
            import ast
            ast.parse(code)
            stats['syntax_valid'] += 1
            is_syntax_valid = True
        except:
            continue

        # Test on multiple grids
        program_results = []
        grids_to_test = input_grids[dream_idx * args.grids_per_program:(dream_idx + 1) * args.grids_per_program]

        for task_id, input_grid in grids_to_test:
            stats['total_tests'] += 1

            success, output_grid, message = execute_program_on_grid(code, input_grid)

            if success:
                stats['successful_tests'] += 1
                stats['execution_success'] += 1

                program_results.append({
                    'task_id': task_id,
                    'input': input_grid,
                    'output': output_grid,
                    'success': True
                })
            else:
                program_results.append({
                    'task_id': task_id,
                    'success': False,
                    'error': message
                })

        # If at least one execution succeeded, save this dream
        if any(r['success'] for r in program_results):
            successful_dreams.append({
                'dream_id': dream_idx,
                'code': code,
                'tokens': tokens,
                'results': program_results,
                'success_rate': sum(r['success'] for r in program_results) / len(program_results)
            })

    # Print statistics
    print("\n" + "=" * 80)
    print("DREAMING STATISTICS")
    print("=" * 80)
    print(f"Programs sampled: {stats['total_sampled']}")
    print(f"Syntax valid: {stats['syntax_valid']} ({stats['syntax_valid']/stats['total_sampled']*100:.1f}%)")
    print(f"Programs with ≥1 successful execution: {stats['execution_success']} ({stats['execution_success']/stats['total_sampled']*100:.1f}%)")
    print(f"\nTotal execution attempts: {stats['total_tests']}")
    if stats['total_tests'] > 0:
        print(f"Successful executions: {stats['successful_tests']} ({stats['successful_tests']/stats['total_tests']*100:.1f}%)")
    else:
        print(f"Successful executions: 0 (no tests run)")
    print(f"\nSuccessful dreams saved: {len(successful_dreams)}")

    # Show sampled programs for debugging
    print("\n" + "=" * 80)
    print("SAMPLED PROGRAMS (first 5)")
    print("=" * 80)
    for i, code in enumerate(all_sampled_programs[:5], 1):
        print(f"\nProgram {i}:")
        print(code)
        print("-" * 40)

    # Show examples
    if successful_dreams:
        print("\n" + "=" * 80)
        print("EXAMPLE DREAMS")
        print("=" * 80)

        for dream in successful_dreams[:3]:
            print(f"\nDream #{dream['dream_id']} (success rate: {dream['success_rate']*100:.0f}%)")
            print("Code:")
            print(dream['code'])
            print("\nSuccessful executions:")
            for result in dream['results']:
                if result['success']:
                    print(f"  Task {result['task_id']}: {len(result['input'])}x{len(result['input'][0])} → {len(result['output'])}x{len(result['output'][0])}")
            print("-" * 80)

    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            'stats': stats,
            'dreams': successful_dreams,
            'config': {
                'num_dreams': args.num_dreams,
                'mode': args.mode,
                'temperature': args.temperature,
                'use_constrained': args.use_constrained
            }
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
