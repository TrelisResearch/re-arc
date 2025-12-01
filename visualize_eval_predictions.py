"""
Visualize model predictions on evaluation tasks.

Generates side-by-side comparisons of:
- Input grid
- Ground truth output
- Model prediction

Usage:
    uv run python visualize_eval_predictions.py --checkpoint models/re-arc-neuro-dsl-20ke/model_epoch_20000.pt --num-tasks 10
"""

import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from train import DecoderOnlyDSLTransformer, execute_and_score
from tokenizer import DSLTokenizer
from evaluate import greedy_generate, parallel_sample_generate, get_device
import dsl


def execute_code_on_grid(code, input_grid):
    """Execute generated code on input grid and return output grid."""
    try:
        # Build function from code
        func_code = "def generated_program(I):\n"
        for line in code.split('\n'):
            if line.strip():
                func_code += f"    {line.strip()}\n"

        # Execute in DSL namespace
        namespace = {name: getattr(dsl, name) for name in dir(dsl) if not name.startswith('__')}
        exec(func_code, namespace)
        program = namespace['generated_program']

        # Run on input
        output = program(input_grid)

        # Validate output is a grid
        if not isinstance(output, tuple) or len(output) == 0:
            return None, "Invalid output format"
        if any(not isinstance(row, tuple) for row in output):
            return None, "Invalid row format"
        if any(len(row) == 0 for row in output):
            return None, "Empty rows"
        if len(output) > 30 or any(len(row) > 30 for row in output):
            return None, "Output too large"

        return output, None
    except Exception as e:
        return None, str(e)


def visualize_task(task_id, input_grid, ground_truth, predicted, code,
                   syntax_valid, error_msg, output_dir, sample_info=None):
    """Create visualization comparing input, ground truth, and prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Input
    axes[0].imshow(np.array(input_grid), cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
    axes[0].set_title(f'Input\n{len(input_grid)}x{len(input_grid[0])}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[0].grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)

    # Ground truth
    axes[1].imshow(np.array(ground_truth), cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
    axes[1].set_title(f'Ground Truth\n{len(ground_truth)}x{len(ground_truth[0])}', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    axes[1].grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)

    # Predicted
    if predicted is not None:
        axes[2].imshow(np.array(predicted), cmap='tab10', vmin=0, vmax=9, interpolation='nearest')
        is_correct = (predicted == ground_truth)
        title = f'Predicted {"✓" if is_correct else "✗"}\n{len(predicted)}x{len(predicted[0])}'
        axes[2].set_title(title, fontsize=14, fontweight='bold',
                         color='green' if is_correct else 'red')
        axes[2].axis('off')
        axes[2].grid(True, which='both', color='white', linewidth=0.5, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, f'Execution Failed\n{error_msg}',
                    ha='center', va='center', fontsize=10, wrap=True)
        axes[2].set_title('Predicted ✗', fontsize=14, fontweight='bold', color='red')
        axes[2].axis('off')

    # Add code as text below
    code_text = f"Generated Code:\n{code[:500]}" + ("..." if len(code) > 500 else "")
    plt.figtext(0.1, 0.02, code_text, fontsize=8, family='monospace',
                verticalalignment='bottom', wrap=True)

    # Build title with sample info
    title = f'Task: {task_id}'
    if sample_info:
        title += f' (Sample {sample_info["sample_idx"]}/{sample_info["total_samples"]})'
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])

    # Save
    output_path = output_dir / f'{task_id}.png'
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()

    return str(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num-tasks', type=int, default=10,
                       help='Number of tasks to visualize')
    parser.add_argument('--output-dir', type=str, default='eval_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--challenges', type=str,
                       default='data/arc-agi_evaluation_challenges.json',
                       help='Path to challenges JSON')
    parser.add_argument('--solutions', type=str,
                       default='data/arc-agi_evaluation_solutions.json',
                       help='Path to solutions JSON')
    parser.add_argument('--max-gen-len', type=int, default=2048,
                       help='Maximum generation length')
    parser.add_argument('--num-samples', type=int, default=1,
                       help='Number of samples to generate per task (default: 1 for greedy)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature (only used with --num-samples > 1)')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Top-k sampling parameter')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Setup device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    tokenizer = DSLTokenizer()

    config = checkpoint.get('config', {})
    model = DecoderOnlyDSLTransformer(
        vocab_size=config.get('vocab_size', tokenizer.vocab_size),
        d_model=config.get('d_model', 512),
        n_head=config.get('n_head', 8),
        num_layers=config.get('num_layers', 1),
        num_recursions=config.get('num_recursions', 16)
    ).to(device)

    # Handle torch.compile prefix
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    epoch = checkpoint.get('epoch', -1)
    print(f"Loaded model from epoch {epoch}")

    # Load evaluation tasks
    print(f"\nLoading evaluation tasks from {args.challenges}")
    with open(args.challenges, 'r') as f:
        challenges = json.load(f)
    with open(args.solutions, 'r') as f:
        solutions = json.load(f)

    # Exclude eval tasks already solved by train verifiers (no generalization needed)
    excluded_tasks = {
        '070dd51e', '0c9aba6e', '1a2e2828', '27f8ce4f', '2c0b0aff', '358ba94e',
        '47996f11', '506d28a5', '50a16a69', '5b6cbef5', '60c09cac', '7039b2d7',
        '73182012', '67b4a34d', '73ccf9c2', '7bb29440', '8597cfd7', '981571dc',
        '9a4bb226', '9ddd00f0', 'aa18de87', 'af22c60d', 'bbb1b8b6', 'bf699163',
        'c663677b', 'c7d4e6ad', 'cd3c21df', 'd56f2372', 'e1baa8a4', 'e66aafb8',
        'e7a25a18', 'e95e3d8e', 'ea9794b1', 'ea959feb', 'f823c43c', 'f4081712'
    }

    all_task_ids = [tid for tid in challenges.keys() if tid not in excluded_tasks]
    print(f"Total eval tasks: {len(challenges)}")
    print(f"Excluded (solved by train verifiers): {len(excluded_tasks)}")
    print(f"Remaining for generalization testing: {len(all_task_ids)}")

    # Process tasks
    task_ids = all_task_ids[:args.num_tasks]
    print(f"\nProcessing {len(task_ids)} tasks...")

    stats = {
        'total': 0,
        'syntax_valid': 0,
        'execution_success': 0,
        'correct': 0,
        'identity_programs': 0,  # Programs that just return input
        'samples_tried': 0
    }

    results = []

    for task_id in task_ids:
        if task_id not in solutions:
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print('='*60)

        # Get first test example
        test_example = challenges[task_id]['test'][0]
        input_grid = tuple(tuple(row) for row in test_example['input'])
        output_grid = tuple(tuple(row) for row in solutions[task_id][0])

        print(f"Input shape:  {len(input_grid)}x{len(input_grid[0])}")
        print(f"Output shape: {len(output_grid)}x{len(output_grid[0])}")

        # Encode grids
        input_tokens = tokenizer.encode_grid(input_grid)
        output_tokens = tokenizer.encode_grid(output_grid)
        src_ids = [tokenizer.bos_token_id] + input_tokens + \
                  [tokenizer.sep_token_id] + output_tokens + \
                  [tokenizer.eos_token_id]
        src_tensor = torch.tensor(src_ids)

        # Generate code (single or multiple samples)
        if args.num_samples > 1:
            print(f"Generating {args.num_samples} samples...")
            codes = parallel_sample_generate(
                model, src_tensor, tokenizer, device,
                num_samples=args.num_samples,
                max_len=args.max_gen_len,
                temperature=args.temperature,
                top_k=args.top_k,
                debug=False
            )
        else:
            print("Generating code (greedy)...")
            codes = [greedy_generate(model, src_tensor, tokenizer, device,
                                    max_len=args.max_gen_len)]

        # Try each sample until we find a successful one
        best_code = None
        best_predicted = None
        best_error = None
        best_sample_idx = None
        syntax_valid = False

        for sample_idx, code in enumerate(codes, 1):
            stats['samples_tried'] += 1

            # Check syntax
            try:
                import ast
                ast.parse(code)
            except Exception as e:
                if sample_idx == 1:
                    print(f"  Sample {sample_idx}: ✗ Syntax error: {e}")
                continue

            # Execute code
            predicted, error_msg = execute_code_on_grid(code, input_grid)

            if predicted is None:
                if sample_idx == 1:
                    print(f"  Sample {sample_idx}: ✗ Execution failed: {error_msg}")
                continue

            # Check if it's an identity program (just returns input)
            if predicted == input_grid:
                stats['identity_programs'] += 1
                print(f"  Sample {sample_idx}: ✗ Identity program (returns input unchanged)")
                continue

            # Valid non-identity program!
            if best_code is None:
                syntax_valid = True
                best_code = code
                best_predicted = predicted
                best_error = error_msg
                best_sample_idx = sample_idx

                is_correct = (predicted == output_grid)
                print(f"  Sample {sample_idx}: ✓ Valid execution: {len(predicted)}x{len(predicted[0])}")
                if is_correct:
                    print(f"  Sample {sample_idx}: ✓ CORRECT!")
                    stats['correct'] += 1
                    break  # Found correct answer, stop trying
                else:
                    print(f"  Sample {sample_idx}: ✗ Incorrect output")

        # Use best result (or first sample if all failed)
        code = best_code if best_code else codes[0]
        predicted = best_predicted
        error_msg = best_error

        if syntax_valid:
            stats['syntax_valid'] += 1
        if predicted is not None:
            stats['execution_success'] += 1

        stats['total'] += 1

        # Visualize
        sample_info = None
        if best_sample_idx and args.num_samples > 1:
            sample_info = {
                'sample_idx': best_sample_idx,
                'total_samples': args.num_samples
            }
        viz_path = visualize_task(
            task_id, input_grid, output_grid, predicted,
            code, syntax_valid, error_msg, output_dir, sample_info
        )

        results.append({
            'task_id': task_id,
            'syntax_valid': syntax_valid,
            'execution_success': predicted is not None,
            'correct': predicted == output_grid if predicted else False,
            'visualization': viz_path
        })

        print(f"Saved: {viz_path}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total tasks: {stats['total']}")
    if args.num_samples > 1:
        print(f"Total samples generated: {stats['samples_tried']}")
        print(f"Identity programs (filtered): {stats['identity_programs']}")
    print(f"Syntax valid: {stats['syntax_valid']} ({stats['syntax_valid']/stats['total']*100:.1f}%)")
    print(f"Execution success: {stats['execution_success']} ({stats['execution_success']/stats['total']*100:.1f}%)")
    print(f"Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.1f}%)")
    print(f"\nVisualizations saved to: {output_dir}/")

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'stats': stats,
            'results': results,
            'config': {
                'checkpoint': args.checkpoint,
                'num_tasks': args.num_tasks
            }
        }, f, indent=2)

    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
