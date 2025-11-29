"""
Baseline: Apply training task DSL verifiers to evaluation tasks.

Tests how many eval tasks can be solved by directly applying
ground-truth DSL code from training tasks (verifiers).

Usage:
    uv run experimental/baseline_transfer.py
    uv run experimental/baseline_transfer.py --num-tasks 10
    uv run experimental/baseline_transfer.py --num-tasks 10 --workers 4
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import signal
import resource
from multiprocessing import Pool, Process, Queue
from functools import partial
from tqdm import tqdm

import verifiers
from dataset import ARCDataset


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


def load_eval_tasks(challenges_path='data/arc-agi_evaluation_challenges.json',
                   solutions_path='data/arc-agi_evaluation_solutions.json'):
    """Load ARC evaluation tasks."""
    with open(challenges_path, 'r') as f:
        challenges = json.load(f)
    with open(solutions_path, 'r') as f:
        solutions = json.load(f)
    return challenges, solutions


def set_limits():
    """Set resource limits for child processes."""
    import platform

    # Memory limits don't work well on macOS, skip them
    if platform.system() != 'Darwin':
        # Limit memory to 1GB (Linux only)
        memory_limit = 1 * 1024 * 1024 * 1024  # 1GB in bytes
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        except Exception:
            pass  # Silently ignore if not supported

    # Timeout handled by multiprocessing.Pool timeout parameter


def try_single_verifier(verifier_name, test_examples, timeout=1.0):
    """
    Try applying a single verifier to test examples with timeout.

    Returns verifier_name if successful, None otherwise.
    """
    # Set up timeout signal handler (Unix only)
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(int(timeout))

    try:
        # Get verifier function
        verifier_fn = getattr(verifiers, f'verify_{verifier_name}', None)
        if verifier_fn is None:
            signal.alarm(0)  # Cancel alarm
            signal.signal(signal.SIGALRM, old_handler)
            return None

        # Try on all test examples
        for example, expected_output in test_examples:
            input_grid = tuple(tuple(row) for row in example['input'])
            expected_grid = tuple(tuple(row) for row in expected_output)

            # Apply verifier (ground-truth DSL)
            result = verifier_fn(input_grid)

            # Check if it matches expected output
            if result != expected_grid:
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
                return None

        # All test examples passed!
        signal.alarm(0)  # Cancel alarm
        signal.signal(signal.SIGALRM, old_handler)
        return verifier_name
    except TimeoutException:
        # Timeout
        signal.signal(signal.SIGALRM, old_handler)
        return None
    except MemoryError:
        # OOM - skip this verifier
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return None
    except Exception as e:
        # Other error
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return None


def test_eval_task(eval_task_data, train_task_ids, timeout_per_verifier=1.0):
    """
    Test one eval task against all train verifiers.

    Args:
        eval_task_data: (eval_task_id, test_examples)
        train_task_ids: List of training task IDs to try
        timeout_per_verifier: Timeout in seconds per verifier attempt

    Returns:
        (eval_task_id, solving_train_task_id or None)
    """
    eval_task_id, test_examples = eval_task_data

    # Try each train verifier (sequentially within this eval task)
    for i, train_task_id in enumerate(train_task_ids):
        result = try_single_verifier(train_task_id, test_examples, timeout=timeout_per_verifier)

        if result is not None:
            # This verifier worked!
            return (eval_task_id, result)

    # No verifier worked for this eval task
    return (eval_task_id, None)


def main():
    parser = argparse.ArgumentParser(description="Baseline: Transfer train DSL to eval tasks")
    parser.add_argument("--challenges", type=str, default="data/arc-agi_evaluation_challenges.json",
                       help="Path to evaluation challenges JSON")
    parser.add_argument("--solutions", type=str, default="data/arc-agi_evaluation_solutions.json",
                       help="Path to evaluation solutions JSON")
    parser.add_argument("--num-tasks", type=int, default=None,
                       help="Limit to first N evaluation tasks (default: all 400)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    parser.add_argument("--timeout", type=float, default=1.0,
                       help="Timeout per verifier attempt in seconds (default: 1.0)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress information")

    args = parser.parse_args()

    # Load training task IDs and verifiers
    dataset = ARCDataset(diff_lb=0.0, diff_ub=0.5)
    train_task_ids = dataset.tasks
    print(f"Loaded {len(train_task_ids)} training task verifiers")

    # Verify at least one verifier is callable
    sample_verifier = getattr(verifiers, f'verify_{train_task_ids[0]}', None)
    if sample_verifier is None:
        print(f"ERROR: Could not load verifier for {train_task_ids[0]}")
        return
    if args.verbose:
        print(f"Sample verifier check: verify_{train_task_ids[0]} is callable")

    # Load evaluation tasks
    challenges, solutions = load_eval_tasks(args.challenges, args.solutions)
    eval_task_ids = list(challenges.keys())

    if args.num_tasks:
        eval_task_ids = eval_task_ids[:args.num_tasks]

    print(f"Testing on {len(eval_task_ids)} evaluation tasks")
    print(f"Using {args.workers} parallel workers")
    print(f"Timeout per verifier: {args.timeout}s")
    print(f"Strategy: Try each of {len(train_task_ids)} train verifiers until one succeeds\n")

    # Prepare eval task data
    eval_tasks_data = []
    for eval_task_id in eval_task_ids:
        if eval_task_id not in solutions:
            continue

        test_examples = list(zip(
            challenges[eval_task_id]['test'],
            solutions[eval_task_id]
        ))
        eval_tasks_data.append((eval_task_id, test_examples))

    # Process eval tasks in parallel
    test_func = partial(test_eval_task, train_task_ids=train_task_ids, timeout_per_verifier=args.timeout)

    results = {
        'total': len(eval_tasks_data),
        'solved': 0,
        'solved_tasks': []  # (eval_task_id, train_task_id that solved it)
    }

    with Pool(processes=args.workers) as pool:
        for eval_task_id, solving_train_id in tqdm(
            pool.imap_unordered(test_func, eval_tasks_data),
            total=len(eval_tasks_data),
            desc="Testing eval tasks"
        ):
            if solving_train_id is not None:
                results['solved'] += 1
                results['solved_tasks'].append((eval_task_id, solving_train_id))

    # Print results
    print("\n" + "="*60)
    print("BASELINE TRANSFER RESULTS")
    print("="*60)
    print(f"Total eval tasks tested: {results['total']}")
    print(f"Solved by train verifiers: {results['solved']}")
    print(f"Success rate: {100*results['solved']/results['total']:.1f}%")

    if results['solved_tasks']:
        print("\nSuccessful transfers:")
        for eval_id, train_id in results['solved_tasks']:
            print(f"  {eval_id} ← {train_id}")

    print("="*60)


if __name__ == "__main__":
    main()
