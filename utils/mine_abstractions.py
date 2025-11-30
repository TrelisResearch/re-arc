"""
Mine common patterns from verifier programs to propose new DSL abstractions.

Usage:
    uv run python utils/mine_abstractions.py --min-freq 3 --max-patterns 20
"""

import ast
import re
from collections import Counter, defaultdict
import argparse
import json
import os
import sys

# Add parent directory to path and construct correct path to verifiers.py
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

VERIFIERS_PATH = os.path.join(parent_dir, 'verifiers.py')


def extract_verifier_programs():
    """Extract all verifier function bodies from verifiers.py"""
    with open(VERIFIERS_PATH, 'r') as f:
        content = f.read()

    tree = ast.parse(content)
    lines = content.splitlines()

    programs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith('verify_'):
            task_id = node.name.replace('verify_', '')

            # Extract function body lines
            start_line = node.body[0].lineno - 1
            end_line = node.end_lineno
            body_lines = lines[start_line:end_line]

            # Remove indentation
            if body_lines:
                indent = len(body_lines[0]) - len(body_lines[0].lstrip())
                body_lines = [line[indent:] if len(line) > indent else line
                             for line in body_lines]

                # Filter out return statements for pattern mining
                code_lines = [line for line in body_lines
                             if line.strip() and not line.strip().startswith('return')]

                programs[task_id] = code_lines

    return programs


def normalize_line(line):
    """
    Normalize a line for pattern matching by replacing variable names with placeholders.
    e.g., "x0 = palette(I)" -> "VAR = palette(INPUT)"
          "x5 = other(x0, ZERO)" -> "VAR = other(VAR, ZERO)"
    """
    line = line.strip()

    # Replace input variable
    line = re.sub(r'\bI\b', 'INPUT', line)

    # Replace variable assignments and references
    # Match pattern: x0 = ...
    line = re.sub(r'\bx\d+\s*=', 'VAR =', line)

    # Replace variable references in function calls
    line = re.sub(r'\bx\d+\b', 'VAR', line)

    return line


def extract_patterns(programs, window_size=2):
    """
    Extract common sequential patterns from programs.

    Args:
        programs: Dict of task_id -> list of code lines
        window_size: Size of sliding window (2 = pairs, 3 = triplets, etc.)

    Returns:
        Dict of pattern -> list of (task_id, line_number) occurrences
    """
    patterns = defaultdict(list)

    for task_id, lines in programs.items():
        # Slide window over lines
        for i in range(len(lines) - window_size + 1):
            window = lines[i:i+window_size]

            # Normalize each line in the window
            normalized = tuple(normalize_line(line) for line in window)

            # Skip if window contains empty lines
            if any(not line for line in normalized):
                continue

            # Store original (non-normalized) pattern with occurrence info
            original_window = tuple(line.strip() for line in window)
            patterns[normalized].append({
                'task_id': task_id,
                'line_num': i,
                'original': original_window
            })

    return patterns


def score_pattern(pattern, occurrences, pattern_length):
    """
    Score a pattern based on frequency and potential token savings.

    Returns:
        (tokens_saved, frequency, pattern)
    """
    frequency = len(occurrences)

    # Estimate tokens per line (rough heuristic)
    tokens_per_line = sum(len(line.split()) for line in pattern) / len(pattern)
    pattern_tokens = int(tokens_per_line * len(pattern))

    # Tokens saved = frequency * (pattern_length - 1)
    # -1 because replacement is 1 token
    tokens_saved = frequency * (pattern_tokens - 1)

    return tokens_saved, frequency


def generate_name(pattern):
    """Generate a mechanical name for a pattern"""
    functions = []

    for line in pattern:
        # Extract function name: "VAR = function(...)" -> "function"
        match = re.search(r'=\s*([a-z_]+)\s*\(', line)
        if match:
            functions.append(match.group(1))

    if not functions:
        return "learned_pattern"

    # Join first 3 functions
    name = '_'.join(functions[:3])
    return f"learned_{name}"


def analyze_pattern_occurrences(occurrences):
    """Analyze where a pattern occurs to show examples"""
    # Group by unique original text
    unique_originals = defaultdict(list)
    for occ in occurrences:
        original_key = occ['original']
        unique_originals[original_key].append(occ['task_id'])

    return unique_originals


def main():
    parser = argparse.ArgumentParser(description='Mine common patterns from verifiers')
    parser.add_argument('--min-freq', type=int, default=3,
                       help='Minimum frequency for a pattern to be considered (default: 3)')
    parser.add_argument('--max-patterns', type=int, default=20,
                       help='Maximum number of patterns to display (default: 20)')
    parser.add_argument('--window-size', type=int, default=2,
                       help='Size of pattern window (default: 2)')
    parser.add_argument('--min-tokens-saved', type=int, default=10,
                       help='Minimum tokens saved to consider pattern (default: 10)')
    parser.add_argument('--output-json', type=str, default=None,
                       help='Output results to JSON file')

    args = parser.parse_args()

    print("Extracting verifier programs...")
    programs = extract_verifier_programs()
    print(f"Found {len(programs)} verifier programs")

    print(f"\nMining patterns (window_size={args.window_size})...")
    patterns = extract_patterns(programs, window_size=args.window_size)
    print(f"Found {len(patterns)} unique patterns")

    # Filter and score patterns
    print(f"\nFiltering patterns (min_freq={args.min_freq}, min_tokens_saved={args.min_tokens_saved})...")
    scored_patterns = []

    for pattern, occurrences in patterns.items():
        frequency = len(occurrences)

        if frequency < args.min_freq:
            continue

        tokens_saved, _ = score_pattern(pattern, occurrences, args.window_size)

        if tokens_saved < args.min_tokens_saved:
            continue

        scored_patterns.append({
            'pattern': pattern,
            'occurrences': occurrences,
            'frequency': frequency,
            'tokens_saved': tokens_saved,
            'name': generate_name(pattern)
        })

    # Sort by tokens saved
    scored_patterns.sort(key=lambda x: x['tokens_saved'], reverse=True)

    print(f"Found {len(scored_patterns)} patterns meeting criteria")
    print(f"\nTop {args.max_patterns} patterns by tokens saved:\n")
    print("=" * 100)

    results = []
    for i, item in enumerate(scored_patterns[:args.max_patterns], 1):
        pattern = item['pattern']
        frequency = item['frequency']
        tokens_saved = item['tokens_saved']
        name = item['name']
        occurrences = item['occurrences']

        print(f"\n#{i} - {name}")
        print(f"   Frequency: {frequency} occurrences")
        print(f"   Tokens saved: {tokens_saved}")
        print(f"   Pattern (normalized):")
        for line in pattern:
            print(f"      {line}")

        # Show a few example occurrences
        unique_originals = analyze_pattern_occurrences(occurrences)
        print(f"   Example original forms ({len(unique_originals)} unique):")

        for j, (original, task_ids) in enumerate(list(unique_originals.items())[:3], 1):
            print(f"      Form {j} (appears in {len(task_ids)} tasks, e.g., {task_ids[0]}):")
            for line in original:
                print(f"         {line}")

        print("-" * 100)

        # Store for JSON output
        # Get all tasks where this pattern appears
        all_tasks = []
        for task_ids in unique_originals.values():
            all_tasks.extend(task_ids)

        results.append({
            'rank': i,
            'name': name,
            'frequency': frequency,
            'tokens_saved': tokens_saved,
            'pattern_normalized': list(pattern),
            'example_tasks': all_tasks[:10],  # First 10 tasks for testing
            'num_unique_forms': len(unique_originals)
        })

    # Output to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n\nResults saved to {args.output_json}")

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Total programs analyzed: {len(programs)}")
    print(f"Total unique patterns found: {len(patterns)}")
    print(f"Patterns meeting criteria: {len(scored_patterns)}")
    print(f"Total tokens that could be saved: {sum(p['tokens_saved'] for p in scored_patterns)}")

    if scored_patterns:
        avg_freq = sum(p['frequency'] for p in scored_patterns) / len(scored_patterns)
        print(f"Average pattern frequency: {avg_freq:.1f}")


if __name__ == '__main__':
    main()
