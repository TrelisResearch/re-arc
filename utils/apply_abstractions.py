"""
Apply abstractions to verifiers with validation.

This script:
1. Takes patterns from mine_abstractions.py
2. Attempts to substitute them in verifier programs
3. VALIDATES each substitution by running test cases
4. Rolls back if substitution breaks the program
5. Creates versioned verifier files

Usage:
    uv run python utils/apply_abstractions.py --patterns utils/patterns.json --max-abstractions 10
"""

import ast
import re
import json
import argparse
from collections import defaultdict
import inspect
import random
import sys
import os

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

VERIFIERS_PATH = os.path.join(parent_dir, 'verifiers.py')

import dsl
from dsl import *
import verifiers
import generators


def extract_verifier_source(task_id):
    """Get the source code of a verifier function"""
    with open(VERIFIERS_PATH, 'r') as f:
        content = f.read()

    tree = ast.parse(content)
    lines = content.splitlines()

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == f'verify_{task_id}':
            start_line = node.body[0].lineno - 1
            end_line = node.end_lineno
            body_lines = lines[start_line:end_line]

            if body_lines:
                indent = len(body_lines[0]) - len(body_lines[0].lstrip())
                body_lines = [line[indent:] if len(line) > indent else line
                             for line in body_lines]

            return '\n'.join(body_lines)

    return None


def normalize_line(line):
    """Normalize line for pattern matching"""
    line = line.strip()
    line = re.sub(r'\bI\b', 'INPUT', line)
    line = re.sub(r'\bx\d+\s*=', 'VAR =', line)
    line = re.sub(r'\bx\d+\b', 'VAR', line)
    return line


def pattern_matches(program_lines, start_idx, pattern_normalized):
    """Check if pattern matches at given position"""
    if start_idx + len(pattern_normalized) > len(program_lines):
        return False

    for i, pattern_line in enumerate(pattern_normalized):
        program_line = program_lines[start_idx + i].strip()
        if normalize_line(program_line) != pattern_line:
            return False

    return True


def calculate_pattern_specificity(pattern_normalized):
    """
    Calculate how specific/constrained a pattern is.
    Returns score: higher = more specific = safer to abstract
    """
    score = 0
    pattern_str = ' '.join(pattern_normalized)

    # Count specific tokens (good indicators of semantic meaning)
    if 'INPUT' in pattern_str:
        score += 2  # References input grid - very specific

    # Count specific function names (not just VAR)
    functions = re.findall(r'([a-z_]+)\(', pattern_str)
    score += len(functions)  # Each named function adds specificity

    # Count constants (ZERO, ONE, T, F, etc.)
    constants = re.findall(r'\b[A-Z][A-Z]+\b', pattern_str)
    score += len(constants)

    # Penalize generic VAR usage
    var_count = pattern_str.count('VAR')
    genericity = var_count / max(len(pattern_normalized), 1)

    return score, genericity


def should_abstract_pattern(pattern_normalized, frequency, num_unique_forms):
    """
    Decide if pattern should be abstracted based on:
    1. Specificity (does it have semantic constraints?)
    2. Consistency (ratio of frequency to unique forms)
    """
    specificity, genericity = calculate_pattern_specificity(pattern_normalized)
    ratio = frequency / max(num_unique_forms, 1)

    # High specificity: safe to abstract
    if specificity >= 4:
        return True, "high_specificity", specificity

    # Medium specificity: check consistency ratio
    if specificity >= 2:
        if ratio >= 2.0:
            return True, "medium_specificity_high_ratio", specificity
        else:
            return False, "medium_specificity_low_ratio", specificity

    # Low specificity: very high ratio required
    if specificity >= 1:
        if ratio >= 3.0:
            return True, "low_specificity_very_high_ratio", specificity
        else:
            return False, "low_specificity", specificity

    # No specificity: don't abstract
    return False, "too_generic", specificity


def substitute_pattern(program_lines, start_idx, pattern_len, new_primitive_call):
    """
    Substitute pattern with new primitive and renumber variables.
    Returns new program lines.
    """
    # Get lines before and after pattern
    before = program_lines[:start_idx]
    after = program_lines[start_idx + pattern_len:]

    # The pattern spans pattern_len lines, we're replacing with 1 line
    # Need to extract what the pattern produces
    pattern_lines = program_lines[start_idx:start_idx + pattern_len]

    # Last line of pattern determines output variable
    last_line = pattern_lines[-1].strip()
    match = re.match(r'(x\d+)\s*=', last_line)
    if not match:
        return None

    output_var = match.group(1)

    # Create replacement line
    replacement = f"    {output_var} = {new_primitive_call}"

    # Combine
    new_lines = before + [replacement] + after

    # Renumber variables after substitution point
    # (Pattern was N lines, now 1 line, so we saved N-1 lines)
    lines_saved = pattern_len - 1

    if lines_saved > 0:
        new_lines = renumber_variables(new_lines, start_idx + 1, lines_saved)

    return new_lines


def renumber_variables(lines, start_idx, offset):
    """
    Renumber variables after start_idx by reducing by offset.
    E.g., if we removed 2 lines, x5 becomes x3, x6 becomes x4, etc.
    """
    new_lines = []

    for i, line in enumerate(lines):
        if i < start_idx:
            new_lines.append(line)
        else:
            # Replace variable assignments: x5 = ... -> x3 = ...
            new_line = re.sub(
                r'\b(x)(\d+)\b',
                lambda m: f"x{max(0, int(m.group(2)) - offset)}",
                line
            )
            new_lines.append(new_line)

    return new_lines


def validate_substitution(task_id, new_program_code, pattern_info, primitive_name, num_test_cases=5):
    """
    Validate that new program produces same outputs as original verifier.

    Returns:
        (is_valid, error_message)
    """
    try:
        # Get original verifier
        original_verifier = getattr(verifiers, f'verify_{task_id}')

        # Get generator for test cases
        generator_func = getattr(generators, f'generate_{task_id}', None)
        if generator_func is None:
            return False, "No generator available"

        # Generate test cases
        test_cases = []
        for _ in range(num_test_cases):
            try:
                example = generator_func(0.0, 0.5)
                test_cases.append(example['input'])
            except:
                continue

        if len(test_cases) < 3:
            return False, "Could not generate enough test cases"

        # Create the primitive function from the pattern
        # This defines what the new abstraction actually does
        pattern_code = create_primitive_function(primitive_name, pattern_info)

        # Execute new program as a function
        # Build function from code
        func_code = f"def verify_{task_id}_new(I):\n"
        for line in new_program_code.split('\n'):
            if line.strip() and not line.strip().startswith('return'):
                func_code += f"    {line.strip()}\n"

        # Extract return statement
        return_match = re.search(r'return\s+(.+)', new_program_code)
        if return_match:
            func_code += f"    return {return_match.group(1)}\n"
        else:
            return False, "No return statement found"

        # Execute function definition in DSL namespace
        namespace = {name: getattr(dsl, name) for name in dir(dsl) if not name.startswith('__')}

        # Add the primitive function to namespace
        exec(pattern_code, namespace)

        # Now execute the new verifier
        exec(func_code, namespace)
        new_verifier = namespace[f'verify_{task_id}_new']

        # Test on all test cases
        for test_input in test_cases:
            original_output = original_verifier(test_input)
            new_output = new_verifier(test_input)

            if original_output != new_output:
                return False, f"Output mismatch"

        return True, "All tests passed"

    except Exception as e:
        return False, f"Execution error: {str(e)}"


def create_primitive_function(primitive_name, pattern_info):
    """
    Create a Python function definition from a pattern.

    For example, if pattern is:
        VAR = leastcolor(INPUT)
        VAR = ofcolor(INPUT, VAR)

    Creates:
        def learned_leastcolor_ofcolor(I):
            x0 = leastcolor(I)
            x1 = ofcolor(I, x0)
            return x1
    """
    pattern_normalized = pattern_info['pattern_normalized']

    # Get example original form to extract actual code
    # We'll use the first example task
    example_tasks = pattern_info['example_tasks']
    if not example_tasks:
        return ""

    task_id = example_tasks[0]
    original_code = extract_verifier_source(task_id)
    if not original_code:
        return ""

    lines = original_code.split('\n')

    # Find the pattern in the original code
    for i in range(len(lines) - len(pattern_normalized) + 1):
        if pattern_matches(lines, i, tuple(pattern_normalized)):
            # Found it! Extract these lines
            pattern_lines = [lines[i + j].strip() for j in range(len(pattern_normalized))]

            # Build function
            func_code = f"def {primitive_name}(I):\n"
            for line in pattern_lines:
                func_code += f"    {line}\n"

            # Return last variable
            last_line = pattern_lines[-1]
            var_match = re.match(r'(x\d+)\s*=', last_line)
            if var_match:
                func_code += f"    return {var_match.group(1)}\n"

            return func_code

    return ""


def apply_abstraction_to_task(task_id, pattern_info, primitive_name):
    """
    Apply a single abstraction to a single task.
    Returns (success, new_code, message)
    """
    original_code = extract_verifier_source(task_id)
    if not original_code:
        return False, None, "Could not extract source"

    lines = original_code.split('\n')
    pattern_normalized = tuple(pattern_info['pattern_normalized'])
    pattern_len = len(pattern_normalized)

    # Find pattern occurrences
    matches = []
    for i in range(len(lines) - pattern_len + 1):
        if pattern_matches(lines, i, pattern_normalized):
            matches.append(i)

    if not matches:
        return False, None, "Pattern not found"

    # Try substituting (substitute from end to maintain indices)
    new_lines = lines[:]
    for match_idx in reversed(matches):
        # Extract input to pattern (usually INPUT/I)
        pattern_first_line = lines[match_idx].strip()

        # Build primitive call
        # For patterns operating on INPUT, call is: primitive_name(I)
        if 'INPUT' in ' '.join(pattern_normalized) or '(I' in pattern_first_line:
            primitive_call = f"{primitive_name}(I)"
        else:
            # Pattern operates on variables - need to extract input
            match = re.search(r'\(([^)]+)\)', pattern_first_line)
            if match:
                inputs = match.group(1)
                primitive_call = f"{primitive_name}({inputs})"
            else:
                continue

        new_lines = substitute_pattern(new_lines, match_idx, pattern_len, primitive_call)
        if new_lines is None:
            return False, None, "Substitution failed"

    new_code = '\n'.join(new_lines)

    # Validate
    is_valid, message = validate_substitution(task_id, new_code, pattern_info, primitive_name)

    if is_valid:
        return True, new_code, "Validated successfully"
    else:
        return False, None, f"Validation failed: {message}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patterns', type=str, required=True,
                       help='JSON file with mined patterns')
    parser.add_argument('--max-abstractions', type=int, default=10,
                       help='Maximum number of abstractions to apply')
    parser.add_argument('--output', type=str, default='verifiers_v1.py',
                       help='Output file for modified verifiers')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without modifying files')

    args = parser.parse_args()

    # Load patterns
    with open(args.patterns, 'r') as f:
        patterns = json.load(f)

    print(f"Loaded {len(patterns)} patterns")
    print("\nEvaluating patterns for abstraction...\n")

    # Filter patterns by specificity and consistency
    selected_patterns = []

    for pattern_info in patterns[:args.max_abstractions * 3]:  # Check 3x to get enough
        pattern_normalized = tuple(pattern_info['pattern_normalized'])
        frequency = pattern_info['frequency']
        num_unique_forms = pattern_info['num_unique_forms']

        should_abstract, reason, specificity = should_abstract_pattern(
            pattern_normalized, frequency, num_unique_forms
        )

        print(f"Pattern: {pattern_info['name']}")
        print(f"  Frequency: {frequency}, Unique forms: {num_unique_forms}")
        print(f"  Specificity score: {specificity}")
        print(f"  Decision: {'✅ ABSTRACT' if should_abstract else '❌ SKIP'} ({reason})")

        if should_abstract:
            selected_patterns.append(pattern_info)
            if len(selected_patterns) >= args.max_abstractions:
                break
        print()

    print(f"\nSelected {len(selected_patterns)} patterns for abstraction\n")
    print("=" * 100)

    # Apply abstractions with validation
    results = {
        'successful_abstractions': [],
        'failed_abstractions': [],
        'task_modifications': defaultdict(list)
    }

    for pattern_info in selected_patterns:
        pattern_name = pattern_info['name']
        example_tasks = pattern_info['example_tasks']

        print(f"\nApplying abstraction: {pattern_name}")
        print(f"Testing on {len(example_tasks)} example tasks...")

        successful_tasks = []
        failed_tasks = []

        for task_id in example_tasks:
            success, new_code, message = apply_abstraction_to_task(
                task_id, pattern_info, pattern_name
            )

            if success:
                successful_tasks.append(task_id)
                results['task_modifications'][task_id].append({
                    'pattern': pattern_name,
                    'new_code': new_code
                })
                print(f"  ✅ {task_id}: {message}")
            else:
                failed_tasks.append((task_id, message))
                print(f"  ❌ {task_id}: {message}")

        success_rate = len(successful_tasks) / len(example_tasks) if example_tasks else 0

        if success_rate >= 0.8:  # 80% success rate
            results['successful_abstractions'].append({
                'name': pattern_name,
                'pattern': pattern_info['pattern_normalized'],
                'success_rate': success_rate,
                'successful_tasks': successful_tasks,
                'failed_tasks': [t[0] for t in failed_tasks]
            })
            print(f"  ✅ Pattern validated: {success_rate*100:.0f}% success rate")
        else:
            results['failed_abstractions'].append({
                'name': pattern_name,
                'reason': f"Low success rate: {success_rate*100:.0f}%",
                'failed_tasks': failed_tasks
            })
            print(f"  ❌ Pattern rejected: {success_rate*100:.0f}% success rate (need 80%)")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Successful abstractions: {len(results['successful_abstractions'])}")
    print(f"Failed abstractions: {len(results['failed_abstractions'])}")
    print(f"Tasks modified: {len(results['task_modifications'])}")

    if results['successful_abstractions']:
        print("\n✅ Validated abstractions:")
        for abstraction in results['successful_abstractions']:
            print(f"  - {abstraction['name']} ({abstraction['success_rate']*100:.0f}% success)")

    # Save results
    output_json = args.patterns.replace('.json', '_applied.json')
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_json}")


if __name__ == '__main__':
    main()
