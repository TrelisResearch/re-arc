"""
Test tokenizer encode/decode round-trip on real verifier functions.
Pick random tasks, encode their solutions, decode, and verify execution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
import re
import ast
from tokenizer import DSLTokenizer
import dsl
from dsl import *

def execute_and_score(generated_code, input_grid, target_grid):
    """Same as in train.py, with explicit cell checking"""
    try:
        ast.parse(generated_code)
    except SyntaxError as e:
        return False, False, False, f"Syntax error: {e}"

    if 'return' not in generated_code:
        return True, False, False, "No return statement"

    wrapped_code = f"def solver(I):\n"
    for line in generated_code.split('\n'):
        wrapped_code += f"    {line}\n"

    local_scope = {}
    global_scope = {k: getattr(dsl, k) for k in dir(dsl) if not k.startswith('__')}

    try:
        exec(wrapped_code, global_scope, local_scope)
        solver = local_scope['solver']
        prediction = solver(input_grid)

        # Check shape first
        if len(prediction) != len(target_grid):
            return True, True, False, f"Wrong shape: Expected {len(target_grid)} rows, Got {len(prediction)}"
        if len(prediction) > 0 and len(prediction[0]) != len(target_grid[0]):
            return True, True, False, f"Wrong shape: Expected {len(target_grid[0])} cols, Got {len(prediction[0])}"

        # Check cell-by-cell
        mismatches = []
        for i, (pred_row, target_row) in enumerate(zip(prediction, target_grid)):
            for j, (pred_val, target_val) in enumerate(zip(pred_row, target_row)):
                if pred_val != target_val:
                    mismatches.append(f"({i},{j}): expected {target_val}, got {pred_val}")
                    if len(mismatches) >= 5:  # Limit to first 5 mismatches
                        break
            if len(mismatches) >= 5:
                break

        if mismatches:
            return True, True, False, f"Cell mismatches: {', '.join(mismatches)}"

        return True, True, True, "Correct - all cells match!"
    except Exception as e:
        return True, False, False, f"Runtime error: {e}"

def extract_function_body(func_name, verifiers_path='verifiers.py'):
    """Extract the body of a function from verifiers.py"""
    with open(verifiers_path, 'r') as f:
        content = f.read()

    # Parse the AST to find the function
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            # Get the source lines for this function
            lines = content.split('\n')
            func_lines = lines[node.lineno - 1:node.end_lineno]

            # Extract just the body (skip def line)
            body_lines = func_lines[1:]  # Skip the def line

            # Remove common indentation
            if body_lines:
                # Find minimum indentation
                min_indent = min(len(line) - len(line.lstrip())
                               for line in body_lines if line.strip())
                # Remove that indentation from all lines
                body_lines = [line[min_indent:] if len(line) > min_indent else line
                             for line in body_lines]

            return '\n'.join(body_lines)

    return None

def load_task_data(task_id, data_dir='re_arc/tasks'):
    """Load ARC task data"""
    task_path = os.path.join(data_dir, f'{task_id}.json')
    if not os.path.exists(task_path):
        return None

    with open(task_path, 'r') as f:
        task = json.load(f)

    return task

def test_roundtrip():
    print("="*80)
    print("TOKENIZER ROUND-TRIP TEST")
    print("="*80)

    # Initialize tokenizer (with path to verifiers.py in parent dir)
    verifiers_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'verifiers.py')
    tokenizer = DSLTokenizer(verifiers_path=verifiers_path)
    print(f"\nTokenizer vocab size: {tokenizer.vocab_size}")

    # Get all verify functions
    with open(verifiers_path, 'r') as f:
        content = f.read()

    verify_funcs = re.findall(r'def (verify_[a-f0-9]+)\(', content)
    print(f"Found {len(verify_funcs)} verifier functions")

    # Pick 3 random ones
    random.seed(42)
    selected = random.sample(verify_funcs, min(3, len(verify_funcs)))

    print(f"\nSelected tasks: {selected}\n")

    results = []

    for func_name in selected:
        task_id = func_name.replace('verify_', '')
        print("="*80)
        print(f"Task: {task_id}")
        print("="*80)

        # Extract function body
        original_code = extract_function_body(func_name, verifiers_path)
        if not original_code:
            print(f"❌ Could not extract function body for {func_name}")
            continue

        print("\n--- ORIGINAL CODE ---")
        print(original_code)

        # Encode
        token_ids = tokenizer.encode_code(original_code)
        print(f"\n--- ENCODED ---")
        print(f"Token count: {len(token_ids)}")
        print(f"Token IDs (first 20): {token_ids[:20]}")
        tokens = [tokenizer.id_to_token.get(tid, '?') for tid in token_ids[:20]]
        print(f"Tokens (first 20): {tokens}")

        # Decode
        decoded_code = tokenizer.decode(token_ids)
        print(f"\n--- DECODED CODE ---")
        print(decoded_code)

        # Load task data
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 're_arc', 'tasks')
        task_data = load_task_data(task_id, data_dir)
        if not task_data:
            print(f"\n⚠️  Could not load task data for {task_id}")
            continue

        # Test on first training example
        if task_data and len(task_data) > 0:
            example = task_data[0]
            input_grid = tuple(tuple(row) for row in example['input'])
            output_grid = tuple(tuple(row) for row in example['output'])

            print(f"\n--- EXECUTION TEST ---")
            print(f"Input grid: {len(input_grid)}x{len(input_grid[0])}")
            print(f"Expected output: {len(output_grid)}x{len(output_grid[0])}")

            # Test original
            is_syn_orig, is_run_orig, is_corr_orig, msg_orig = execute_and_score(
                original_code, input_grid, output_grid
            )
            print(f"\nOriginal: Syntax={is_syn_orig}, Runs={is_run_orig}, Correct={is_corr_orig}")
            print(f"  Message: {msg_orig}")

            # Test decoded
            is_syn_dec, is_run_dec, is_corr_dec, msg_dec = execute_and_score(
                decoded_code, input_grid, output_grid
            )
            print(f"Decoded:  Syntax={is_syn_dec}, Runs={is_run_dec}, Correct={is_corr_dec}")
            print(f"  Message: {msg_dec}")

            # Check if round-trip preserved correctness
            roundtrip_ok = (is_syn_orig == is_syn_dec and
                           is_run_orig == is_run_dec and
                           is_corr_orig == is_corr_dec)

            if roundtrip_ok and is_corr_dec:
                print(f"\n✅ ROUND-TRIP SUCCESSFUL!")
            elif roundtrip_ok:
                print(f"\n⚠️  Round-trip preserved behavior but solution doesn't work")
            else:
                print(f"\n❌ ROUND-TRIP FAILED - behavior changed!")

            results.append({
                'task_id': task_id,
                'original': (is_syn_orig, is_run_orig, is_corr_orig),
                'decoded': (is_syn_dec, is_run_dec, is_corr_dec),
                'roundtrip_ok': roundtrip_ok
            })

        print("\n")

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    for r in results:
        status = "✅" if r['roundtrip_ok'] and r['decoded'][2] else ("⚠️" if r['roundtrip_ok'] else "❌")
        print(f"{status} {r['task_id']}: Original={r['original']}, Decoded={r['decoded']}")

    success_rate = sum(1 for r in results if r['roundtrip_ok']) / len(results) if results else 0
    print(f"\nRound-trip success rate: {success_rate:.1%} ({sum(1 for r in results if r['roundtrip_ok'])}/{len(results)})")

if __name__ == "__main__":
    test_roundtrip()
