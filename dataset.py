import torch
from torch.utils.data import Dataset
import generators
import verifiers
import inspect
import re
import ast
import random
from tokenizer import DSLTokenizer

class ARCDataset(Dataset):
    def __init__(self, diff_lb=0.0, diff_ub=0.5):
        """
        One epoch = one pass through all available tasks.
        """
        self.diff_lb = diff_lb
        self.diff_ub = diff_ub
        
        self.tokenizer = DSLTokenizer()
        self.tasks = self._get_available_tasks()
        self.verifier_codes = self._extract_verifier_codes()
        
    def _get_available_tasks(self):
        tasks = []
        for name in dir(generators):
            if name.startswith('generate_'):
                task_id = name.replace('generate_', '')
                tasks.append(task_id)
        return sorted(tasks)

    def _extract_verifier_codes(self):
        codes = {}
        with open('verifiers.py', 'r') as f:
            content = f.read()
        tree = ast.parse(content)
        lines = content.splitlines()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('verify_'):
                task_id = node.name.replace('verify_', '')
                start_line = node.body[0].lineno - 1
                end_line = node.end_lineno
                body_lines = lines[start_line:end_line]
                if body_lines:
                    indent = len(body_lines[0]) - len(body_lines[0].lstrip())
                    body_lines = [line[indent:] for line in body_lines]
                codes[task_id] = '\n'.join(body_lines)
        return codes

    def _grid_exceeds_size(self, grid, max_dim=30):
        if not grid:
            return False
        return len(grid) > max_dim or any(len(row) > max_dim for row in grid)

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        # Deterministic task selection based on index
        task_id = self.tasks[idx]
        
        generator = getattr(generators, f'generate_{task_id}')
        verifier = getattr(verifiers, f'verify_{task_id}', None)
        if verifier is None:
            return self.__getitem__((idx + 1) % len(self))
        
        for attempt in range(5):
            try:
                example = generator(self.diff_lb, self.diff_ub)
                input_grid = example['input']
                output_grid = example['output']
                if self._grid_exceeds_size(input_grid) or self._grid_exceeds_size(output_grid):
                    print(f"[Dataset] Skipping task {task_id}: grid exceeds {30}x{30} (attempt {attempt+1}/5)")
                    continue
                if input_grid == output_grid:
                    print(f"[Dataset] Skipping task {task_id}: input matches output (attempt {attempt+1}/5)")
                    continue
                if verifier(input_grid) != output_grid:
                    print(f"[Dataset] Verifier mismatch for task {task_id} (attempt {attempt+1}/5)")
                    continue
                break
            except Exception:
                continue
        else:
            # If generation fails, pick a random OTHER task to keep batch flowing
            return self.__getitem__((idx + 1) % len(self))

        code = self.verifier_codes.get(task_id, "")
        if not code:
            return self.__getitem__((idx + 1) % len(self))

        input_tokens = self.tokenizer.encode_grid(input_grid)
        output_tokens = self.tokenizer.encode_grid(output_grid)
        
        src_ids = [self.tokenizer.bos_token_id] + \
                  input_tokens + \
                  [self.tokenizer.sep_token_id] + \
                  output_tokens + \
                  [self.tokenizer.eos_token_id]
                  
        code_ids = self.tokenizer.encode_code(code)
        tgt_ids = [self.tokenizer.bos_token_id] + \
                  code_ids + \
                  [self.tokenizer.eos_token_id]
                  
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_list, tgt_list = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=0)
    return src_padded, tgt_padded
