import torch
from torch.utils.data import Dataset
import generators
import inspect
import re
import ast
import random
from tokenizer import DSLTokenizer

class ARCDataset(Dataset):
    def __init__(self, epoch_size=1000, diff_lb=0.0, diff_ub=1.0):
        """
        epoch_size: How many examples to generate per 'epoch' (since data is infinite)
        diff_lb, diff_ub: Difficulty bounds for the generators
        """
        self.epoch_size = epoch_size
        self.diff_lb = diff_lb
        self.diff_ub = diff_ub
        
        self.tokenizer = DSLTokenizer()
        self.tasks = self._get_available_tasks()
        self.verifier_codes = self._extract_verifier_codes()
        
    def _get_available_tasks(self):
        # Find all tasks that have both a generator and a verifier
        # We scan generators.py
        tasks = []
        for name in dir(generators):
            if name.startswith('generate_'):
                task_id = name.replace('generate_', '')
                tasks.append(task_id)
        return sorted(tasks)

    def _extract_verifier_codes(self):
        """
        Reads verifiers.py and extracts the function body for each task.
        Returns a dict: {task_id: code_string}
        """
        codes = {}
        with open('verifiers.py', 'r') as f:
            content = f.read()
            
        # Parse the file to find function definitions
        tree = ast.parse(content)
        
        lines = content.splitlines()
        
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith('verify_'):
                task_id = node.name.replace('verify_', '')
                
                # Extract the body
                # We get the line numbers and slice the original content
                # This preserves formatting which might be nice, though we tokenize anyway
                start_line = node.body[0].lineno - 1
                end_line = node.end_lineno
                
                body_lines = lines[start_line:end_line]
                # Dedent
                if body_lines:
                    indent = len(body_lines[0]) - len(body_lines[0].lstrip())
                    body_lines = [line[indent:] for line in body_lines]
                    
                codes[task_id] = '\n'.join(body_lines)
        return codes

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        # 1. Pick a random task
        task_id = random.choice(self.tasks)
        
        # 2. Generate a grid pair
        generator = getattr(generators, f'generate_{task_id}')
        
        # Retry logic in case generation fails (rare but possible)
        for _ in range(5):
            try:
                example = generator(self.diff_lb, self.diff_ub)
                input_grid = example['input']
                output_grid = example['output']
                break
            except Exception:
                continue
        else:
            # Fallback if generation fails repeatedly
            # Pick another task? or just return empty/pad
            return self.__getitem__((idx + 1) % len(self))

        # 3. Get the code
        code = self.verifier_codes.get(task_id, "")
        if not code:
            # If no code found (shouldn't happen), recurse
            return self.__getitem__((idx + 1) % len(self))

        # 4. Tokenize
        # Source: [BOS] Input [SEP] Output [EOS]
        input_tokens = self.tokenizer.encode_grid(input_grid)
        output_tokens = self.tokenizer.encode_grid(output_grid)
        
        src_ids = [self.tokenizer.bos_token_id] + \
                  input_tokens + \
                  [self.tokenizer.sep_token_id] + \
                  output_tokens + \
                  [self.tokenizer.eos_token_id]
                  
        # Target: [BOS] Code [EOS]
        code_ids = self.tokenizer.encode_code(code)
        tgt_ids = [self.tokenizer.bos_token_id] + \
                  code_ids + \
                  [self.tokenizer.eos_token_id]
                  
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    """
    Pads batch to the max length in the batch.
    """
    src_list, tgt_list = zip(*batch)
    
    # Pad source
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0) # Assuming 0 is [PAD]
    
    # Pad target
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=0)
    
    return src_padded, tgt_padded
