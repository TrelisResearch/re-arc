import re
import dsl

class DSLTokenizer:
    def __init__(self, verifiers_path='verifiers.py'):
        self.verifiers_path = verifiers_path
        self.token_to_id = {}
        self.id_to_token = {}
        self.build_vocab()
        
    def build_vocab(self):
        # 1. Special Tokens
        special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[SEP]', '[ROW]']
        
        # 2. Primitives from DSL
        dsl_tokens = sorted([name for name in dir(dsl) if not name.startswith('__')])
        
        # 3. Tokens found in verifiers (variables, integers, syntax)
        # We scan the file to ensure we catch everything actually used
        verifier_tokens = set()
        token_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+|[(),=]')
        
        with open(self.verifiers_path, 'r') as f:
            content = f.read()
            
        matches = token_pattern.findall(content)
        for m in matches:
            if m in ['def', 'return', 'verify_']: 
                # keep return, discard def and verify_ prefix logic later
                if m == 'return':
                    verifier_tokens.add(m)
                continue
            if m.startswith('verify_'):
                continue
            verifier_tokens.add(m)
            
        # Combine everything
        all_tokens = special_tokens + dsl_tokens + sorted(list(verifier_tokens))
        
        # Remove duplicates while preserving order
        unique_tokens = []
        seen = set()
        for t in all_tokens:
            if t not in seen:
                unique_tokens.append(t)
                seen.add(t)
                
        # Build maps
        for idx, token in enumerate(unique_tokens):
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
            
        self.vocab_size = len(unique_tokens)
        self.pad_token_id = self.token_to_id['[PAD]']
        self.bos_token_id = self.token_to_id['[BOS]']
        self.eos_token_id = self.token_to_id['[EOS]']
        self.sep_token_id = self.token_to_id['[SEP]']
        self.row_token_id = self.token_to_id['[ROW]']
        
    def encode_grid(self, grid):
        """
        Flattens a grid into a sequence of token IDs.
        Format: 0 0 1 [ROW] 2 2 0 [ROW] ...
        """
        tokens = []
        for row in grid:
            for cell in row:
                # cell is an integer, convert to string token
                tokens.append(str(cell))
            tokens.append('[ROW]')
        # Remove last [ROW]
        if tokens:
            tokens.pop()
        
        return [self.token_to_id.get(t, self.token_to_id['[PAD]']) for t in tokens]

    def encode_code(self, code_string):
        """
        Tokenizes a Python code string (the body of a verifier).
        """
        # Simple regex tokenizer matching the vocab building strategy
        token_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+|[(),=]')
        raw_tokens = token_pattern.findall(code_string)
        
        # Filter identifiers that might not be in vocab (shouldn't happen if vocab is complete)
        # and map to IDs
        ids = []
        for t in raw_tokens:
            if t in self.token_to_id:
                ids.append(self.token_to_id[t])
            else:
                # If unknown token (rare), we could skip or use UNK. 
                # For this closed system, we assume vocab is complete.
                pass
        return ids

    def decode(self, token_ids):
        """
        Converts IDs back to a string.
        """
        tokens = []
        for tid in token_ids:
            if tid == self.eos_token_id:
                break
            if tid in [self.pad_token_id, self.bos_token_id]:
                continue
            tokens.append(self.id_to_token.get(tid, ''))
            
        # Heuristic to make it look like code again
        # This is a bit rough, primarily for debugging
        return ' '.join(tokens)
