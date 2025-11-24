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
        special_tokens = ['[PAD]', '[BOS]', '[EOS]', '[SEP]', '[ROW]', '[NEWLINE]']

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
        self.newline_token_id = self.token_to_id['[NEWLINE]']
        
    def encode_grid(self, grid):
        """
        Flattens a grid into a sequence of token IDs.
        Maps color integers to DSL constant names (0->ZERO, 1->ONE, etc.)
        Format: ZERO ZERO ONE [ROW] TWO TWO ZERO [ROW] ...
        """
        # Mapping from integer colors to DSL constant names
        color_to_const = {
            0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR',
            5: 'FIVE', 6: 'SIX', 7: 'SEVEN', 8: 'EIGHT', 9: 'NINE'
        }

        tokens = []
        for row in grid:
            for cell in row:
                # Map cell integer to DSL constant name
                const_name = color_to_const.get(cell, 'ZERO')
                tokens.append(const_name)
            tokens.append('[ROW]')
        # Remove last [ROW]
        if tokens:
            tokens.pop()

        return [self.token_to_id.get(t, self.token_to_id['[PAD]']) for t in tokens]

    def encode_code(self, code_string):
        """
        Tokenizes a Python code string (the body of a verifier).
        """
        # Tokenizer that captures newlines
        token_pattern = re.compile(r'[a-zA-Z_][a-zA-Z0-9_]*|\d+|[(),=]|\n')
        raw_tokens = token_pattern.findall(code_string)

        # Map to IDs, converting \n to [NEWLINE] token
        ids = []
        for t in raw_tokens:
            if t == '\n':
                ids.append(self.newline_token_id)
            elif t in self.token_to_id:
                ids.append(self.token_to_id[t])
            else:
                # Unknown token encountered - this suggests vocab incompleteness
                print(f"WARNING: Unknown token '{t}' encountered during encoding. Skipping.")
                pass
        return ids

    def decode(self, token_ids):
        """
        Converts IDs back to a string with proper newlines.
        """
        result = []
        for tid in token_ids:
            if tid == self.eos_token_id:
                break
            if tid in [self.pad_token_id, self.bos_token_id]:
                continue

            token = self.id_to_token.get(tid, '')
            if tid == self.newline_token_id:
                result.append('\n')
            else:
                result.append(token)

        # Join with spaces, but preserve newlines
        output = []
        current_line = []
        for token in result:
            if token == '\n':
                output.append(' '.join(current_line))
                output.append('\n')
                current_line = []
            else:
                current_line.append(token)

        if current_line:
            output.append(' '.join(current_line))

        return ''.join(output)
