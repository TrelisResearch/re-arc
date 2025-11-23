import json
from tokenizer import DSLTokenizer

tokenizer = DSLTokenizer()

vocab = {
    "tokens": list(tokenizer.token_to_id.keys()),
    "vocab_size": tokenizer.vocab_size
}

with open('vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)

print(f"Saved {tokenizer.vocab_size} tokens to vocab.json")
