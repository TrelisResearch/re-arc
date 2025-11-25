import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ARCDataset, collate_fn
from tokenizer import DSLTokenizer
import math
import argparse
import os
import yaml
import ast
import dsl
from dsl import *
from tqdm import tqdm
from contextlib import nullcontext

try:
    import wandb
except ImportError:
    wandb = None

class DecoderOnlyDSLTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_head=8, num_layers=16, num_recursions=1, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_recursions = num_recursions

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, mask=None, padding_mask=None, past_key_values=None, use_cache=False, position_ids=None):
        """
        Decoder-only forward pass (GPT-style) with optional recursion.
        Args:
            tokens: [batch, seq_len] token IDs
            mask: [seq_len, seq_len] causal attention mask
            padding_mask: [batch, seq_len] padding mask
            past_key_values: list of cached (k, v) tuples
            use_cache: whether to return cache for generation
            position_ids: absolute position indices for tokens
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embed and add positions
        x = self.embedding(tokens)
        x = self.pos_encoder(x, position_ids=position_ids)

        total_layers = self.num_layers * self.num_recursions
        if past_key_values is None:
            past_key_values = [None] * total_layers

        presents = [] if use_cache else None
        layer_offset = 0

        for _ in range(self.num_recursions):
            for layer_idx, layer in enumerate(self.layers):
                cache_idx = layer_offset + layer_idx
                past = past_key_values[cache_idx] if cache_idx < len(past_key_values) else None
                x, present = layer(
                    x,
                    attn_mask=mask,
                    key_padding_mask=padding_mask,
                    past_key_value=past,
                    use_cache=use_cache
                )
                if use_cache:
                    presents.append(present)
            layer_offset += self.num_layers

        x = self.final_norm(x)
        logits = self.fc_out(x)
        if use_cache:
            return logits, presents
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        position_ids = position_ids.clamp(0, self.pe.size(0) - 1)
        pe = self.pe[position_ids]
        x = x + pe
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None, past_key_value=None, use_cache=False):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask_ = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask_ = attn_mask.unsqueeze(1)
            else:
                raise ValueError("Unsupported attn_mask dimension")
            attn_scores = attn_scores.masked_fill(attn_mask_, float('-inf'))

        if key_padding_mask is not None:
            padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        present = (k, v) if use_cache else None
        return attn_output, present


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, attn_mask=None, key_padding_mask=None, past_key_value=None, use_cache=False):
        residual = x
        normed = self.norm1(x)
        attn_output, present = self.self_attn(
            normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        x = residual + self.dropout(attn_output)

        residual = x
        normed = self.norm2(x)
        ff_output = self.linear2(self.activation(self.linear1(normed)))
        x = residual + self.dropout(ff_output)

        return x, present


def clean_state_dict(state_dict):
    prefixes = ['_orig_mod.', 'module.']
    cleaned = dict(state_dict)
    for prefix in prefixes:
        cleaned = {
            (key[len(prefix):] if key.startswith(prefix) else key): value
            for key, value in cleaned.items()
        }
    return cleaned


def create_arc_dataset(cfg):
    return ARCDataset(
        diff_lb=cfg['dataset']['diff_lb'],
        diff_ub=cfg['dataset']['diff_ub']
    )


def create_dataloader(cfg, dataset, device, shuffle=True):
    num_workers = cfg['training'].get('num_workers', 4)
    pin_mem = device.type == 'cuda'
    persistent = num_workers > 0 and pin_mem
    return DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent
    )


def get_autocast_context(use_amp, amp_dtype):
    return torch.amp.autocast('cuda', dtype=amp_dtype) if use_amp else nullcontext()


def generate_square_subsequent_mask(sz):
    """Generate causal mask for autoregressive decoding (boolean version for PyTorch 2.0+)"""
    mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
    return mask  # True = masked (ignore), False = attend

def reconstruct_grid_from_tokens(token_ids, tokenizer):
    # Mapping from DSL constant names back to integers
    const_to_color = {
        'ZERO': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4,
        'FIVE': 5, 'SIX': 6, 'SEVEN': 7, 'EIGHT': 8, 'NINE': 9
    }

    tokens = [tokenizer.id_to_token[t] for t in token_ids if t not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.eos_token_id]]
    grid = []
    current_row = []
    for t in tokens:
        if t == '[ROW]':
            if current_row:
                grid.append(tuple(current_row))
            current_row = []
        elif t in const_to_color:
            # Map DSL constant back to integer
            current_row.append(const_to_color[t])
    if current_row:
        grid.append(tuple(current_row))
    return tuple(grid)

def execute_and_score(generated_code, input_grid, target_grid):
    try:
        ast.parse(generated_code)
    except SyntaxError:
        return False, False, False

    if 'return' not in generated_code:
        return True, False, False

    wrapped_code = f"def solver(I):\n"
    for line in generated_code.split('\n'):
        wrapped_code += f"    {line}\n"
        
    local_scope = {}
    global_scope = {k: getattr(dsl, k) for k in dir(dsl) if not k.startswith('__')}
    
    try:
        exec(wrapped_code, global_scope, local_scope)
        solver = local_scope['solver']
        prediction = solver(input_grid)
        if prediction == target_grid:
            return True, True, True
        else:
            return True, True, False
    except Exception:
        return True, False, False

def calculate_metrics(logits, targets, pad_idx):
    preds = torch.argmax(logits, dim=-1)
    mask = (targets != pad_idx)
    correct_tokens = (preds == targets) & mask
    
    total_valid_tokens = mask.sum().item()
    total_correct = correct_tokens.sum().item()
    token_acc = total_correct / total_valid_tokens if total_valid_tokens > 0 else 0.0
    
    wrong_tokens = (preds != targets) & mask
    row_has_error = wrong_tokens.any(dim=1)
    exact_match_acc = (~row_has_error).float().mean().item()
    
    return token_acc, exact_match_acc

def run_generation_batch(model, src_batch, tokenizer, device, max_len):
    """Parallel batch generation for decoder-only model"""
    model.eval()
    batch_size = src_batch.size(0)

    with torch.no_grad():
        curr_tokens = src_batch.to(device)
        past_key_values = None
        generated = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            if past_key_values is None:
                seq_len = curr_tokens.size(1)
                causal_mask = generate_square_subsequent_mask(seq_len).to(device)
                padding_mask = (curr_tokens == tokenizer.pad_token_id)
                logits, past_key_values = model(
                    curr_tokens,
                    mask=causal_mask,
                    padding_mask=padding_mask,
                    use_cache=True
                )
            else:
                past_seq_len = past_key_values[0][0].size(2)
                position_ids = torch.full(
                    (batch_size, curr_tokens.size(1)),
                    past_seq_len,
                    dtype=torch.long,
                    device=device
                )
                logits, past_key_values = model(
                    curr_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=position_ids
                )

            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            next_tokens = torch.where(
                finished,
                torch.full_like(next_tokens, tokenizer.eos_token_id),
                next_tokens
            )

            for i in range(batch_size):
                if finished[i]:
                    continue
                token_id = next_tokens[i].item()
                if token_id == tokenizer.eos_token_id:
                    finished[i] = True
                else:
                    generated[i].append(token_id)

            if finished.all():
                break

            curr_tokens = next_tokens.unsqueeze(1)

        results = [tokenizer.decode(tokens) for tokens in generated]

    model.train()
    return results

def run_validation(model, dataloader, tokenizer, device, num_examples, global_step, max_gen_len):
    print(f"\n--- Running Validation on {num_examples} examples ---")
    
    total_loss = 0
    total_token_acc = 0
    
    syntax_valid_count = 0
    runtime_success_count = 0
    correct_count = 0
    processed_count = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    model.eval()
    # Iterate batch by batch until we hit num_examples
    pbar = tqdm(enumerate(dataloader), total=min(len(dataloader), (num_examples + dataloader.batch_size - 1) // dataloader.batch_size), desc="Validation")
    for batch_idx, (src, tgt) in pbar:
        if processed_count >= num_examples:
            break
            
        # 1. Standard Metrics (Loss/Acc) - on the whole batch
        with torch.no_grad():
            src_batch, tgt_batch = src.to(device), tgt.to(device)

            # Decoder-only: concatenate src and tgt
            tokens_batch = torch.cat([src_batch, tgt_batch], dim=1)
            input_tokens_batch = tokens_batch[:, :-1]
            target_tokens_batch = tokens_batch[:, 1:]

            seq_len_batch = input_tokens_batch.size(1)
            causal_mask_batch = generate_square_subsequent_mask(seq_len_batch).to(device)
            padding_mask_batch = (input_tokens_batch == tokenizer.pad_token_id)

            logits_batch = model(input_tokens_batch, mask=causal_mask_batch, padding_mask=padding_mask_batch)
            loss_batch = criterion(logits_batch.reshape(-1, logits_batch.shape[-1]), target_tokens_batch.reshape(-1))
            token_acc_batch, _ = calculate_metrics(logits_batch, target_tokens_batch, tokenizer.pad_token_id)

            total_loss += loss_batch.item()
            total_token_acc += token_acc_batch
        
        # 2. Batch Generation & Execution
        batch_size_actual = min(src.size(0), num_examples - processed_count)
        if batch_size_actual <= 0:
            break

        src_batch_slice = src_batch[:batch_size_actual]
        generated_codes = run_generation_batch(model, src_batch_slice, tokenizer, device, max_len=max_gen_len)

        batch_syn, batch_run, batch_corr = 0, 0, 0
        for j in range(batch_size_actual):
            try:
                src_cpu = src_batch[j].cpu().tolist()

                # Reconstruct Inputs
                try:
                    sep_idx = src_cpu.index(tokenizer.sep_token_id)
                    input_ids = src_cpu[1:sep_idx]
                    input_grid = reconstruct_grid_from_tokens(input_ids, tokenizer)

                    output_ids = src_cpu[sep_idx+1:]
                    if tokenizer.eos_token_id in output_ids:
                        output_ids = output_ids[:output_ids.index(tokenizer.eos_token_id)]
                    target_grid = reconstruct_grid_from_tokens(output_ids, tokenizer)

                    # Execute
                    code = generated_codes[j]
                    is_syn, is_run, is_corr = execute_and_score(code, input_grid, target_grid)

                    if is_syn:
                        syntax_valid_count += 1
                        batch_syn += 1
                    if is_run:
                        runtime_success_count += 1
                        batch_run += 1
                    if is_corr:
                        correct_count += 1
                        batch_corr += 1
                    processed_count += 1

                    if processed_count <= 3:
                        print(f"\n--- Generated Code Sample {processed_count} (batch={batch_idx}, j={j}) ---")
                        print(f"SEP index: {sep_idx}, Total source length: {len(src_cpu)}")
                        print(f"Input token IDs (BOS to SEP): {src_cpu[1:sep_idx][:30]}...")
                        print(f"Decoded input tokens: {[tokenizer.id_to_token.get(t, '?') for t in src_cpu[1:sep_idx][:30]]}")
                        print(f"Input grid shape: {len(input_grid)}x{len(input_grid[0]) if input_grid else 0}")
                        if input_grid:
                            print(f"Input grid preview: {input_grid[:2]}")

                        # Check for newlines in generated code
                        newline_check = code.count('\n')
                        print(f"Generated code length: {len(code)}, Newlines: {newline_check}")
                        print("=== FULL GENERATED CODE ===")
                        print(code)
                        print("=== END CODE ===")
                        print(f"Syn={is_syn}, Run={is_run}, Corr={is_corr}\n")

                except ValueError:
                    continue
            except Exception as e:
                # print(f"Val Error (execution): {e}") # Too verbose
                continue

        # Update progress bar with cumulative rates
        pbar.set_postfix({
            'processed': f'{processed_count}/{num_examples}',
            'syn': f'{syntax_valid_count}/{processed_count}' if processed_count else '0/0',
            'run': f'{runtime_success_count}/{processed_count}' if processed_count else '0/0',
            'corr': f'{correct_count}/{processed_count}' if processed_count else '0/0'
        })

    pbar.close()

    # Final Stats
    num_batches_for_std_metrics = batch_idx + 1
    avg_loss = total_loss / num_batches_for_std_metrics if num_batches_for_std_metrics else 0
    avg_token_acc = total_token_acc / num_batches_for_std_metrics if num_batches_for_std_metrics else 0

    syn_rate = syntax_valid_count / processed_count if processed_count else 0
    run_rate = runtime_success_count / processed_count if processed_count else 0
    corr_rate = correct_count / processed_count if processed_count else 0

    model.train()
    
    if wandb and wandb.run: # Log to wandb if active
        wandb.log({
            "val/loss": avg_loss,
            "val/token_acc": avg_token_acc,
            "val/syntax_rate": syn_rate,
            "val/runtime_rate": run_rate,
            "val/correct_rate": corr_rate,
            "_step": global_step # Use global step for validation logs
        })
    
    return avg_loss, avg_token_acc, syn_rate, run_rate, corr_rate

def train():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--use_wandb', action='store_true', help='Override config to enable wandb')
    parser.add_argument('--limit_batches', type=int, default=None, help='For testing: limit training batches per epoch')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if cfg['training']['device'] == "auto":
        if torch.backends.mps.is_available(): DEVICE = torch.device("mps")
        elif torch.cuda.is_available(): DEVICE = torch.device("cuda")
        else: DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(cfg['training']['device'])

    # Enable TF32 for better performance on Ampere/Hopper GPUs
    if DEVICE.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        print("Enabled TF32 for matmul operations")

    print(f"Using Device: {DEVICE}")

    use_wandb = cfg['wandb']['enabled'] or args.use_wandb
    if use_wandb and wandb:
        wandb.init(project=cfg['wandb']['project'], config=cfg)

    dataset = create_arc_dataset(cfg)
    val_dataset = create_arc_dataset(cfg)
    dataloader = create_dataloader(cfg, dataset, DEVICE, shuffle=True)
    val_dataloader = create_dataloader(cfg, val_dataset, DEVICE, shuffle=True)
    
    tokenizer = dataset.tokenizer

    model = DecoderOnlyDSLTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg['model']['d_model'],
        n_head=cfg['model']['n_head'],
        num_layers=cfg['model']['num_layers'],
        num_recursions=cfg['model']['num_recursions'],
        dropout=cfg['model']['dropout']
    ).to(DEVICE)

    save_dir = cfg['checkpoint']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_val_correct = 0.0
    global_step = 0 # To track total optimizer steps for WandB
    start_epoch = 0
    optimizer_state = None

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        state_dict = clean_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(state_dict)
        optimizer_state = checkpoint.get('optimizer_state_dict', None)
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        best_val_correct = checkpoint.get('best_val_correct', 0.0)
        print(f"Resumed from epoch {start_epoch}, global_step {global_step}")

    # Compile model for speedup (PyTorch 2.0+)
    if hasattr(torch, 'compile') and cfg.get('training', {}).get('use_torch_compile', False):
        try:
            print("Compiling model with torch.compile...")
            model = torch.compile(model, dynamic=True)
            print("Model compiled successfully!")
        except Exception as e:
            print(f"torch.compile failed: {e}")
            print("Continuing without compilation...")

    optimizer = optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Mixed precision training (BF16 on CUDA for better stability on Ampere/Hopper)
    use_amp = DEVICE.type == 'cuda'
    amp_dtype = torch.bfloat16
    if use_amp:
        print(f"Using AMP with dtype: {amp_dtype}")

    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Gradient accumulation
    gradient_accumulation_steps = cfg['training'].get('gradient_accumulation_steps', 1)
    effective_batch_size = cfg['training']['batch_size'] * gradient_accumulation_steps
    print(f"Batch size: {cfg['training']['batch_size']}, Accumulation steps: {gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")

    model.train()
    for epoch in range(start_epoch, cfg['training']['epochs']):
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at start of epoch

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        for batch_idx, (src, tgt) in pbar:
            if args.limit_batches and batch_idx >= args.limit_batches:
                break

            # Decoder-only: concatenate src and tgt into single sequence
            # src already contains: [BOS] input_grid [SEP] output_grid [SEP]
            # tgt contains: code tokens [EOS]
            # Concatenate them: [BOS] input_grid [SEP] output_grid [SEP] code [EOS]
            tokens = torch.cat([src, tgt], dim=1).to(DEVICE)

            # Shift for next-token prediction
            input_tokens = tokens[:, :-1]   # Everything except last token
            target_tokens = tokens[:, 1:]   # Everything except first token

            # Create causal mask and padding mask
            seq_len = input_tokens.size(1)
            causal_mask = generate_square_subsequent_mask(seq_len).to(DEVICE)
            padding_mask = (input_tokens == tokenizer.pad_token_id)

            # Mixed precision forward pass (BF16 doesn't need GradScaler)
            with get_autocast_context(use_amp, amp_dtype):
                logits = model(input_tokens, mask=causal_mask, padding_mask=padding_mask)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), target_tokens.reshape(-1))
                # Scale loss by accumulation steps for correct gradient magnitude
                loss = loss / gradient_accumulation_steps
            loss.backward()

            # Accumulate the unscaled loss for logging
            total_loss += loss.item() * gradient_accumulation_steps

            # Only update weights every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Update progress bar (show actual loss, not scaled)
                pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}', 'grad_norm': f'{grad_norm:.4f}'})

                if use_wandb:
                    wandb.log({
                        "train/loss": loss.item() * gradient_accumulation_steps,
                        "train/grad_norm": grad_norm,
                        "_step": global_step
                    })
            else:
                # Just show loss without grad_norm during accumulation
                pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}', 'accumulating': f'{(batch_idx + 1) % gradient_accumulation_steps}/{gradient_accumulation_steps}'})

        pbar.close()
        avg_loss = total_loss / (batch_idx + 1) if (batch_idx + 1) else 0
        print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % cfg['validation']['interval_epochs'] == 0:
            val_loss, val_token_acc, syn_rate, run_rate, corr_rate = run_validation(
                model, val_dataloader, tokenizer, DEVICE,
                num_examples=cfg['validation']['num_examples'],
                global_step=global_step,
                max_gen_len=cfg['validation']['max_gen_len']
            )
            
            print(f"VAL >> Loss: {val_loss:.4f} | TokAcc: {val_token_acc:.2%} | Syn: {syn_rate:.1%} | Run: {run_rate:.1%} | Corr: {corr_rate:.1%}")
            
            if cfg['checkpoint']['keep_best'] and corr_rate >= best_val_correct:
                best_val_correct = corr_rate
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'global_step': global_step,
                    'best_val_correct': best_val_correct
                }
                torch.save(checkpoint, os.path.join(save_dir, "model_best.pt"))
                print("Saved Best Model!")

        if (epoch + 1) % cfg['checkpoint']['save_every_n_epochs'] == 0:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'global_step': global_step,
                'best_val_correct': best_val_correct
            }
            torch.save(checkpoint, os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    train()
