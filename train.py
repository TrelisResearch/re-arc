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

try:
    import wandb
except ImportError:
    wandb = None

class RecursiveDSLTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_head=8, num_recursions=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_recursions = num_recursions
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.encoder_input_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.recursive_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=2048, dropout=dropout, batch_first=True)
        
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=2048, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=2) 
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None):
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        memory = self.encoder_input_layer(src_emb, src_key_padding_mask=src_padding_mask)
        
        for _ in range(self.num_recursions):
            memory = self.recursive_layer(memory, src_key_padding_mask=src_padding_mask)
            
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                              tgt_key_padding_mask=tgt_padding_mask,
                              memory_key_padding_mask=src_padding_mask)
        
        return self.fc_out(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def reconstruct_grid_from_tokens(token_ids, tokenizer):
    tokens = [tokenizer.id_to_token[t] for t in token_ids if t not in [tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.sep_token_id, tokenizer.eos_token_id]]
    grid = []
    current_row = []
    for t in tokens:
        if t == '[ROW]':
            if current_row:
                grid.append(tuple(current_row))
            current_row = []
        else:
            try:
                current_row.append(int(t))
            except:
                pass
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

def run_generation(model, src, tokenizer, device, max_len=100):
    model.eval()
    with torch.no_grad():
        src_emb = model.pos_encoder(model.embedding(src) * math.sqrt(model.d_model))
        memory = model.encoder_input_layer(src_emb)
        for _ in range(model.num_recursions):
            memory = model.recursive_layer(memory)
            
        curr_tgt = torch.tensor([[tokenizer.bos_token_id]], device=device)
        pred_tokens = []
        
        for _ in range(max_len): 
            tgt_emb = model.pos_encoder(model.embedding(curr_tgt) * math.sqrt(model.d_model))
            tgt_mask = generate_square_subsequent_mask(curr_tgt.size(1)).to(device)
            output = model.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            next_token = torch.argmax(model.fc_out(output[:, -1, :]), dim=-1).item()
            
            if next_token == tokenizer.eos_token_id:
                break
                
            pred_tokens.append(next_token)
            next_token_tensor = torch.tensor([[next_token]], device=device)
            curr_tgt = torch.cat([curr_tgt, next_token_tensor], dim=1)
    
    model.train()
    return tokenizer.decode(pred_tokens)

def run_validation(model, dataloader, tokenizer, device, num_examples):
    print(f"\n--- Running Validation on {num_examples} examples ---")
    
    total_loss = 0
    total_token_acc = 0
    
    syntax_valid_count = 0
    runtime_success_count = 0
    correct_count = 0
    processed_count = 0
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    for batch_idx, (src, tgt) in enumerate(dataloader):
        if processed_count >= num_examples:
            break
            
        with torch.no_grad():
            src, tgt = src.to(device), tgt.to(device)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            src_padding_mask = (src == tokenizer.pad_token_id)
            tgt_padding_mask = (tgt_input == tokenizer.pad_token_id)
            
            logits = model(src, tgt_input, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            token_acc, _ = calculate_metrics(logits, tgt_output, tokenizer.pad_token_id)
            
            total_loss += loss.item()
            total_token_acc += token_acc
        
        for j in range(src.size(0)):
            if processed_count >= num_examples:
                break
            
            try:
                src_item = src[j:j+1].to(device)
                src_cpu = src[j].cpu().tolist()
                
                try:
                    sep_idx = src_cpu.index(tokenizer.sep_token_id)
                    input_ids = src_cpu[1:sep_idx]
                    input_grid = reconstruct_grid_from_tokens(input_ids, tokenizer)
                    
                    output_ids = src_cpu[sep_idx+1:]
                    if tokenizer.eos_token_id in output_ids:
                        output_ids = output_ids[:output_ids.index(tokenizer.eos_token_id)]
                    target_grid = reconstruct_grid_from_tokens(output_ids, tokenizer)
                    
                    code = run_generation(model, src_item, tokenizer, device)
                    
                    is_syn, is_run, is_corr = execute_and_score(code, input_grid, target_grid)
                    
                    if is_syn: syntax_valid_count += 1
                    if is_run: runtime_success_count += 1
                    if is_corr: correct_count += 1
                    processed_count += 1
                    
                    if processed_count <= 2: 
                        print(f"Ex {processed_count}: Syn={{is_syn}}, Run={{is_run}}, Corr={{is_corr}}")
                except ValueError:
                    continue 
            except Exception as e:
                print(f"Val Error: {e}")
                continue

    batches_run = batch_idx + 1 
    avg_loss = total_loss / batches_run if batches_run else 0
    avg_token_acc = total_token_acc / batches_run if batches_run else 0
    
    syn_rate = syntax_valid_count / processed_count if processed_count else 0
    run_rate = runtime_success_count / processed_count if processed_count else 0
    corr_rate = correct_count / processed_count if processed_count else 0
    
    return avg_loss, avg_token_acc, syn_rate, run_rate, corr_rate

def train():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--use_wandb', action='store_true', help='Override config to enable wandb')
    parser.add_argument('--limit_batches', type=int, default=None, help='For testing: limit training batches per epoch')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if cfg['training']['device'] == "auto":
        if torch.backends.mps.is_available(): DEVICE = torch.device("mps")
        elif torch.cuda.is_available(): DEVICE = torch.device("cuda")
        else: DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device(cfg['training']['device'])
    
    print(f"Using Device: {DEVICE}")

    use_wandb = cfg['wandb']['enabled'] or args.use_wandb
    if use_wandb and wandb:
        wandb.init(project=cfg['wandb']['project'], config=cfg)

    dataset = ARCDataset() 
    dataloader = DataLoader(dataset, batch_size=cfg['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    val_dataset = ARCDataset()
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    tokenizer = dataset.tokenizer
    
    model = RecursiveDSLTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg['model']['d_model'],
        n_head=cfg['model']['n_head'],
        num_recursions=cfg['model']['num_recursions'],
        dropout=cfg['model']['dropout']
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=float(cfg['training']['lr']))
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    save_dir = cfg['checkpoint']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_val_correct = 0.0

    model.train()
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0
        batches_processed = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            if args.limit_batches and batch_idx >= args.limit_batches:
                break
            
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input, tgt_output = tgt[:, :-1], tgt[:, 1:]
            
            tgt_mask = generate_square_subsequent_mask(tgt_input.size(1)).to(DEVICE)
            src_padding_mask = (src == tokenizer.pad_token_id)
            tgt_padding_mask = (tgt_input == tokenizer.pad_token_id)
            
            logits = model(src, tgt_input, tgt_mask=tgt_mask, src_padding_mask=src_padding_mask, tgt_padding_mask=tgt_padding_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batches_processed += 1
            
            if use_wandb:
                wandb.log({"train_loss": loss.item()})
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / batches_processed if batches_processed else 0
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % cfg['validation']['interval_epochs'] == 0:
            val_loss, val_token_acc, syn_rate, run_rate, corr_rate = run_validation(
                model, val_dataloader, tokenizer, DEVICE, 
                num_examples=cfg['validation']['num_examples']
            )
            
            print(f"VAL >> Loss: {val_loss:.4f} | TokAcc: {val_token_acc:.2%} | Syn: {syn_rate:.1%} | Run: {run_rate:.1%} | Corr: {corr_rate:.1%}")
            
            if use_wandb:
                wandb.log({
                    "val_loss": val_loss,
                    "val_token_acc": val_token_acc,
                    "val_syntax_rate": syn_rate,
                    "val_runtime_rate": run_rate,
                    "val_correct_rate": corr_rate,
                    "epoch": epoch + 1
                })
            
            if cfg['checkpoint']['keep_best'] and corr_rate >= best_val_correct:
                best_val_correct = corr_rate
                torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pt"))
                print("Saved Best Model!")

        if (epoch + 1) % cfg['checkpoint']['save_every_n_epochs'] == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pt"))

if __name__ == "__main__":
    train()
