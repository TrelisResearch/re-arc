"""
Utility to download a trained RE-ARC model from Hugging Face Hub.

Usage:
    uv run utils/pull_from_hf.py <repo_id> --checkpoint <checkpoint.pt>
    uv run utils/pull_from_hf.py Trelis/re-arc-model --checkpoint model_epoch_1000.pt
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from train import DecoderOnlyDSLTransformer
from tokenizer import DSLTokenizer


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def download_model(repo_id: str, output_dir: str = None, checkpoint_name: str = None):
    """
    Download a model from Hugging Face Hub.

    Args:
        repo_id: HF repo ID (e.g., "username/model-name")
        output_dir: Local directory to save the model
        checkpoint_name: Specific checkpoint file to download (e.g., "checkpoint_epoch_1000.pt")
                        If None, downloads the entire repo
    """
    output_path = Path(output_dir or f"./models/{repo_id.split('/')[-1]}")
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from {repo_id} to {output_path}")

    if checkpoint_name:
        # Download specific checkpoint file
        print(f"Downloading checkpoint: {checkpoint_name}")
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_name,
            local_dir=output_path
        )
        print(f"✓ Downloaded checkpoint to: {checkpoint_path}")
    else:
        # Download entire repo
        print("Downloading entire repository...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=output_path,
            local_dir_use_symlinks=False
        )
        print(f"✓ Downloaded repository to: {output_path}")

    return output_path


def load_model_from_checkpoint(checkpoint_path: str, device: str = None):
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        device: Device to load model on

    Returns:
        model, tokenizer, metadata dict
    """
    print(f"Loading model from {checkpoint_path}")

    # Auto-detect device if not specified
    if device is None:
        device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize tokenizer
    tokenizer = DSLTokenizer()

    # Get model config from checkpoint or use defaults
    config = checkpoint.get('config', {})
    vocab_size = config.get('vocab_size', tokenizer.vocab_size)
    d_model = config.get('d_model', 512)
    n_head = config.get('n_head', 8)
    num_layers = config.get('num_layers', 1)
    num_recursions = config.get('num_recursions', 16)

    # Initialize model
    model = DecoderOnlyDSLTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        num_recursions=num_recursions
    ).to(device)

    # Load weights (handle torch.compile prefix)
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', -1),
        'global_step': checkpoint.get('global_step', -1),
        'config': config
    }

    print(f"✓ Loaded model from epoch {metadata['epoch']}, step {metadata['global_step']}")
    print(f"  Model: {d_model}d, {n_head} heads, {num_layers} layers × {num_recursions} recursions")

    return model, tokenizer, metadata


def main():
    parser = argparse.ArgumentParser(description="Download RE-ARC model from Hugging Face")
    parser.add_argument("repo_id", type=str, help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Specific checkpoint file to download (e.g., checkpoint_epoch_1000.pt)")
    parser.add_argument("--load", action="store_true",
                       help="Also load the model into memory (requires checkpoint)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to load model on (auto-detects cuda/mps/cpu if not specified)")

    args = parser.parse_args()

    # Download
    output_path = download_model(args.repo_id, args.output, args.checkpoint)

    # Optionally load
    if args.load:
        if args.checkpoint:
            checkpoint_path = output_path / args.checkpoint
        else:
            # Find first .pt file
            checkpoints = list(output_path.glob("*.pt"))
            if not checkpoints:
                print("No checkpoint files found. Use --checkpoint to specify one.")
                return
            checkpoint_path = checkpoints[0]
            print(f"Using checkpoint: {checkpoint_path}")

        model, tokenizer, metadata = load_model_from_checkpoint(str(checkpoint_path), args.device)
        print("\n✓ Model loaded successfully!")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Ready for inference")


if __name__ == "__main__":
    main()
