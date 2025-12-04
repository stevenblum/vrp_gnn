import subprocess
import itertools
from pathlib import Path

def run_experiment():
    """Run training for all combinations of network parameters."""
    
    # Define parameter combinations
    learning_rates = [1e-4]
    embedding_dims = [256]
    num_encoder_layers = [6]
    num_attn_heads = [8]
    train_data_sizes = [1_000_000]
    batch_sizes = [64]
    limit_train_batches = [.2]
    train_decode_type = ["sampling"]
    train_num_starts = [0]
    val_decode_type = ["multistart_greedy"]
    
    # Base log directory
    base_log_dir = Path("lightning_logs/small_exp1")
    base_log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate all combinations
    combinations = list(itertools.product(learning_rates, embedding_dims, num_encoder_layers, num_attn_heads))
    
    # Save Combinations to a file
    combos_file = base_log_dir / "experiment_combinations.txt"
    with combos_file.open("w") as f:
        for lr, emb_dim, enc_layers, attn_heads in combinations:
            f.write(f"LR: {lr}, Embedding Dim: {emb_dim}, Encoder Layers: {enc_layers}, Attention Heads: {attn_heads}\n")

    print(f"Running {len(combinations)} experiments...")
    print("=" * 60)
    
    for i, (lr, emb_dim, enc_layers, attn_heads) in enumerate(combinations, 1):
        # Create folder name based on parameters
        model_name = f"lr{lr:.0e}_emb{emb_dim}_enc{enc_layers}_attn{attn_heads}"
        
        print(f"\n[{i}/{len(combinations)}] Training: {model_name}")
        print(f"  - Learning Rate: {lr}")
        print(f"  - Embedding Dim: {emb_dim}")
        print(f"  - Encoder Layers: {enc_layers}")
        print(f"  - Attention Heads: {attn_heads}")
        print(f"  - Train Data Size: {train_data_sizes[0]}")
        print(f"  - Batch Size: {batch_sizes[0]}")
        print(f"  - Limit Train Batches: {limit_train_batches[0]}")
        print(f"  - Train Num Starts: {train_num_starts}")
        print("-" * 60)
        
        # Build command
        cmd = [
            "python", "CVRP_Run.py",
            "--learning_rate", str(lr),
            "--embedding_dim", str(emb_dim),
            "--num_encoder_layers", str(enc_layers),
            "--num_attn_heads", str(attn_heads),
            "--train_data_size", str(train_data_sizes[0]),
            "--batch_size", str(batch_sizes[0]),
            "--limit_train_batches", str(limit_train_batches[0]),
            "--log_base_dir", base_log_dir.as_posix(),
            "--train_decode_type", train_decode_type[0],
            "--train_num_starts", str(train_num_starts[0]),
            "--val_decode_type", val_decode_type[0],

        ]
        
        # Run the training
        try:
            result = subprocess.run(
                cmd,
                cwd="/home/scblum/Projects/vrp_gnn/cvrp",
                check=True,
                capture_output=False,
                text=True
            )
            print(f"✓ Completed: {model_name}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {model_name}")
            print(f"  Error: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("All experiments completed!")

if __name__ == "__main__":
    run_experiment()

# Attention Heads: 8 -OR- 16