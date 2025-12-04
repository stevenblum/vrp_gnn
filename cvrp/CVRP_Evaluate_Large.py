"""
Evaluate trained CVRP model with large number of samples using batched processing.
This script avoids OOM by sampling in batches rather than all at once.
"""
import torch
from pathlib import Path
from rl4co.models import POMO, AttentionModelPolicy
from rl4co.envs.routing import CVRPEnv, CVRPGenerator
from classes.CVRPLibHelpers import load_val_instance, load_bks_cost, calculate_normalized_bks
import argparse

def evaluate_with_batched_sampling(model, td, env, total_samples=100_000, batch_size=1_000):
    """
    Evaluate instance by sampling in batches to avoid OOM.
    
    Args:
        model: Trained POMO model
        td: TensorDict with single instance
        env: CVRP environment
        total_samples: Total number of solutions to sample
        batch_size: Number of samples per batch
        
    Returns:
        best_cost: Best cost found
        best_actions: Actions for best solution
    """
    device = next(model.parameters()).device
    td = td.to(device)
    
    best_cost = float('inf')
    best_actions = None
    
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    print(f"    Sampling {total_samples} solutions in {num_batches} batches of {batch_size}...")
    
    model.eval()
    with torch.no_grad():
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, total_samples - batch_idx * batch_size)
            
            # Run POMO with this batch size
            out = model.policy(
                td.clone(),
                env,
                phase="test",
                decode_type="multistart_sampling",
                num_starts=current_batch_size,
                return_actions=True,
            )
            
            # Get rewards (negative cost)
            rewards = out["reward"]
            costs = -rewards
            
            # Find best in this batch
            batch_min_cost, batch_min_idx = costs.min(dim=-1)
            
            if batch_min_cost.item() < best_cost:
                best_cost = batch_min_cost.item()
                # Extract best actions
                if "actions" in out:
                    best_actions = out["actions"][0, batch_min_idx]
            
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
                print(f"      Batch {batch_idx + 1}/{num_batches} - Best so far: {best_cost:.4f}")
    
    return best_cost, best_actions


def main():
    parser = argparse.ArgumentParser(description='Evaluate CVRP model with large sampling')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--total_samples', type=int, default=100_000,
                       help='Total number of samples to take (default: 100,000)')
    parser.add_argument('--batch_size', type=int, default=1_000,
                       help='Batch size for sampling (default: 1,000)')
    parser.add_argument('--instances', type=str, nargs='+', 
                       default=["X-n110-k13", "X-n115-k10", "X-n120-k6"],
                       help='Instance names to evaluate')
    args = parser.parse_args()
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    hparams = checkpoint.get('hyper_parameters', {})
    
    # Extract hyperparameters
    embed_dim = hparams.get('embed_dim', 256)
    num_encoder_layers = hparams.get('num_encoder_layers', 6)
    num_heads = hparams.get('num_heads', 8)
    
    print(f"Model config: embed_dim={embed_dim}, layers={num_encoder_layers}, heads={num_heads}")
    
    # Create environment
    temp_env = CVRPEnv(CVRPGenerator(num_loc=100))
    
    # Create policy
    policy = AttentionModelPolicy(
        env_name='cvrp',
        embed_dim=embed_dim,
        num_encoder_layers=num_encoder_layers,
        num_heads=num_heads,
    )
    
    # Create model
    model = POMO(temp_env, policy)
    
    # Load weights
    state_dict = {k: v for k, v in checkpoint['state_dict'].items() if not k.startswith('env.')}
    model.load_state_dict(state_dict, strict=False)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"\nEvaluating with {args.total_samples} total samples per instance")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)
    
    # Evaluate each instance
    results = []
    for instance_name in args.instances:
        print(f"\n{instance_name}:")
        
        # Load instance
        instance_path = Path(f"cvrplib_instances/cvrplib_x_npz/instances/{instance_name}.npz")
        if not instance_path.exists():
            print(f"  ✗ Instance not found: {instance_path}")
            continue
        
        td = load_val_instance(str(instance_path))
        
        # Get BKS
        bks_normalized = calculate_normalized_bks(instance_name)
        bks_raw = load_bks_cost(instance_name)
        
        # Create environment for this instance size
        num_loc = td["locs"].shape[1]
        temp_gen = CVRPGenerator(num_loc=num_loc)
        temp_env = CVRPEnv(temp_gen)
        
        # Debugging: Print tensor dimensions
        print(f"  Debug: td['locs'] shape: {td['locs'].shape}")
        print(f"  Debug: td['demand'] shape: {td['demand'].shape}")
        print(f"  Debug: td['visited'] shape: {td['visited'].shape}")
        print(f"  Debug: num_loc: {num_loc}")

        # Debugging: Validate environment setup
        print(f"  Debug: Environment generator num_loc: {temp_gen.num_loc}")
        
        # Evaluate
        best_cost, best_actions = evaluate_with_batched_sampling(
            model, td, temp_env, 
            total_samples=args.total_samples,
            batch_size=args.batch_size
        )
        
        # Calculate gap
        if bks_normalized:
            gap = 100.0 * (best_cost - bks_normalized) / bks_normalized
            print(f"  Model cost: {best_cost:.4f}")
            print(f"  BKS (norm): {bks_normalized:.4f}")
            print(f"  Gap: {gap:.2f}%")
        else:
            gap = None
            print(f"  Model cost: {best_cost:.4f}")
            print(f"  BKS not available")
        
        results.append({
            'instance': instance_name,
            'model_cost': best_cost,
            'bks_normalized': bks_normalized,
            'bks_raw': bks_raw,
            'gap': gap,
            'num_samples': args.total_samples
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("-" * 80)
    print(f"{'Instance':<20} {'Model Cost':>12} {'BKS (norm)':>12} {'Gap %':>8}")
    print("-" * 80)
    for r in results:
        gap_str = f"{r['gap']:.2f}" if r['gap'] is not None else "N/A"
        bks_str = f"{r['bks_normalized']:.4f}" if r['bks_normalized'] else "N/A"
        print(f"{r['instance']:<20} {r['model_cost']:>12.4f} {bks_str:>12} {gap_str:>8}")
    
    if any(r['gap'] is not None for r in results):
        avg_gap = sum(r['gap'] for r in results if r['gap'] is not None) / sum(1 for r in results if r['gap'] is not None)
        print("-" * 80)
        print(f"{'Average Gap':<20} {avg_gap:>45.2f}")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
