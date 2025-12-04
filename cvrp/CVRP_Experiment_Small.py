import itertools
import subprocess
from pathlib import Path


def run_experiment():
    """Run training for all combinations of network parameters."""

    base_args = {
        "max_epochs": 5,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "train_data_size": 50_000,
        "log_base_dir": "lightning_logs/small_exp2",
        "normalization": "instance",
        "train_seed": "random",
        #"limit_train_batches": 100_000,
        #"train_decode_type": "sampling",
        #"train_num_starts": 0,
        #"val_decode_type": "multistart_greedy",
    }

    grid = {
        "weight_decay": [10**(-6),10**(-5)],
        "embedding_dim": [256, 512],
        "num_encoder_layers": [5, 7],
        "num_attn_heads": [8, 16],
        "vehicle_capacity": [120,150],
        "num_train_locs": [100, 130],
        "train_set_clusters": [4, 8],
    }

    base_log_dir = Path(base_args["log_base_dir"])
    base_log_dir.mkdir(parents=True, exist_ok=True)

    combos_file = base_log_dir / "experiment_combinations.txt"
    keys = list(grid.keys())
    combinations = list(itertools.product(*(grid[k] for k in keys)))

    with combos_file.open("w") as f:
        for combo in combinations:
            params = dict(zip(keys, combo))
            f.write(", ".join(f"{k}: {v}" for k, v in params.items()) + "\n")

    print(f"Running {len(combinations)} experiments...")
    print("=" * 60)

    def build_cmd(params: dict) -> list[str]:
        args = {**base_args, **params}
        cmd = ["python", "CVRP_Run.py"]
        for k, v in args.items():
            flag = f"--{k}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(v)])
        return cmd

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        model_name = f"combo_{i}"

        print(f"\n[{i}/{len(combinations)}] Training: {model_name}")
        for k, v in {**base_args, **params}.items():
            print(f"  - {k}: {v}")
        print("-" * 60)

        cmd = build_cmd({**params, "run_name": model_name})
        try:
            subprocess.run(
                cmd,
                cwd="/home/scblum/Projects/vrp_gnn/cvrp",
                check=True,
                capture_output=False,
                text=True,
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
