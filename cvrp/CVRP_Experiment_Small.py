from classes.FacExp import FacExp
import subprocess
import sys
import os
import time
import torch
from pathlib import Path

def run_experiment():
    """Run training for all combinations of network parameters."""

    base_args = {
        "max_epochs": 5,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "train_data_size": 1_000,
        "log_base_dir": "lightning_logs/small_exp2",
        "normalization": "instance",
        "train_seed": "random",
        #"limit_train_batches": 100_000,
        #"train_decode_type": "sampling",
        #"train_num_starts": 0,
        #"val_decode_type": "multistart_greedy",
    }

    grid = {
        "embedding_dim": [256, 512], #A
        "num_encoder_layers": [5, 7], #B
        "num_attn_heads": [8, 16], #C
        "vehicle_capacity": [120,150], #D
        "num_train_locs": [100, 130], #E
        "train_set_clusters": [4, 8], #F
        "weight_decay": [10**(-6),10**(-5)], #G
    }

    # Specifically arranged so when the principal fraction is selected
    # ABCE + will include testing the "largest" model

    # You can add a defining_contrasts argument if you want a fractional factorial
    facexp = FacExp(grid, defining_contrasts=['ABCE',"BCDF","ACDG"]) #2^(7-3), 16 Combinations, Resolution 4
    combinations_df = facexp.get_exp_combinations()
    combinations = combinations_df.to_dict(orient="records")
    keys = list(grid.keys())

    base_log_dir = Path(base_args["log_base_dir"])
    base_log_dir.mkdir(parents=True, exist_ok=True)

    combos_file = base_log_dir / "experiment_combinations.txt"
    with combos_file.open("w") as f:
        for params in combinations:
            f.write(", ".join(f"{k}: {v}" for k, v in params.items()) + "\n")

    print(f"Running {len(combinations)} experiments...")
    print("=" * 60)

    def build_cmd(params: dict) -> list[str]:
        args = {**base_args, **params}
        # use the same Python interpreter that's running this script
        cmd = [sys.executable, "CVRP_Run.py"]
        for k, v in args.items():
            flag = f"--{k}"
            # skip unset values
            if v is None:
                continue
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(v)])
    # do not pass accelerator as an argparse flag (CVRP_Run may not accept it)
    # the accelerator will be provided via environment variable when launching
    # the subprocess.
        return cmd

    # determine the directory this script lives in so subprocess can run
    # `CVRP_Run.py` relative to this file instead of using a hard-coded path
    script_dir = Path(__file__).resolve().parent

    # Device pool: allow running multiple jobs concurrently across devices.
    # Devices can be provided via EXP_DEVICES env var (comma-separated), e.g.
    # EXP_DEVICES="cuda:0,cuda:1,cpu". If not provided we auto-detect available devices.
    exp_devices = os.environ.get("EXP_DEVICES")
    if exp_devices:
        devices = [d.strip() for d in exp_devices.split(",") if d.strip()]
    else:
        # auto-detect
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            devices = [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices = ["mps"]
        else:
            devices = ["cpu"]

    print(f"Using device pool: {devices}")

    # track running processes and device availability
    device_free = {d: True for d in devices}
    running = []  # list of dicts: {'proc': Popen, 'device': d, 'name': model_name}

    try:
        for i, params in enumerate(combinations, 1):
            model_name = f"combo_{i}"

            print(f"\n[{i}/{len(combinations)}] Training: {model_name}")
            for k, v in {**base_args, **params}.items():
                print(f"  - {k}: {v}")
            print("-" * 60)

            # wait for a free device
            chosen = None
            while chosen is None:
                # free up finished processes
                for entry in running[:]:
                    ret = entry["proc"].poll()
                    if ret is not None:
                        print(f"Process for {entry['name']} finished (exit {ret}) on {entry['device']}")
                        device_free[entry["device"]] = True
                        running.remove(entry)

                for d, free in device_free.items():
                    if free:
                        chosen = d
                        device_free[d] = False
                        break

                if chosen is None:
                    time.sleep(1)

            # assign device and launch subprocess
            params_with_device = {**params, "run_name": model_name, "device": chosen}
            cmd = build_cmd(params_with_device)

            print(f"Launching: {cmd} on device {chosen}")
            env = os.environ.copy()
            # Do NOT set ACCELERATOR here; CVRP_Run.py will respect --device
            p = subprocess.Popen(cmd, cwd=str(script_dir), env=env)
            running.append({"proc": p, "device": chosen, "name": model_name})

        # wait for remaining processes
        while running:
            for entry in running[:]:
                ret = entry["proc"].poll()
                if ret is not None:
                    print(f"Process for {entry['name']} finished (exit {ret}) on {entry['device']}")
                    device_free[entry["device"]] = True
                    running.remove(entry)
            time.sleep(1)

    except KeyboardInterrupt:
        print("Interrupted: terminating running jobs...")
        for entry in running:
            try:
                entry["proc"].terminate()
            except Exception:
                pass
        raise

    print("\n" + "=" * 60)
    print("All experiments completed!")


if __name__ == "__main__":
    run_experiment()
