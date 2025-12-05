from classes.FacExp import FacExp
import subprocess
import sys
import os
import time
import torch
from pathlib import Path
from datetime import datetime
import wandb
import argparse

'''

exp_name, anything that allows me to remember where/when I ran it
exp_name will also be the wandb project name and the local log directory

run_name = [combo_name]_[run_id]
combo name will just be the number order in the experiment
run_id is datetime stamp

'''

EXP_NAME = "server1-7-3"

BASE_ARGS = {
        "exp_name": EXP_NAME,
        "max_epochs": 100,
        "learning_rate": 1e-4,
        "batch_size": 64,
        "train_data_size": 100_000,
        "normalization": "instance",
        "train_seed": "random",
        "optimizer": "Adam",
        #"limit_train_batches": 100_000,
        #"train_decode_type": "sampling",
        #"train_num_starts": 0,
        #"val_decode_type": "multistart_greedy",
    }

EXP_GRID = {
    "embedding_dim": [256, 512], #A
    "num_encoder_layers": [5, 7], #B
    "num_attn_heads": [8, 16], #C
    "train_vehicle_capacity": [120,150], #D
    "train_num_locs": [100, 130], #E
    "train_set_clusters": [4, 8], #F
    "weight_decay": [10**(-6),10**(-5)], #G
}

EXP_DEFINING_CONTRASTS = ['ABCE',"BCDF","ACDG"]

# Specifically arranged so when the principal fraction is selected
# ABCE + will include testing the "largest" model

def run_experiment():
    """Run training for all combinations of network parameters."""

    # You can add a defining_contrasts argument if you want a fractional factorial
    facexp = FacExp(EXP_GRID, defining_contrasts=EXP_DEFINING_CONTRASTS) #2^(7-3), 16 Combinations, Resolution 4
    combinations_df = facexp.get_exp_combinations()
    combinations = combinations_df.to_dict(orient="records")

    
    base_log_dir = Path("Logs",EXP_NAME)
    base_log_dir.mkdir(parents=True, exist_ok=True)
    combos_file = base_log_dir / "experiment_combinations.txt"
    with combos_file.open("w") as f:
        for params in combinations:
            f.write(", ".join(f"{k}: {v}" for k, v in params.items()) + "\n")

    ok = wandb.login(key="47163152a32c387490415222e0197fc4f8ec6898")
    if not ok:
        print("Failed to log in to Weights & Biases. Check your API key.")
        return

    print(f"Running {len(combinations)} experiments...")
    print("=" * 60)

    def build_cmd(params: dict) -> list[str]:
        args = {**BASE_ARGS, **params}
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
        return cmd

    # determine the directory this script lives in so subprocess can run
    # `CVRP_Run.py` relative to this file instead of using a hard-coded path
    script_dir = Path(__file__).resolve().parent

    # Device pool: allow running multiple jobs concurrently across devices.
    # Devices can be provided via EXP_DEVICES env var (comma-separated), e.g.
    # EXP_DEVICES="cuda:0,cuda:1,cpu". If not provided we auto-detect available devices.
    exp_devices = os.environ.get("EXP_DEVICES")
    if exp_devices:
        device_names = [d.strip() for d in exp_devices.split(",") if d.strip()]
    else:
        # auto-detect
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            device_names = [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device_names = ["mps"]
        else:
            device_names = ["cpu"]

    print(f"Using device pool: {device_names}")

    # track running processes and device availability
    device_available = {d: True for d in device_names}
    running = []  # list of dicts: {'proc': Popen, 'device_name': d, 'name': model_name}

    try:
        for i, params in enumerate(combinations, 1):
            # wait for a free device
            chosen_name = None
            device_name = None
            device_num = None
            while chosen_name is None:
                # free up finished processes
                for entry in running[:]:
                    ret = entry["proc"].poll()
                    if ret is not None:
                        print(f"Process for {entry['name']} finished (exit {ret}) on {entry['device_name']}")
                        device_available[entry["device_name"]] = True
                        running.remove(entry)

                for dname, free in device_available.items():
                    if free:
                        chosen_name = dname
                        device_available[dname] = False
                        # Assign device_name and device_num immediately
                        if isinstance(chosen_name, str) and chosen_name.startswith('cuda:'):
                            device_name = chosen_name  # e.g., 'cuda:2'
                            device_num = int(chosen_name.split(':')[1])
                        elif chosen_name == 'cpu':
                            device_name = 'cpu'
                            device_num = 0
                        elif chosen_name == 'mps':
                            device_name = 'mps'
                            device_num = 0
                        else:
                            device_name = chosen_name
                            device_num = 0
                        break

                if chosen_name is None:
                    time.sleep(10)

            time.sleep(2)

            combo_name = f"combo-{i:03d}"
            now = datetime.now()
            run_id = now.strftime("%y%m%d-%H%M") +"-"+ f"{now.microsecond//10000:02d}"
            run_name = f"{combo_name}_{run_id}"
            log_dir = Path(base_log_dir,run_name)

            print(f"\n[{i}/{len(combinations)}] Training: {run_name}")
            for k, v in {**BASE_ARGS, **params}.items():
                print(f"  - {k}: {v}")
            print("-" * 60)

            # device_name and device_num already assigned above
            params_with_device = {**params, "device_name": device_name, "device_num": device_num, "run_name":run_name, "combo_name": combo_name,"run_id":run_id, "log_dir":log_dir }
            cmd = build_cmd(params_with_device)

            print(f"Launching: {cmd} on device {device_name}")
            env = os.environ.copy()
            # Do NOT set ACCELERATOR here; CVRP_Run.py will respect device_name/device_num
            p = subprocess.Popen(cmd, cwd=str(script_dir), env=env)
            running.append({"proc": p, "device_name": device_name, "name": combo_name})

        # wait for remaining processes
        while running:
            for entry in running[:]:
                ret = entry["proc"].poll()
                if ret is not None:
                    print(f"Process for {entry['name']} finished (exit {ret}) on {entry['device_name']}")
                    device_available[entry["device_name"]] = True
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
