import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tensordict import TensorDict

sys.path.append(str(Path(__file__).resolve().parents[1]))

from classes.CVRPLibGraphPlotCallback import CVRPLibGraphPlotCallback
from classes.CVRPLibHelpers import load_val_instance, batchify_td


def make_val_loader():
    # Use one small instance from the fixtures (absolute path)
    base = Path(__file__).resolve().parents[1]  # cvrp/
    inst_path = base / "cvrplib_instances/cvrplib_x_npz/instances/X-n110-k13.npz"
    td = load_val_instance(str(inst_path))
    td = batchify_td(td)
    dataset = [td]

    def collate_fn(batch):
        return batch[0]

    return DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


def test_callback_on_loader():
    loader = make_val_loader()

    # Minimal dummy model with a policy that returns fixed actions/rewards
    class DummyPolicy(torch.nn.Module):
        def forward(self, td, phase=None, decode_type=None, **kwargs):
            B, n = td["locs"].shape[:2]
            # simple action with route separators (0): two routes then depot
            seq = [1, 2, 0, 3, 4, 0]
            actions = torch.tensor([seq for _ in range(B)])
            return {"actions": actions, "reward": torch.zeros(B)}

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = DummyPolicy()
            self.device = torch.device("cpu")
            self.val_metrics = ["reward"]
            self.train_metrics = ["reward"]
            self.test_metrics = ["reward"]

        def val_dataloader(self):
            return loader

    base = Path(__file__).resolve().parents[1]
    inst_name = "X-n110-k13"
    callback = CVRPLibGraphPlotCallback(
        env=None,
        instance_names=[inst_name],
        sol_base_dir=str(base / "cvrplib_instances/X"),
        decode_type="greedy",
        logger=None,
        use_model_solutions=False,
    )
    callback.eval_num_samples = 1  # simplify for test

    class DummyTrainer:
        def __init__(self):
            self.log_dir = str(Path(__file__).resolve().parent)
            self.current_epoch = 0
            self.global_step = 0
            self.is_global_zero = True
            self.sanity_checking = False

    trainer = DummyTrainer()
    model = DummyModel()

    # Should run without raising
    callback.on_validation_epoch_end(trainer, model)
    print("✓ Callback executed on dummy loader")


if __name__ == "__main__":
    test_callback_on_loader()
