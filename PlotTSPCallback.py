# PlotTSPCallback.py
import os
import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback

class PlotTSPCallback(Callback):
    def __init__(self, env, td_plot, subdir="val_plots", decode_type="greedy"):
        super().__init__()
        self.env = env
        self.td_plot = td_plot
        self.subdir = subdir
        self.decode_type = decode_type
        self.out_dir = None  # will be set once we know trainer.log_dir

    def _ensure_out_dir(self, trainer):
        if self.out_dir is None:
            # e.g. lightning_logs/version_0/val_plots
            self.out_dir = os.path.join(trainer.log_dir, self.subdir)
            os.makedirs(self.out_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # only one process writes files (important if you ever go multi-GPU)
        if not trainer.is_global_zero:
            return

        self._ensure_out_dir(trainer)

        pl_module.eval()
        with torch.no_grad():
            td = self.td_plot.to(pl_module.device)
            out = pl_module.policy(
                td.clone(),
                phase="test",
                decode_type=self.decode_type,
                return_actions=True,
            )
            actions = out["actions"].cpu()
            rewards = out["reward"].cpu()

        for i, td_i in enumerate(td.cpu()):
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            self.env.render(td_i, actions[i], ax=ax)
            ax.set_title(
                f"Epoch {trainer.current_epoch} | ex {i} | cost = {-rewards[i].item():.3f}"
            )
            fname = os.path.join(
                self.out_dir,
                f"epoch{trainer.current_epoch:03d}_example{i}.png",
            )
            fig.savefig(fname, bbox_inches="tight")
            plt.close(fig)
