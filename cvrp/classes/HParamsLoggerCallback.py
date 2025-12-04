import math
import torch
from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class HParamsLoggerCallback(Callback):
    """Log train/val rewards into the logger's HParams metrics each epoch.

    - Logs `train_reward` (from `train/reward`) and `val_reward` (from `val/reward`).
    - Tracks the best `val_reward` and the `best_val_step` (trainer.global_step when best seen).
    - Uses `logger.log_hyperparams({}, metrics=...)` when available, otherwise falls back to `logger.log_metrics(...)`.
    """

    def __init__(self):
        super().__init__()
        self.best_val_reward: Optional[float] = None
        self.best_val_step: Optional[int] = None

    def _extract_metric(self, trainer, key: str) -> Optional[float]:
        v = trainer.callback_metrics.get(key)
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            try:
                return float(v.detach().cpu().item())
            except Exception:
                return None
        try:
            return float(v)
        except Exception:
            return None

    def _log_metrics(self, trainer, metrics: dict):
        logger = trainer.logger
        step = getattr(trainer, "global_step", None)
        # Prefer hparams metrics if available
        if logger is not None and hasattr(logger, "log_hyperparams"):
            try:
                logger.log_hyperparams({}, metrics=metrics)
                return
            except Exception:
                # Fall through to log_metrics
                pass

        # Fallback
        if logger is not None and hasattr(logger, "log_metrics"):
            try:
                logger.log_metrics(metrics, step=step)
            except Exception:
                # Last resort: print
                print("HParamsLoggerCallback: failed to log metrics to logger; metrics=", metrics)
        else:
            print("HParamsLoggerCallback: no logger available; metrics=", metrics)

    def _maybe_update_best(self, val_reward: Optional[float], trainer) -> None:
        if val_reward is None:
            return
        if self.best_val_reward is None or val_reward > self.best_val_reward:
            self.best_val_reward = val_reward
            self.best_val_step = getattr(trainer, "global_step", None)

    def _gather_and_log(self, trainer, pl_module) -> None:
        train_r = self._extract_metric(trainer, "train/reward")
        val_r = self._extract_metric(trainer, "val/reward")

        # Update best
        self._maybe_update_best(val_r, trainer)

        metrics = {}
        if train_r is not None:
            metrics["train_reward"] = train_r
        if val_r is not None:
            metrics["val_reward"] = val_r
        metrics["best_val_reward"] = self.best_val_reward if self.best_val_reward is not None else -math.inf
        metrics["best_val_step"] = self.best_val_step if self.best_val_step is not None else -1

        # Log
        self._log_metrics(trainer, metrics)

    # Run at the end of training epoch and validation epoch to ensure metrics are captured
    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._gather_and_log(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._gather_and_log(trainer, pl_module)
