from lightning.pytorch.callbacks import Callback
from rl4co.utils.ops import batchify
import torch
from classes.CVRPLibHelpers import batchify_td

class CVRPLibValSamplerCallback(Callback):

    def __init__(self,num_samples,temperature):
        super().__init__()
        self.num_samples = num_samples
        self.temperature = temperature

    def on_validation_epoch_end(self, trainer, model):
        # Lightning returns a list even for a single val loader
        val_loader = trainer.val_dataloaders
        device = model.device
        model.eval()

        best_rewards_per_instance = []

        with torch.no_grad():
            for batch in val_loader:
                instance_name = batch["instance_name"]
                # print(f"Instance Name: {instance_name}, {type(instance_name)}")
                
                best_reward = None

                td = batch.clone().to(device)
                td = batchify_td(td) # Unsqueezes every value in td
                td_reset = model.env.reset(td=td)

                out = model(
                    td_reset.clone(),
                    decode_type="sampling",
                    num_samples=self.num_samples,
                    temperature=self.temperature,
                    return_actions=False,
                )
                best_reward = out["reward"].max().item()

                best_rewards_per_instance.append(best_reward)
                model.log(f"val-1/{instance_name}", best_reward, prog_bar=False, on_epoch=True, logger=True)

        mean_reward = sum(best_rewards_per_instance) / len(best_rewards_per_instance)
        mean_reward_small = sum(best_rewards_per_instance[:8]) / len(best_rewards_per_instance)*2
        mean_reward_large = sum(best_rewards_per_instance[8:]) / len(best_rewards_per_instance)*2
        model.log("val/10k-reward", mean_reward, prog_bar=False, on_epoch=True, logger=True)
        model.log("val/10k-reward-small",mean_reward_small, prog_bar=False, on_epoch=True, logger=True )
        model.log("val/10k-reward-large",mean_reward_large, prog_bar=False, on_epoch=True, logger=True )

        print("CVRPLib Val Sampler - Best rewards per instance:", best_rewards_per_instance)

        return best_rewards_per_instance
    