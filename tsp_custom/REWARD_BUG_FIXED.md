# Critical Bug Fixed: Rewards Were Never Computed!

## The Problem

The environment's `_step()` method was **never calling the reward computation**!

In `custom_tsp_env.py`, line 197:
```python
# OLD CODE (BROKEN):
reward = torch.zeros(batch_size, 1, dtype=torch.float32, device=adjacency.device)
# Reward stayed at 0.0 forever!
```

The `_get_reward()` method existed but was never invoked.

## The Fix

Updated `_step()` to actually compute rewards when episodes finish:

```python
# NEW CODE (FIXED):
if done.any():
    rewards_full = CustomTSPEnv._get_reward_static(td)
    reward = torch.where(
        done.unsqueeze(-1),
        rewards_full.unsqueeze(-1),
        torch.zeros(batch_size, 1, dtype=torch.float32, device=adjacency.device)
    )
else:
    reward = torch.zeros(batch_size, 1, dtype=torch.float32, device=adjacency.device)
```

Also created `_get_reward_static()` so it can be called from the static `_step()` method.

## Verification

Test confirms rewards are now computed:

```bash
$ python tsp_custom/tests/test_reward_fix.py

================================================================================
Summary:
  Non-zero rewards: 4 / 4
  Mean reward (done episodes): -10003.0420
  
✓ SUCCESS: Rewards are being computed!
================================================================================
```

## What You Need to Do

**The training process that was running needs to be restarted** because it loaded the old broken environment code.

### Stop Current Training

If training is still running, stop it with Ctrl+C.

### Start Fresh Training

Run with the fixed code:

```bash
# Quick test (recommended first)
bash tsp_custom/quick_train.sh

# Or medium training
bash tsp_custom/train_n20.sh
```

### You Should Now See

```
Epoch 0:  10%|████ | 10/79 [00:45<06:15, train/reward=-8.245, train/loss=0.324]
```

Instead of:
```
train/reward=0.000, train/loss=0.000  # This was the bug!
```

## Expected Reward Values

- **Random policy initial rewards:** -5000 to -15000 (hitting limits, invalid tours)
- **After ~10-20 epochs:** -10 to -20 (valid tours, suboptimal)
- **After 50+ epochs:** -4 to -6 (good tours for N=20)
- **Optimal TSP-20:** ~-3.84

Negative values because reward = -tour_length.

## Files Changed

1. `tsp_custom/envs/custom_tsp_env.py`:
   - Added reward computation in `_step()`
   - Created `_get_reward_static()` method
   
2. `tsp_custom/tests/test_reward_fix.py`:
   - New test to verify rewards are computed

3. `tsp_custom/models/custom_pomo_model.py`:
   - Added `on_step=True` to logging (separate fix)

## Summary

The bug was subtle but critical:
- Environment had reward computation logic ✓
- But never actually called it ✗
- Rewards stayed at 0.0 forever
- Model couldn't learn anything useful

**Fix is applied. Restart training to see results!**
