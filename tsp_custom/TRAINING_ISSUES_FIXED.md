# Training Issues Fixed

## Problems Identified

Your training was experiencing two major issues:

### 1. **40,000 Batches Per Epoch (25+ hours!)**

**Root cause:** Default dataset size is 1,280,000 instances (from POMO paper)
- With `batch_size=32`: 1,280,000 / 32 = **40,000 batches per epoch**
- At ~2 seconds per batch: ~22-25 hours per epoch
- For 100 epochs: ~2,500 hours = **104 days!**

**Why this happened:** The default settings in `train.py` are for full-scale production training, not testing.

### 2. **0.0 Rewards and Loss**

**Root cause:** Lightning logging wasn't configured to show step-level metrics during training
- Metrics were only being logged at epoch end
- Progress bar showed `0.000` because no step-level updates

## Fixes Applied

### Fix #1: Updated Logging in `custom_pomo_model.py`

Changed the `shared_step()` method to log metrics with `on_step=True` for training:

```python
# Before:
self.log(f"{phase}/reward", rewards.mean(), prog_bar=True)

# After:
on_step = (phase == 'train')
self.log(f"{phase}/reward", rewards.mean(), prog_bar=True, on_step=on_step, on_epoch=True)
```

This enables:
- Real-time updates in progress bar during training
- Epoch-level aggregation for validation/test
- Immediate feedback on learning

### Fix #2: Created Quick Training Scripts

**`quick_train.sh`** - Fast testing (2-3 min/epoch)
```bash
bash tsp_custom/quick_train.sh
```

Settings:
- Dataset: 10,000 instances → **78 batches/epoch**
- Batch size: 128
- Problem size: N=10
- Epochs: 10

**`train_n20.sh`** - Reasonable training (30-40 min/epoch)
```bash
bash tsp_custom/train_n20.sh
```

Settings:
- Dataset: 128,000 instances → **1,000 batches/epoch**
- Batch size: 128
- Problem size: N=20
- Epochs: 50

### Fix #3: Added Safety Checks

Added check for empty log_probs (edge case where episodes finish immediately):
```python
if len(all_log_probs) == 0:
    log.warning("No log probs collected!")
    return {'loss': torch.tensor(0.0, device=rewards.device)}
```

## How to Resume Training Properly

**Stop your current training** (it will take 104 days at current rate!)

Then choose one of these options:

### Option 1: Quick Test (Recommended First)
```bash
# Stop current training with Ctrl+C
bash tsp_custom/quick_train.sh
```

This will:
- Complete in ~30 minutes total (10 epochs × 3 min/epoch)
- Verify everything works
- Show you real-time metrics

### Option 2: Reasonable Training
```bash
bash tsp_custom/train_n20.sh
```

This will:
- Complete in ~24 hours (50 epochs × 30 min/epoch)
- Train a decent model on N=20
- Be much more manageable

### Option 3: Custom Settings
```bash
python tsp_custom/train.py \
    --num_loc 20 \
    --batch_size 256 \
    --train_data_size 50000 \
    --val_data_size 5000 \
    --max_epochs 20
```

Adjust parameters based on your needs (see TRAINING_GUIDE.md)

## What You'll See Now

With the fixes, your progress bar will show:

```
Epoch 0:   5%|████▌           | 100/2000 [00:45<14:15,  2.22it/s, v_num=0, train/reward=-8.245, train/loss=0.324]
```

Notice:
- ✓ **train/reward** showing actual values (e.g., -8.245)
- ✓ **train/loss** showing actual values (e.g., 0.324)
- ✓ Reasonable iteration counts (2000 not 40000)
- ✓ Reasonable time estimates (15 minutes not 25 hours)

## Key Takeaways

1. **Dataset Size Matters:**
   - Larger dataset = slower epochs but more diverse training
   - Start small for testing, scale up for production

2. **Batch Size Matters:**
   - Larger batches = fewer iterations, better GPU usage
   - Recommended: 128-256 for single GPU

3. **Always Test First:**
   - Use quick_train.sh first to verify everything works
   - Then scale up to bigger problems

4. **Monitor TensorBoard:**
   ```bash
   tensorboard --logdir tsp_custom/lightning_logs
   ```

## Files Created

- `tsp_custom/quick_train.sh` - Fast testing script
- `tsp_custom/train_n20.sh` - Reasonable training script  
- `tsp_custom/TRAINING_GUIDE.md` - Comprehensive training guide
- `tsp_custom/TRAINING_ISSUES_FIXED.md` - This document

## Expected Timeline

| Configuration | Batches/Epoch | Time/Epoch | Total Time |
|--------------|---------------|------------|------------|
| **Current (wrong!)** | 40,000 | 25 hours | 104 days |
| **quick_train.sh** | 78 | 3 min | 30 min |
| **train_n20.sh** | 1,000 | 30 min | 24 hours |
| **Full POMO** | 20,000 | 2 hours | 8 days |

## Next Steps

1. **Stop current training** (Ctrl+C in the terminal)
2. **Run quick test:** `bash tsp_custom/quick_train.sh`
3. **Verify metrics appear** in progress bar
4. **Check TensorBoard** for detailed plots
5. **If all good, run:** `bash tsp_custom/train_n20.sh`

The training should now show real-time progress and complete in reasonable time!
