# Training Guide for Custom TSP Model

## Problem: Training Too Slow

If you're seeing training estimates of 20+ hours per epoch, the issue is:

**Default dataset is TOO LARGE for testing**
- Default: 1,280,000 training instances (from POMO paper)
- With batch_size=32: **40,000 batches per epoch** 
- At 2 seconds per batch: **22+ hours per epoch**

## Solutions

### 1. Quick Test (N=10, ~2-3 min/epoch)
```bash
bash tsp_custom/quick_train.sh
```

Or manually:
```bash
python tsp_custom/train.py \
    --num_loc 10 \
    --batch_size 128 \
    --train_data_size 10000 \
    --val_data_size 1000 \
    --max_epochs 10
```

**Expected:** ~78 batches/epoch, ~2-3 minutes per epoch

### 2. Medium Test (N=20, ~30-40 min/epoch)
```bash
bash tsp_custom/train_n20.sh
```

Or manually:
```bash
python tsp_custom/train.py \
    --num_loc 20 \
    --batch_size 128 \
    --train_data_size 128000 \
    --val_data_size 10000 \
    --max_epochs 50
```

**Expected:** ~1,000 batches/epoch, ~30-40 minutes per epoch

### 3. Full Training (N=20, POMO-style, 2-4 hours/epoch)
Only use this after validating the model works on smaller datasets!

```bash
python tsp_custom/train.py \
    --num_loc 20 \
    --batch_size 64 \
    --train_data_size 1280000 \
    --val_data_size 10000 \
    --max_epochs 100
```

**Expected:** ~20,000 batches/epoch, ~2-4 hours per epoch on GPU

## Key Parameters to Adjust

### Dataset Size (`--train_data_size`)
- Controls how many training instances per epoch
- **Smaller = faster epochs, less diverse data**
- **Larger = slower epochs, more diverse data**

Recommended:
- Testing: 10,000 - 50,000
- Development: 100,000 - 200,000
- Production: 1,000,000+

### Batch Size (`--batch_size`)
- Controls how many instances processed together
- **Larger = fewer iterations, better GPU utilization**
- **Smaller = more iterations, potentially better gradients**

Recommended:
- CPU: 32-64
- GPU: 128-512
- Multi-GPU: 512-1024

### Problem Size (`--num_loc`)
- Number of cities in TSP
- **Larger = more complex, slower per batch**

Typical:
- Small: 10-20
- Medium: 50-100
- Large: 100-500

## Monitoring Training

### Metrics to Watch

1. **train/reward**: Higher is better (less negative)
   - Random policy: ~-5.0 to -10.0 for N=20
   - Good policy: ~-3.8 to -4.0 for N=20
   
2. **train/tour_length**: Lower is better
   - Should decrease over epochs
   - Optimal TSP-20: ~3.8
   
3. **train/loss**: REINFORCE loss
   - Can be positive or negative
   - Should stabilize after warmup

4. **delete_bias**: Curriculum schedule
   - Starts at -5.0
   - Ends at 0.0
   - Linear interpolation

### TensorBoard

```bash
# In a separate terminal
tensorboard --logdir tsp_custom/lightning_logs
```

Open browser: http://localhost:6006

### Progress Bar

The progress bar shows:
- Current iteration / total iterations
- Time per iteration
- Estimated time remaining
- Real-time metrics: reward, loss

## Troubleshooting

### Issue: 0.0 rewards and loss

**Cause:** Logging wasn't set up properly (now fixed)

**Solution:** The fix I just applied adds `on_step=True` to logging

### Issue: Training too slow

**Cause:** Dataset too large or batch size too small

**Solution:** Use the quick training scripts above

### Issue: Out of memory (OOM)

**Cause:** Batch size too large for GPU

**Solution:** Reduce `--batch_size` or `--num_loc`

```bash
# Smaller batch
python tsp_custom/train.py --batch_size 32

# Or smaller problem
python tsp_custom/train.py --num_loc 10
```

### Issue: Reward not improving

**Possible causes:**
1. Learning rate too high/low
2. Delete bias schedule too aggressive
3. Model capacity too small
4. Need more training time

**Solutions:**
```bash
# Adjust learning rate
python tsp_custom/train.py --lr 5e-5  # Lower
python tsp_custom/train.py --lr 2e-4  # Higher

# Slower delete bias warmup
python tsp_custom/train.py --delete_bias_warmup_epochs 200

# Larger model
python tsp_custom/train.py --embed_dim 256 --num_encoder_layers 12
```

## Expected Performance

### TSP-20 (20 cities)
- Random policy: ~10.0 tour length
- Greedy heuristic: ~4.5 tour length  
- Optimal: ~3.84 tour length
- **Target:** ~4.0-4.2 tour length after 50-100 epochs

### TSP-50 (50 cities)
- Random policy: ~18.0 tour length
- Greedy heuristic: ~6.5 tour length
- Optimal: ~5.7 tour length
- **Target:** ~6.0-6.3 tour length after 100+ epochs

### TSP-100 (100 cities)
- Random policy: ~30.0 tour length
- Greedy heuristic: ~8.5 tour length
- Optimal: ~7.76 tour length
- **Target:** ~8.0-8.5 tour length after 200+ epochs

## Recommended Training Workflow

1. **Quick sanity check** (5-10 minutes)
   ```bash
   bash tsp_custom/quick_train.sh
   ```
   - Verify training runs without errors
   - Check that reward improves slightly

2. **Small-scale validation** (1-2 hours)
   ```bash
   python tsp_custom/train.py --num_loc 20 --train_data_size 50000 --max_epochs 20
   ```
   - Verify model can learn on small dataset
   - Check convergence behavior

3. **Medium-scale training** (12-24 hours)
   ```bash
   bash tsp_custom/train_n20.sh
   ```
   - Train on reasonable dataset size
   - Achieve good performance

4. **Full-scale training** (2-4 days)
   ```bash
   python tsp_custom/train.py --num_loc 20 --max_epochs 100
   ```
   - Use full POMO dataset size
   - Achieve best performance

## GPU Recommendations

### Single GPU
```bash
python tsp_custom/train.py --accelerator gpu --devices 1 --batch_size 128
```

### Multi-GPU (Data Parallel)
```bash
python tsp_custom/train.py --accelerator gpu --devices 4 --batch_size 512
```

The batch size should scale roughly with number of GPUs for efficiency.

## Checkpoints

Checkpoints are saved to `tsp_custom/checkpoints/` by default.

Best checkpoints based on validation reward are kept automatically.

To resume training:
```bash
python tsp_custom/train.py --resume_from_checkpoint tsp_custom/checkpoints/last.ckpt
```

## Next Steps After Training

After successfully training a model:

1. Evaluate on validation set
2. Visualize learned tours (Step 9)
3. Compare to baselines (greedy, optimal)
4. Try larger problem sizes
5. Experiment with model architecture
