# Callback Fixes Summary

## Date: November 22, 2025

## Issues Fixed

### 1. Device Mismatch Error ✓
**Error:** `Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

**Root Cause:** When creating TensorDict for environment reset, the locs tensor was not explicitly moved to the correct device, causing some operations to happen on GPU while others expected CPU.

**Fix Applied:**
- Added explicit device transfer in `ValidationVisualizationCallback._create_episode_gif()`
- Ensured TensorDict is created with correct device parameter
- Added `.detach().cpu()` when converting tensors to numpy for visualization

**Changed Code** (callbacks.py lines 135-146):
```python
# Create initial TensorDict - ensure on correct device
if isinstance(locs, torch.Tensor) and locs.dim() == 2:
    locs = locs.unsqueeze(0)  # Add batch dimension

# Ensure locs is on the correct device
device = pl_module.device
locs = locs.to(device)

batch = TensorDict({"locs": locs}, batch_size=[1], device=device)
td = env.reset(batch)
```

### 2. NoneType Format Error ✓
**Error:** `TypeError: unsupported format string passed to NoneType.__format__`

**Root Cause:** During the first validation epoch (sanity check), training metrics don't exist yet (are None). The f-string was trying to format None values with `.4f` format specification, which raises TypeError.

**Fix Applied:**
- Pre-format metric strings with None checks before using in f-string
- Use "N/A" as fallback for None values

**Changed Code** (callbacks.py lines 520-525):
```python
# Format metrics with fallback for None values
train_reward_str = f"{latest_train_reward:.4f}" if latest_train_reward is not None else "N/A"
val_reward_str = f"{latest_val_reward:.4f}" if latest_val_reward is not None else "N/A"
train_tour_str = f"{latest_train_tour:.4f}" if latest_train_tour is not None else "N/A"
val_tour_str = f"{latest_val_tour:.4f}" if latest_val_tour is not None else "N/A"
```

### 3. Save Directory Location ✓
**Issue:** Files were being saved to checkpoint directory instead of Lightning logs directory

**Fix Applied:**
- Changed all three callbacks to use `trainer.log_dir` instead of `trainer.checkpoint_callback.dirpath`
- This ensures GIFs and plots are saved in the same directory as TensorBoard logs

**Changed Locations:**
1. `ValidationVisualizationCallback.on_validation_epoch_end()` - line 77
2. `MetricsPlotCallback._create_plots()` - line 381
3. `CombinedMetricsPlotCallback._create_combined_plot()` - line 485

**Changed Code:**
```python
# Before:
save_dir = Path(trainer.checkpoint_callback.dirpath) if trainer.checkpoint_callback else Path(".")

# After:
save_dir = Path(trainer.log_dir) if trainer.log_dir else Path(".")
```

## Testing

### Unit Tests Created: `tsp_custom/tests/test_callbacks.py`

**Test Coverage:**
- 20 unit tests covering all three callback classes
- Tests verify:
  - ✓ Device consistency (no CUDA/CPU mismatches)
  - ✓ None value handling (no format errors)
  - ✓ File creation and naming conventions
  - ✓ Save directory correctness
  - ✓ Metric recording accuracy
  - ✓ Integration between callbacks

**Test Results:**
```
20 passed, 5 warnings in 4.79s
```

### Key Tests:
1. `test_creates_gifs_with_correct_device` - Ensures no device mismatch errors
2. `test_handles_none_values_in_summary` - Specifically tests the NoneType fix
3. `test_saves_to_lightning_logs_by_default` - Verifies correct save location
4. `test_device_consistency_across_epochs` - Tests multiple epochs
5. `test_all_callbacks_together` - Integration test

## Output Files

All files are now saved to the Lightning logs directory (e.g., `tsp_custom/lightning_logs/version_X/`):

### GIF Files:
- `epoch{epoch:03d}_val_instance{idx:02d}.gif`
- Example: `epoch000_val_instance00.gif`, `epoch005_val_instance01.gif`

### Metric Plots:
- Individual: `epoch{epoch:03d}_metrics_{metric_name}.png`
- Combined: `epoch{epoch:03d}_combined_metrics.png`
- Example: `epoch000_metrics_reward.png`, `epoch005_combined_metrics.png`

## Running Tests

```bash
# Run all callback tests
python -m pytest tsp_custom/tests/test_callbacks.py -v

# Run specific test
python -m pytest tsp_custom/tests/test_callbacks.py::TestValidationVisualizationCallback::test_creates_gifs_with_correct_device -v

# Run with coverage
python -m pytest tsp_custom/tests/test_callbacks.py --cov=tsp_custom.callbacks --cov-report=html
```

## Verified Working

- ✓ No device mismatch errors when creating GIFs
- ✓ No format errors when metrics are None
- ✓ Files saved to correct directory (lightning_logs)
- ✓ All file naming conventions followed
- ✓ Callbacks work together without conflicts
- ✓ Multi-epoch consistency maintained

## Next Steps

1. Run full training to verify fixes in production:
   ```bash
   bash tsp_custom/train_improved.sh
   ```

2. Check outputs in Lightning logs directory:
   ```bash
   ls -la tsp_custom/lightning_logs/version_*/
   ```

3. Monitor for any remaining issues during training

## Files Modified

1. `tsp_custom/callbacks/callbacks.py` - 4 changes
   - Device handling fix (line 135-146)
   - Save directory fix (lines 77, 381, 485)
   - NoneType format fix (lines 520-525)

2. `tsp_custom/tests/test_callbacks.py` - NEW FILE
   - 650+ lines of comprehensive unit tests
   - 20 test cases covering all functionality

## Technical Details

### Device Management
- All tensors explicitly moved to `pl_module.device` before operations
- TensorDict created with device parameter
- Numpy conversions use `.detach().cpu()` pattern

### None Handling Pattern
```python
# Robust pattern for formatting potentially None values
value_str = f"{value:.4f}" if value is not None else "N/A"
```

### Directory Structure
```
tsp_custom/lightning_logs/
└── version_0/
    ├── epoch000_val_instance00.gif
    ├── epoch000_val_instance01.gif
    ├── epoch000_combined_metrics.png
    ├── checkpoints/
    │   └── epoch=0-step=100.ckpt
    └── events.out.tfevents...
```

## Warnings (Non-Critical)

The tests show some matplotlib warnings about empty legends. These are cosmetic and don't affect functionality:
```
UserWarning: No artists with labels found to put in legend.
```

This occurs when plots have no data yet (first epoch). It's expected behavior and doesn't break anything.

---

**Status:** ✅ All issues resolved and tested
**Ready for:** Production training runs
