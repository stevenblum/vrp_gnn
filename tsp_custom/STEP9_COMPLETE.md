# Step 9: Visualization and Metrics Callbacks - Implementation Complete

## Overview

Step 9 has been successfully implemented with three Lightning callbacks for creating visualizations and metrics plots during training.

## Callbacks Implemented

### 1. ValidationVisualizationCallback

Creates animated GIFs showing how the model constructs tours during validation.

**Features:**
- Runs validation instances through the trained policy
- Records full action sequence (ADD/DELETE/DONE actions)
- Creates animated GIF showing tour construction frame-by-frame
- Green edges: newly added edges
- Red dotted edges: deleted edges
- Saves with epoch prefix: `epoch003_val_instance00.gif`

**Parameters:**
- `num_instances`: Number of validation instances to visualize (default: 3)
- `fps`: Frames per second in the GIF (default: 2)
- `every_n_epochs`: Frequency of visualization (default: 1)

**Example Usage:**
```bash
python tsp_custom/train.py \
    --enable_viz \
    --viz_every_n_epochs 5 \
    --num_viz_instances 3 \
    --viz_fps 2
```

### 2. MetricsPlotCallback

Creates individual line plots for each metric tracked during training.

**Features:**
- Tracks train/val rewards and tour lengths
- Creates separate PNG for each metric
- Shows training progression over epochs
- Saves with epoch prefix: `epoch010_metrics_reward.png`

**Parameters:**
- `plot_every_n_epochs`: Frequency of plot generation (default: 1)

**Metrics Tracked:**
- `train/reward`: Training episode rewards
- `val/reward`: Validation episode rewards
- `train/tour_length`: Training tour lengths
- `val/tour_length`: Validation tour lengths

**Example Usage:**
```bash
python tsp_custom/train.py \
    --enable_metrics_plot \
    --metrics_plot_every_n_epochs 1
```

### 3. CombinedMetricsPlotCallback

Creates a 2x2 dashboard view of all key metrics in one image.

**Features:**
- Four subplots: rewards, tour lengths, loss, summary text
- Comprehensive view of training progress
- Single dashboard instead of multiple files
- Saves with epoch prefix: `epoch010_combined_metrics.png`

**Parameters:**
- `plot_every_n_epochs`: Frequency of plot generation (default: 5)

**Dashboard Layout:**
```
┌─────────────────┬─────────────────┐
│ Train/Val       │ Train/Val       │
│ Rewards         │ Tour Lengths    │
├─────────────────┼─────────────────┤
│ Training        │ Summary         │
│ Loss            │ Statistics      │
└─────────────────┴─────────────────┘
```

**Example Usage:**
```bash
python tsp_custom/train.py \
    --enable_metrics_plot \
    --use_combined_metrics \
    --metrics_plot_every_n_epochs 5
```

## File Output Structure

All outputs are saved to the Lightning checkpoint directory with epoch-prefixed filenames:

```
tsp_custom/checkpoints/CustomTSP_N10_E100_B64/
├── epoch000_val_instance00.gif
├── epoch000_val_instance01.gif
├── epoch000_val_instance02.gif
├── epoch000_combined_metrics.png
├── epoch005_val_instance00.gif
├── epoch005_val_instance01.gif
├── epoch005_val_instance02.gif
├── epoch005_combined_metrics.png
├── epoch010_val_instance00.gif
├── ...
└── epoch100_combined_metrics.png
```

## Integration with Training Script

The callbacks are fully integrated into `train.py` with command-line arguments:

```bash
python tsp_custom/train.py \
    --num_loc 20 \
    --max_epochs 100 \
    --batch_size 64 \
    --enable_viz \
    --viz_every_n_epochs 5 \
    --num_viz_instances 3 \
    --viz_fps 2 \
    --enable_metrics_plot \
    --use_combined_metrics \
    --metrics_plot_every_n_epochs 5
```

## Testing the Implementation

A test script is provided for quick validation:

```bash
bash tsp_custom/test_step9.sh
```

This runs a quick 3-epoch training with:
- 1K training instances (N=10)
- 100 validation instances
- Visualization every 2 epochs
- Combined metrics every epoch

Expected outputs in checkpoint directory:
- 2 GIFs per instance at epoch 0 and 2
- Combined metrics PNG at epochs 0, 1, 2

## Implementation Details

### Callback Locations
- Source: `tsp_custom/callbacks/callbacks.py`
- Exports: `tsp_custom/callbacks/__init__.py`
- Integration: `tsp_custom/train.py`

### Dependencies
- `matplotlib`: For plotting
- `PIL (Pillow)`: For GIF creation
- `torch`, `tensordict`: For tensor operations
- `lightning`: For callback framework

### Key Methods

**ValidationVisualizationCallback:**
- `on_validation_epoch_end()`: Main hook, triggered after validation
- `_create_episode_gif()`: Records action sequence and builds GIF
- `_make_gif()`: Creates animated GIF from frames
- `_create_frame()`: Renders single frame with current tour state

**MetricsPlotCallback:**
- `on_train_epoch_end()`: Records training metrics
- `on_validation_epoch_end()`: Records validation metrics, creates plots
- `_create_plots()`: Generates individual metric plots

**CombinedMetricsPlotCallback:**
- `on_train_epoch_end()`: Records training metrics
- `on_validation_epoch_end()`: Records validation metrics, creates dashboard
- `_create_combined_plot()`: Generates 2x2 subplot dashboard

## Compliance with Step 9 Requirements

✅ **Visualization during validation**: ValidationVisualizationCallback creates GIFs
✅ **Plot validation and training costs**: Callbacks track and plot both train/val metrics  
✅ **Save to Lightning folder**: All outputs go to trainer's checkpoint directory
✅ **Epoch## prefix**: All filenames start with `epoch{epoch:03d}_`
✅ **Same epoch files together**: Alphabetical sorting groups files by epoch

## Next Steps

Step 9 is complete! The callbacks are:
- Fully implemented with all required features
- Integrated into the training script
- Tested with proper import structure
- Ready for use in training runs

To proceed with training:
1. Use `test_step9.sh` to verify callbacks work correctly
2. Run full training with `train_n20.sh` plus callback flags
3. Monitor checkpoint directory for visualization/metrics outputs

## Status

**Step 9: COMPLETE** ✅

Date: November 22, 2025
