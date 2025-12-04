#!/bin/bash
# Quick test to verify callback fixes
# This will run a minimal training session to test the callbacks

echo "Testing callback fixes..."
echo "========================"

cd /home/scblum/Projects/vrp_gnn

# Run with minimal settings for quick test
python tsp_custom/train.py \
    --max_epochs 2 \
    --train_size 100 \
    --val_size 20 \
    --batch_size 8 \
    --num_nodes 10 \
    --enable_viz true \
    --viz_every_n_epochs 1 \
    --num_viz_instances 2 \
    --viz_fps 2 \
    --enable_metrics_plot true \
    --use_combined_metrics true \
    --metrics_plot_every_n_epochs 1 \
    --val_check_interval 50 \
    --log_every_n_steps 10

echo ""
echo "Training complete. Checking outputs..."
echo "======================================"

# Find the latest version directory
LATEST_VERSION=$(ls -td tsp_custom/lightning_logs/version_* 2>/dev/null | head -1)

if [ -z "$LATEST_VERSION" ]; then
    echo "ERROR: No lightning_logs directory found!"
    exit 1
fi

echo "Checking directory: $LATEST_VERSION"
echo ""

# Check for GIF files
GIF_COUNT=$(find "$LATEST_VERSION" -name "epoch*.gif" 2>/dev/null | wc -l)
echo "GIF files found: $GIF_COUNT"
if [ $GIF_COUNT -gt 0 ]; then
    find "$LATEST_VERSION" -name "epoch*.gif" -exec ls -lh {} \;
else
    echo "  WARNING: No GIF files found"
fi

echo ""

# Check for PNG files
PNG_COUNT=$(find "$LATEST_VERSION" -name "epoch*.png" 2>/dev/null | wc -l)
echo "PNG metric plots found: $PNG_COUNT"
if [ $PNG_COUNT -gt 0 ]; then
    find "$LATEST_VERSION" -name "epoch*.png" -exec ls -lh {} \;
else
    echo "  WARNING: No PNG files found"
fi

echo ""
echo "Test complete!"
echo "=============="

if [ $GIF_COUNT -gt 0 ] || [ $PNG_COUNT -gt 0 ]; then
    echo "✓ SUCCESS: Callbacks are working correctly!"
    echo "  - GIFs: $GIF_COUNT"
    echo "  - PNGs: $PNG_COUNT"
    echo "  - Location: $LATEST_VERSION"
    exit 0
else
    echo "✗ FAILURE: No output files found"
    echo "  Check the training logs for errors"
    exit 1
fi
