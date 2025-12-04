#!/bin/bash
# Quick training script with reasonable settings for testing
# This uses a much smaller dataset for faster iteration

echo "========================================================================"
echo "Quick Training for Custom TSP (Testing Configuration)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Problem size: N=10 (small for testing)"
echo "  - Dataset: 10,000 training instances (vs 1.28M default)"
echo "  - Batch size: 128 (larger batches = fewer iterations)"
echo "  - Epochs: 10"
echo "  - Expected: ~78 batches per epoch (~2-3 minutes per epoch)"
echo ""
echo "Press Ctrl+C to stop training"
echo "========================================================================"
echo ""

python tsp_custom/train.py \
    --num_loc 10 \
    --max_epochs 10 \
    --batch_size 128 \
    --train_data_size 10000 \
    --val_data_size 1000 \
    --test_data_size 1000 \
    --experiment_name quick_test_n10 \
    --delete_bias_warmup_epochs 10
