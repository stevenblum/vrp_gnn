#!/bin/bash
# Reasonable training script for N=20 (medium-scale testing)
# Balances dataset size with training time

echo "========================================================================"
echo "Training for Custom TSP N=20 (Reasonable Configuration)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Problem size: N=20"
echo "  - Dataset: 128,000 training instances (10% of POMO paper)"
echo "  - Batch size: 128"
echo "  - Epochs: 50"
echo "  - Expected: ~1,000 batches per epoch (~30-40 minutes per epoch)"
echo ""
echo "Press Ctrl+C to stop training"
echo "========================================================================"
echo ""

python tsp_custom/train.py \
    --num_loc 20 \
    --max_epochs 50 \
    --batch_size 128 \
    --train_data_size 128000 \
    --val_data_size 10000 \
    --test_data_size 10000 \
    --experiment_name train_n20 \
    --delete_bias_warmup_epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-6
