#!/bin/bash
#
# Test Step 9 callbacks with a quick training run
#
# This script runs 3 epochs with:
# - Validation visualization (every 2 epochs)
# - Combined metrics plotting (every epoch)
#

python tsp_custom/train.py \
    --num_loc 10 \
    --max_epochs 3 \
    --train_data_size 1000 \
    --val_data_size 100 \
    --batch_size 32 \
    --val_batch_size 100 \
    --num_workers 0 \
    --enable_viz \
    --viz_every_n_epochs 2 \
    --num_viz_instances 2 \
    --viz_fps 2 \
    --enable_metrics_plot \
    --use_combined_metrics \
    --metrics_plot_every_n_epochs 1 \
    --experiment_name "Test_Step9_Callbacks"
