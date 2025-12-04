
............................................................................................................................................
python tsp_custom/train.py \
    --num_loc 10 \
    --max_epochs 100 \
    --train_data_size 512 \
    --val_data_size 64 \
    --batch_size 128 \
    --val_batch_size 128 \
    --lr 5e-4 \
    --delete_every_n_steps 4 \
    --num_workers 31 \
    --enable_metrics_plot \
    --use_combined_metrics \
    --metrics_plot_every_n_epochs 1 \
    --enable_viz \
    --num_viz_instances 5 \
    --viz_fps 2 \
    --viz_every_n_epochs 1 \
    --experiment_name "ImprovedRewardScaling_N10"
