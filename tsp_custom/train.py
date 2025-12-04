"""
Training script for Custom TSP Model with Edge Selection/Deletion

This script implements Step 8 of USER_DEVELOPMENT_PLAN.txt:
- Trains CustomPOMOPolicy with CustomTSPEnv
- Uses REINFORCE with shared baseline (POMO-style)
- Integrates with PyTorch Lightning via RL4COTrainer
- Includes logging, checkpointing, and callbacks
- Supports visualization during validation

Usage:
    python tsp_custom/train.py --num_loc 20 --max_epochs 100 --batch_size 64
    
Author: Step 8 implementation
Date: November 22, 2025
"""

import sys
import os
import argparse
import torch
import logging
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tsp_custom.envs import CustomTSPEnv, CustomTSPGenerator
from tsp_custom.models import CustomPOMOPolicy
from tsp_custom.models.custom_pomo_model import CustomPOMOModel  # Will create this
from tsp_custom.callbacks import (
    ValidationVisualizationCallback,
    MetricsPlotCallback,
    CombinedMetricsPlotCallback,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Custom TSP Model with Edge Selection/Deletion'
    )
    
    # Environment arguments
    parser.add_argument('--num_loc', type=int, default=20,
                       help='Number of nodes in TSP instance (default: 20)')
    parser.add_argument('--delete_every_n_steps', type=int, default=4,
                       help='Allow deletions only every N steps to prevent loops (default: 4, set to 1 for always)')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension (default: 128)')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                       help='Number of transformer encoder layers (default: 6)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--feedforward_dim', type=int, default=512,
                       help='Feedforward dimension (default: 512)')
    
    # Delete bias scheduling
    parser.add_argument('--delete_bias_start', type=float, default=-5.0,
                       help='Initial delete bias (default: -5.0)')
    parser.add_argument('--delete_bias_end', type=float, default=0.0,
                       help='Final delete bias (default: 0.0)')
    parser.add_argument('--delete_bias_warmup_epochs', type=int, default=100,
                       help='Epochs for delete bias warmup (default: 100)')
    
    # POMO arguments
    parser.add_argument('--num_starts', type=int, default=4,
                       help='Number of multi-start rollouts per problem (default: 4)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Training batch size (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=1024,
                       help='Validation batch size (default: 1024)')
    parser.add_argument('--test_batch_size', type=int, default=1024,
                       help='Test batch size (default: 1024)')
    
    parser.add_argument('--train_data_size', type=int, default=1_280_000,
                       help='Training dataset size (default: 1,280,000)')
    parser.add_argument('--val_data_size', type=int, default=10_000,
                       help='Validation dataset size (default: 10,000)')
    parser.add_argument('--test_data_size', type=int, default=10_000,
                       help='Test dataset size (default: 10,000)')
    
    # Optimizer arguments
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                       help='Weight decay (default: 1e-6)')
    
    # Trainer arguments
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs (default: 100)')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value (default: 1.0)')
    parser.add_argument('--accelerator', type=str, default='auto',
                       help='Accelerator type (default: auto)')
    parser.add_argument('--devices', type=int, default=1,
                       help='Number of devices (default: 1)')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='tsp_custom/lightning_logs',
                       help='Logging directory (default: tsp_custom/lightning_logs)')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='tsp_custom',
                       help='W&B project name (default: tsp_custom)')
    
    # Checkpointing arguments
    parser.add_argument('--checkpoint_dir', type=str, default='tsp_custom/checkpoints',
                       help='Checkpoint directory (default: tsp_custom/checkpoints)')
    parser.add_argument('--save_top_k', type=int, default=3,
                       help='Save top k checkpoints (default: 3)')
    
    # Visualization/Metrics arguments (Step 9)
    parser.add_argument('--enable_viz', action='store_true',
                       help='Enable validation visualization callbacks')
    parser.add_argument('--viz_every_n_epochs', type=int, default=5,
                       help='Create visualizations every N epochs (default: 5)')
    parser.add_argument('--num_viz_instances', type=int, default=3,
                       help='Number of validation instances to visualize (default: 3)')
    parser.add_argument('--viz_fps', type=int, default=2,
                       help='Frames per second for GIF visualizations (default: 2)')
    parser.add_argument('--enable_metrics_plot', action='store_true',
                       help='Enable metrics plotting callback')
    parser.add_argument('--metrics_plot_every_n_epochs', type=int, default=1,
                       help='Create metrics plots every N epochs (default: 1)')
    parser.add_argument('--use_combined_metrics', action='store_true',
                       help='Use combined metrics dashboard instead of individual plots')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers (default: 4)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed for reproducibility
    if args.seed is not None:
        L.seed_everything(args.seed, workers=True)
        log.info(f"Set random seed to {args.seed}")
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"CustomTSP_N{args.num_loc}_E{args.max_epochs}_B{args.batch_size}"
    
    log.info("="*80)
    log.info(f"Training Custom TSP Model: {args.experiment_name}")
    log.info("="*80)
    log.info(f"Environment: N={args.num_loc}")
    log.info(f"Model: embed_dim={args.embed_dim}, layers={args.num_encoder_layers}")
    log.info(f"Training: epochs={args.max_epochs}, batch_size={args.batch_size}")
    log.info(f"Delete bias: {args.delete_bias_start} → {args.delete_bias_end} over {args.delete_bias_warmup_epochs} epochs")
    log.info("="*80)
    
    # Create environment
    log.info("Creating environment...")
    generator = CustomTSPGenerator(num_loc=args.num_loc)
    env = CustomTSPEnv(
        generator=generator,
        delete_every_n_steps=args.delete_every_n_steps
    )
    log.info(f"Environment: {env.name}, N={args.num_loc}, delete_every_n_steps={args.delete_every_n_steps}")
    
    # Create policy
    log.info("Creating policy...")
    policy = CustomPOMOPolicy(
        num_loc=args.num_loc,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        feedforward_dim=args.feedforward_dim,
        normalization='instance',
        delete_bias_start=args.delete_bias_start,
        delete_bias_end=args.delete_bias_end,
        delete_bias_warmup_epochs=args.delete_bias_warmup_epochs,
    )
    
    num_params = sum(p.numel() for p in policy.parameters())
    log.info(f"Policy parameters: {num_params:,}")
    
    # Create model (POMO-style REINFORCE)
    log.info("Creating model...")
    model = CustomPOMOModel(
        env=env,
        policy=policy,
        baseline='shared',  # POMO uses shared baseline
        num_starts=args.num_starts,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        test_batch_size=args.test_batch_size,
        train_data_size=args.train_data_size,
        val_data_size=args.val_data_size,
        test_data_size=args.test_data_size,
        optimizer_kwargs={
            'lr': args.lr,
            'weight_decay': args.weight_decay
        },
    )
    log.info(f"Model: {model.__class__.__name__} with {args.num_starts} starts and shared baseline")
    
    # Create callbacks
    log.info("Setting up callbacks...")
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename=f"{args.experiment_name}_epoch{{epoch:03d}}_reward{{val/reward:.4f}}",
        save_top_k=args.save_top_k,
        save_last=True,
        monitor="val/reward",
        mode="max",
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    log.info(f"Checkpoint callback: save_top_k={args.save_top_k}, monitor='val/reward'")
    
    # Model summary callback
    rich_model_summary = RichModelSummary(max_depth=3)
    callbacks.append(rich_model_summary)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Visualization callback (Step 9)
    if args.enable_viz:
        viz_callback = ValidationVisualizationCallback(
            num_instances=args.num_viz_instances,
            fps=args.viz_fps,
            every_n_epochs=args.viz_every_n_epochs,
        )
        callbacks.append(viz_callback)
        log.info(f"Validation visualization enabled: {args.num_viz_instances} instances, every {args.viz_every_n_epochs} epochs")
    
    # Metrics plotting callback (Step 9)
    if args.enable_metrics_plot:
        if args.use_combined_metrics:
            metrics_callback = CombinedMetricsPlotCallback(
                plot_every_n_epochs=args.metrics_plot_every_n_epochs,
            )
            log.info(f"Combined metrics plotting enabled: every {args.metrics_plot_every_n_epochs} epochs")
        else:
            metrics_callback = MetricsPlotCallback(
                plot_every_n_epochs=args.metrics_plot_every_n_epochs,
            )
            log.info(f"Individual metrics plotting enabled: every {args.metrics_plot_every_n_epochs} epochs")
        callbacks.append(metrics_callback)

    
    # Setup logger
    log.info("Setting up logger...")
    if args.use_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.experiment_name,
            save_dir=args.log_dir,
        )
        log.info(f"Using W&B logger: project={args.wandb_project}")
    else:
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name=args.experiment_name,
        )
        log.info(f"Using TensorBoard logger: {args.log_dir}")
    
    # Create trainer
    log.info("Creating trainer...")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=1,  # Important for RL: generate new data each epoch
        log_every_n_steps=1,  # Log every batch for detailed monitoring
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    log.info(f"Trainer: max_epochs={args.max_epochs}, accelerator={args.accelerator}, devices={args.devices}")
    
    # Train
    log.info("="*80)
    log.info("Starting training...")
    log.info("="*80)
    
    trainer.fit(
        model,
        ckpt_path=args.resume_from_checkpoint
    )
    
    # Test
    log.info("="*80)
    log.info("Running test...")
    log.info("="*80)
    
    trainer.test(model)
    
    # Save final model
    final_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_final.ckpt")
    trainer.save_checkpoint(final_path)
    log.info(f"Saved final checkpoint to {final_path}")
    
    log.info("="*80)
    log.info("Training complete!")
    log.info("="*80)
    
    # Print best checkpoint info
    if checkpoint_callback.best_model_path:
        log.info(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        log.info(f"Best reward: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    main()
