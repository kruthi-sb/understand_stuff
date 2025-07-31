"""
Main training script for Hierarchical Transformer VAE.

This script provides the main entry point for training the HT-VAE model
on pose sequence data.
"""

import os
import sys
import argparse
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from configs.config import Config
from src.data import PoseDataLoader
from src.models import HierarchicalTransformerVAE
from src.training import HTVAETrainer, TrainingUtilities
from src.utils import Logger, ConfigManager, FileUtils


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train Hierarchical Transformer VAE for Pose Data")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/pose_csvs",
                       help="Directory containing pose CSV files")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    
    # Model arguments
    parser.add_argument("--embed_dim", type=int, default=256,
                       help="Embedding dimension")
    parser.add_argument("--latent_dim", type=int, default=128,
                       help="Latent space dimension")
    parser.add_argument("--num_heads", type=int, default=8,
                       help="Number of attention heads")
    parser.add_argument("--chunk_size", type=int, default=50,
                       help="Chunk size for hierarchical attention")
    
    # Training arguments
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for training")
    
    # Experiment arguments
    parser.add_argument("--experiment_name", type=str, default="ht_vae_pose",
                       help="Name of the experiment")
    parser.add_argument("--checkpoint_dir", type=str, default="experiments/checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="experiments/logs",
                       help="Directory to save logs")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file to override defaults")
    parser.add_argument("--debug", action="store_true",
                       help="Run in debug mode with reduced data")
    
    return parser.parse_args()


def setup_environment(args, config):
    """Set up the training environment.
    
    Args:
        args: Command line arguments
        config: Configuration object
    """
    # Set random seed for reproducibility
    TrainingUtilities.set_seed(args.seed)
    
    # Set up CUDA
    if torch.cuda.is_available() and args.device == "cuda":
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        print("Using CPU device")
    
    # Create directories
    FileUtils.ensure_dir_exists(config.experiment.checkpoint_dir)
    FileUtils.ensure_dir_exists(config.experiment.log_dir)
    FileUtils.ensure_dir_exists(config.experiment.tensorboard_dir)
    
    # Save configuration
    config_save_path = os.path.join(config.experiment.log_dir, "config.json")
    ConfigManager.save_config(config, config_save_path)


def create_data_loaders(config):
    """Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_loader = PoseDataLoader(config)
    train_loader, val_loader, test_loader = data_loader.setup_data()
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader


def create_model(config):
    """Create the HT-VAE model.
    
    Args:
        config: Configuration object
        
    Returns:
        HT-VAE model
    """
    model = HierarchicalTransformerVAE(config)
    
    # Print model information
    num_params = TrainingUtilities.count_parameters(model)
    model_size = TrainingUtilities.get_model_size(model)
    
    print(f"Model created with {num_params:,} trainable parameters")
    print(f"Model size: {model_size:.2f} MB")
    
    return model


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    if args.config:
        config = ConfigManager.load_config(args.config)
    else:
        config = Config()
    
    # Override config with command line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.embed_dim:
        config.model.embed_dim = args.embed_dim
    if args.latent_dim:
        config.model.latent_dim = args.latent_dim
    if args.num_heads:
        config.model.num_heads = args.num_heads
    if args.chunk_size:
        config.data.chunk_size = args.chunk_size
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.device:
        config.training.device = args.device
    if args.experiment_name:
        config.experiment.experiment_name = args.experiment_name
    if args.checkpoint_dir:
        config.experiment.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.experiment.log_dir = args.log_dir
    
    # Debug mode
    if args.debug:
        config.data.num_samples = min(config.data.num_samples, 20)
        config.training.num_epochs = 2
        config.training.validate_every = 1
        print("Running in debug mode with reduced data and epochs")
    
    # Set up environment
    setup_environment(args, config)
    
    # Set up logger
    log_file = os.path.join(config.experiment.log_dir, "training.log")
    logger = Logger("HT-VAE", log_file=log_file)
    logger.info(f"Starting experiment: {config.experiment.experiment_name}")
    
    try:
        # Create data loaders
        logger.info("Setting up data loaders...")
        train_loader, val_loader, test_loader = create_data_loaders(config)
        
        # Create model
        logger.info("Creating model...")
        model = create_model(config)
        
        # Move model to device
        device = torch.device(config.training.device)
        model = model.to(device)
        
        # Compile model if specified (PyTorch 2.0+)
        if config.training.compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model...")
            model = torch.compile(model)
        
        # Create trainer
        logger.info("Setting up trainer...")
        trainer = HTVAETrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            test_loader=test_loader
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer._load_checkpoint(args.resume)
        
        # Start training
        logger.info("Starting training...")
        training_history = trainer.train()
        
        # Test the final model
        logger.info("Testing final model...")
        test_metrics = trainer.test()
        
        logger.info("Training completed successfully!")
        logger.info(f"Final test metrics: {test_metrics}")
        
        # Generate sample outputs
        logger.info("Generating sample outputs...")
        samples_dir = os.path.join(config.experiment.log_dir, "samples")
        FileUtils.ensure_dir_exists(samples_dir)
        
        # Generate new samples
        sample_path = os.path.join(samples_dir, "generated_samples.pt")
        generated_samples = trainer.generate_samples(num_samples=8, save_path=sample_path)
        
        # Reconstruct some validation samples
        val_batch = next(iter(val_loader))
        val_sequences = val_batch['sequence'][:4]  # Take first 4 samples
        reconstruction_path = os.path.join(samples_dir, "reconstructions.pt")
        reconstructions = trainer.reconstruct_samples(val_sequences, save_path=reconstruction_path)
        
        logger.info(f"Sample outputs saved to: {samples_dir}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    
    finally:
        logger.info("Training script finished.")


if __name__ == "__main__":
    main()
