"""
Evaluation script for trained Hierarchical Transformer VAE models.

This script provides utilities for evaluating trained models,
generating samples, and analyzing model performance.
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from configs.config import Config
from src.data import PoseDataLoader
from src.models import HierarchicalTransformerVAE
from src.training import HTVAETrainer
from src.utils import Logger, ConfigManager, Visualizer, FileUtils


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate Hierarchical Transformer VAE")
    
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/pose_csvs",
                       help="Directory containing pose CSV files")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=16,
                       help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for evaluation")
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations")
    parser.add_argument("--extract_latents", action="store_true",
                       help="Extract latent representations")
    
    return parser.parse_args()


def load_model_and_config(checkpoint_path, config_path=None):
    """Load trained model and configuration.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config file
        
    Returns:
        Tuple of (model, config)
    """
    # Load config
    if config_path:
        config = ConfigManager.load_config(config_path)
    else:
        # Try to find config in checkpoint directory
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_file = os.path.join(checkpoint_dir, "..", "logs", "config.json")
        if os.path.exists(config_file):
            config = ConfigManager.load_config(config_file)
        else:
            config = Config()
            print("Warning: Using default config. Results may not be accurate.")
    
    # Create model
    model = HierarchicalTransformerVAE(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, config


def evaluate_reconstruction(model, data_loader, device, num_batches=None):
    """Evaluate reconstruction quality.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to use
        num_batches: Number of batches to evaluate (None for all)
        
    Returns:
        Dictionary with reconstruction metrics
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if num_batches and i >= num_batches:
                break
            
            sequences = batch['sequence'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)
            
            # Forward pass
            output = model(sequences, mask=mask)
            reconstruction = output['reconstruction']
            
            # Compute metrics
            if mask is not None:
                # Apply mask to both predictions and targets
                valid_positions = mask.unsqueeze(-1).expand_as(sequences)
                mse = ((reconstruction - sequences) ** 2 * valid_positions).sum() / valid_positions.sum()
                mae = (torch.abs(reconstruction - sequences) * valid_positions).sum() / valid_positions.sum()
            else:
                mse = torch.mean((reconstruction - sequences) ** 2)
                mae = torch.mean(torch.abs(reconstruction - sequences))
            
            total_mse += mse.item() * sequences.size(0)
            total_mae += mae.item() * sequences.size(0)
            total_samples += sequences.size(0)
    
    return {
        'reconstruction_mse': total_mse / total_samples,
        'reconstruction_mae': total_mae / total_samples
    }


def generate_samples(model, config, device, num_samples=16):
    """Generate new samples from the model.
    
    Args:
        model: Trained model
        config: Configuration
        device: Device to use
        num_samples: Number of samples to generate
        
    Returns:
        Generated samples tensor
    """
    model.eval()
    
    with torch.no_grad():
        samples = model.generate(
            batch_size=num_samples,
            sequence_length=config.data.sequence_length,
            device=device
        )
    
    return samples


def extract_latent_representations(model, data_loader, device):
    """Extract latent representations for all data.
    
    Args:
        model: Trained model
        data_loader: Data loader
        device: Device to use
        
    Returns:
        Numpy array of latent representations
    """
    model.eval()
    latent_codes = []
    
    with torch.no_grad():
        for batch in data_loader:
            sequences = batch['sequence'].to(device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(device)
            
            # Get latent representation
            latent = model.get_latent_representation(sequences, mask=mask)
            latent_codes.append(latent.cpu().numpy())
    
    return np.concatenate(latent_codes, axis=0)


def main():
    """Main evaluation function."""
    args = parse_arguments()
    
    # Set up output directory
    FileUtils.ensure_dir_exists(args.output_dir)
    
    # Set up logger
    log_file = os.path.join(args.output_dir, "evaluation.log")
    logger = Logger("HT-VAE-Eval", log_file=log_file)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Load model and config
        logger.info("Loading model and configuration...")
        model, config = load_model_and_config(args.checkpoint, args.config)
        model = model.to(device)
        
        # Update data directory if provided
        if args.data_dir:
            config.data.data_dir = args.data_dir
        
        # Create data loaders
        logger.info("Setting up data loaders...")
        data_loader = PoseDataLoader(config)
        train_loader, val_loader, test_loader = data_loader.setup_data()
        
        # Evaluate on test set
        logger.info("Evaluating reconstruction quality on test set...")
        test_metrics = evaluate_reconstruction(model, test_loader, device)
        logger.info(f"Test reconstruction MSE: {test_metrics['reconstruction_mse']:.6f}")
        logger.info(f"Test reconstruction MAE: {test_metrics['reconstruction_mae']:.6f}")
        
        # Save metrics
        metrics_path = os.path.join(args.output_dir, "test_metrics.json")
        FileUtils.save_json(test_metrics, metrics_path)
        
        # Generate samples
        logger.info(f"Generating {args.num_samples} samples...")
        generated_samples = generate_samples(model, config, device, args.num_samples)
        samples_path = os.path.join(args.output_dir, "generated_samples.pt")
        torch.save(generated_samples, samples_path)
        logger.info(f"Generated samples saved to: {samples_path}")
        
        # Extract latent representations if requested
        if args.extract_latents:
            logger.info("Extracting latent representations...")
            test_latents = extract_latent_representations(model, test_loader, device)
            latents_path = os.path.join(args.output_dir, "test_latents.npy")
            np.save(latents_path, test_latents)
            logger.info(f"Latent representations saved to: {latents_path}")
        
        # Generate visualizations if requested
        if args.visualize:
            logger.info("Generating visualizations...")
            visualizer = Visualizer(save_dir=args.output_dir)
            
            # Plot some generated samples
            sample_indices = np.random.choice(args.num_samples, size=min(4, args.num_samples), replace=False)
            for i, idx in enumerate(sample_indices):
                sample_seq = generated_samples[idx].cpu().numpy()
                # Reshape from [seq_len, input_dim] to [seq_len, num_joints, feature_dim]
                sample_seq = sample_seq.reshape(sample_seq.shape[0], config.data.num_joints, config.data.feature_dim)
                
                save_path = os.path.join(args.output_dir, f"generated_sample_{i}.png")
                visualizer.plot_pose_sequence(sample_seq, save_path=save_path)
            
            # Plot reconstructions
            test_batch = next(iter(test_loader))
            test_sequences = test_batch['sequence'][:4].to(device)
            
            with torch.no_grad():
                output = model(test_sequences)
                reconstructions = output['reconstruction']
            
            orig_seq = test_sequences[0].cpu().numpy().reshape(-1, config.data.num_joints, config.data.feature_dim)
            recon_seq = reconstructions[0].cpu().numpy().reshape(-1, config.data.num_joints, config.data.feature_dim)
            
            visualizer.plot_reconstruction_comparison(
                orig_seq[None], recon_seq[None], num_samples=1,
                save_path=os.path.join(args.output_dir, "reconstruction_comparison.png")
            )
            
            # Plot latent space if extracted
            if args.extract_latents:
                visualizer.plot_latent_space(
                    test_latents, method="pca",
                    save_path=os.path.join(args.output_dir, "latent_space_pca.png")
                )
                
                visualizer.plot_latent_space(
                    test_latents, method="tsne",
                    save_path=os.path.join(args.output_dir, "latent_space_tsne.png")
                )
            
            logger.info("Visualizations saved to output directory")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
