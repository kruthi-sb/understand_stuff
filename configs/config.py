"""
Configuration file for Hierarchical Transformer VAE for Pose Data.

This module contains all the hyperparameters and configuration settings
for the HT-VAE model training and evaluation.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing.
    
    Attributes:
        data_dir: Directory path containing the CSV files
        num_samples: Total number of samples (CSV files)
        sequence_length: Maximum number of frames per sequence (4000)
        num_joints: Number of joints per frame (12)
        feature_dim: Number of features per joint (x, y coordinates = 2)
        chunk_size: Size of each chunk for hierarchical processing
        overlap_size: Overlap between consecutive chunks
        train_split: Fraction of data to use for training
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        normalize: Whether to normalize the pose data
        augment: Whether to apply data augmentation
    """
    data_dir: str = "data/pose_csvs"
    num_samples: int = 170
    sequence_length: int = 4000
    num_joints: int = 12
    feature_dim: int = 2  # x, y coordinates
    chunk_size: int = 50
    overlap_size: int = 10
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 16
    num_workers: int = 4
    normalize: bool = True
    augment: bool = True


@dataclass
class ModelConfig:
    """Configuration for the Hierarchical Transformer VAE model.
    
    Attributes:
        embed_dim: Embedding dimension for pose features
        hidden_dim: Hidden dimension for transformer layers
        num_heads: Number of attention heads in transformer
        num_layers_local: Number of transformer layers for local attention
        num_layers_global: Number of transformer layers for global attention
        latent_dim: Dimension of the latent space
        dropout_rate: Dropout rate for regularization
        activation: Activation function to use
        use_layer_norm: Whether to use layer normalization
        use_positional_encoding: Whether to use positional encoding
        max_position_embeddings: Maximum position for positional encoding
    """
    embed_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    num_layers_local: int = 4
    num_layers_global: int = 2
    latent_dim: int = 128
    dropout_rate: float = 0.1
    activation: str = "gelu"
    use_layer_norm: bool = True
    use_positional_encoding: bool = True
    max_position_embeddings: int = 5000


@dataclass
class TrainingConfig:
    """Configuration for training the HT-VAE model.
    
    Attributes:
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for regularization
        beta1: Beta1 parameter for Adam optimizer
        beta2: Beta2 parameter for Adam optimizer
        warmup_steps: Number of warmup steps for learning rate scheduler
        gradient_clip_norm: Maximum gradient norm for clipping
        kl_weight: Weight for KL divergence loss
        reconstruction_weight: Weight for reconstruction loss
        scheduler_type: Type of learning rate scheduler
        save_every: Save model every N epochs
        validate_every: Validate model every N epochs
        early_stopping_patience: Patience for early stopping
        device: Device to use for training (cuda/cpu)
        mixed_precision: Whether to use mixed precision training
        compile_model: Whether to compile the model (PyTorch 2.0+)
    """
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0
    kl_weight: float = 1.0
    reconstruction_weight: float = 1.0
    scheduler_type: str = "cosine"
    save_every: int = 10
    validate_every: int = 1
    early_stopping_patience: int = 15
    device: str = "cuda"
    mixed_precision: bool = True
    compile_model: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and logging.
    
    Attributes:
        experiment_name: Name of the experiment
        log_dir: Directory to save logs
        checkpoint_dir: Directory to save model checkpoints
        tensorboard_dir: Directory for tensorboard logs
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity name
        log_every: Log metrics every N steps
        save_predictions: Whether to save model predictions
        save_latent_representations: Whether to save latent representations
    """
    experiment_name: str = "ht_vae_pose"
    log_dir: str = "experiments/logs"
    checkpoint_dir: str = "experiments/checkpoints"
    tensorboard_dir: str = "experiments/tensorboard"
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    log_every: int = 100
    save_predictions: bool = True
    save_latent_representations: bool = True


@dataclass
class Config:
    """Main configuration class combining all sub-configurations.
    
    Attributes:
        data: Data configuration
        model: Model configuration
        training: Training configuration
        experiment: Experiment configuration
    """
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    experiment: ExperimentConfig = ExperimentConfig()
    
    def __post_init__(self):
        """Post-initialization validation and adjustments."""
        # Validate that splits sum to 1.0
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        # Calculate number of chunks based on sequence length and chunk size
        self.data.num_chunks = (self.data.sequence_length - self.data.overlap_size) // (
            self.data.chunk_size - self.data.overlap_size
        )
