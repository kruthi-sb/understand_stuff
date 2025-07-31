# Hierarchical Transformer VAE for Pose Data

A PyTorch implementation of a Hierarchical Transformer Variational Autoencoder (HT-VAE) for modeling long pose sequences. This architecture combines the power of hierarchical attention mechanisms with variational autoencoders to efficiently process and generate realistic human pose sequences.

## Features

- **Hierarchical Attention**: Two-level attention mechanism for handling long sequences (up to 4000 frames)
- **Local & Global Modeling**: Captures both fine-grained movements and global behavior patterns
- **Efficient Processing**: Chunked sequence processing for memory efficiency
- **Flexible Architecture**: Configurable model parameters and training settings
- **Comprehensive Utilities**: Training, evaluation, visualization, and analysis tools

## Architecture Overview

The HT-VAE uses a hierarchical transformer architecture:

1. **Local Attention**: Processes small chunks of the sequence to capture detailed, short-term dependencies
2. **Chunk Summarization**: Aggregates chunk-level information into compact representations
3. **Global Attention**: Models long-term dependencies across chunk summaries
4. **VAE Framework**: Encodes sequences into a structured latent space for generation

## Data Format

The model expects pose data in CSV format with the following structure:
```
frame#,landmark_name,x,y,L2_distance_from_midhip,jd_x,jd_y,jo_x,jo_y,angle
0,RIGHT_WRIST,-0.50893,0.99949,1.12227,0.0,0.0,-0.15263,0.17708,167.35486
0,LEFT_WRIST,0.58417,0.92975,1.09879,0.0,0.0,0.22792,0.05209,153.85601
...
```

For this implementation, only `frame#`, `landmark_name`, `x`, and `y` columns are used.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd HT-VAE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place CSV files in the `data/pose_csvs/` directory
   - Each CSV file should contain one complete pose sequence
   - Ensure all files follow the expected format

## Usage

### Training

Train the model with default settings:
```bash
python train.py --data_dir data/pose_csvs
```

Train with custom parameters:
```bash
python train.py \
    --data_dir data/pose_csvs \
    --batch_size 32 \
    --num_epochs 200 \
    --learning_rate 5e-4 \
    --latent_dim 256 \
    --chunk_size 100 \
    --experiment_name my_experiment
```

### Evaluation

Evaluate a trained model:
```bash
python evaluate.py \
    --checkpoint experiments/checkpoints/best_model.pt \
    --data_dir data/pose_csvs \
    --output_dir evaluation_results \
    --visualize \
    --extract_latents
```

### Configuration

Modify `configs/config.py` to adjust:
- Model architecture parameters
- Training hyperparameters
- Data processing settings
- Experiment configuration

## Project Structure

```
HT-VAE/
├── configs/
│   └── config.py              # Configuration classes
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py         # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py     # Transformer components
│   │   └── vae.py            # VAE components
│   ├── training/
│   │   ├── __init__.py
│   │   ├── losses.py         # Loss functions
│   │   └── trainer.py        # Training utilities
│   └── utils/
│       ├── __init__.py
│       └── utils.py          # Utility functions
├── data/                      # Data directory
├── experiments/              # Experiment outputs
│   ├── checkpoints/         # Model checkpoints
│   ├── logs/               # Training logs
│   └── tensorboard/        # TensorBoard logs
├── train.py                 # Main training script
├── evaluate.py             # Evaluation script
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Model Components

### Core Classes

**Data Processing:**
- `PoseSequenceDataset`: PyTorch dataset for pose sequences
- `PoseDataLoader`: Data loader factory with preprocessing
- `PoseDataProcessor`: Utility functions for data processing

**Model Architecture:**
- `HierarchicalTransformerVAE`: Main model class
- `VAEEncoder`: Hierarchical transformer encoder
- `VAEDecoder`: Hierarchical transformer decoder
- `HierarchicalAttention`: Two-level attention mechanism
- `LatentSpace`: Latent space operations and sampling

**Training:**
- `HTVAETrainer`: Complete training pipeline
- `VAELoss`: Combined VAE loss with multiple components
- `LossManager`: Loss combination and scheduling

**Utilities:**
- `ModelCheckpoint`: Model saving and loading
- `EarlyStopping`: Training termination logic
- `MetricsTracker`: Metrics aggregation and tracking
- `Visualizer`: Visualization utilities

## Key Features

### Hierarchical Processing
- Sequences are split into overlapping chunks
- Local attention processes individual chunks
- Global attention models chunk-to-chunk dependencies
- Efficient handling of very long sequences (4000+ frames)

### VAE Framework
- Structured latent space for controllable generation
- KL divergence regularization for smooth interpolation
- Support for conditional and unconditional generation

### Training Features
- Mixed precision training for efficiency
- Gradient clipping and learning rate scheduling
- Early stopping and best model checkpointing
- Comprehensive metrics tracking and logging

### Evaluation Tools
- Reconstruction quality metrics
- Sample generation capabilities
- Latent space visualization
- Attention map analysis

## Customization

The codebase is designed to be modular and extensible:

1. **Add new loss functions** in `src/training/losses.py`
2. **Modify the architecture** in `src/models/`
3. **Add data augmentations** in `src/data/dataset.py`
4. **Extend evaluation metrics** in `evaluate.py`

## Performance Tips

1. **GPU Memory**: Adjust `chunk_size` and `batch_size` based on available GPU memory
2. **Sequence Length**: Longer sequences provide more context but require more memory
3. **Model Size**: Increase `embed_dim` and `num_layers` for better quality at the cost of speed
4. **Training Speed**: Enable model compilation (`compile_model=True`) for PyTorch 2.0+

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ht_vae_pose,
  title={Hierarchical Transformer VAE for Pose Sequence Modeling},
  author={Your Name},
  year={2025},
  howpublished={\\url{https://github.com/your-username/HT-VAE}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
