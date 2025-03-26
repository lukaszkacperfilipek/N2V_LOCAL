# Noise2Void MRI Denoiser

A tool for denoising brain MRI scans using the Noise2Void deep learning method. This project enables efficient denoising of NIFTI (.nii) files with an easy-to-use command-line interface.

![Brain MRI Denoising Example](https://i.imgur.com/PLACEHOLDER.jpg)

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Basic Usage](#basic-usage)
  - [Command Line Arguments](#command-line-arguments)
  - [Use Cases](#use-cases)
- [Project Structure](#project-structure)
- [About Noise2Void](#about-noise2void)
  - [How It Works](#how-it-works)
  - [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.0+
- CUDA-capable GPU (recommended for faster processing)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/noise2void-mri-denoiser.git
   cd noise2void-mri-denoiser
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the setup script to organize the project structure:
   ```bash
   python setup_project.py
   ```

## Quick Start

Denoise a directory of NIFTI files with default settings:

```bash
python denoise_nifti.py --input_dir /path/to/nifti_files
```

The denoised files will be saved to `./denoised_output` by default.

## Usage Guide

### Basic Usage

To denoise a directory of NIFTI files:

```bash
python denoise_nifti.py --input_dir /path/to/nifti_files --output_dir /path/to/save/results
```

This will:
1. Process all `.nii` and `.nii.gz` files in the input directory
2. Save denoised versions to the output directory with `_denoised` suffix
3. Show a progress bar for each file being processed
4. Skip any files that have already been processed (resumable)

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input_dir` | Directory containing NIFTI files to denoise | (Required) |
| `--output_dir` | Directory to save denoised files | `./denoised_output` |
| `--checkpoint` | Path to model checkpoint file | `models/checkpoint_epoch_014.keras` |
| `--force_reprocess` | Reprocess already processed files | False |
| `--no_progress_bar` | Disable progress bar (for background jobs) | False |
| `--batch_size` | Batch size for denoising | 32 |
| `--log_level` | Set logging level (DEBUG, INFO, WARNING, ERROR) | INFO |
| `--skip_verification` | Skip dimension verification and test inference | False |

### Use Cases

#### Standard Denoising

```bash
python denoise_nifti.py --input_dir /path/to/nifti_files --output_dir /path/to/save/results
```

#### Background Processing (Server/Cluster)

```bash
nohup python denoise_nifti.py --input_dir /path/to/nifti_files --no_progress_bar > denoising.log 2>&1 &
```

#### Continue Interrupted Process

If a previous run was interrupted, simply run the same command again:

```bash
python denoise_nifti.py --input_dir /path/to/nifti_files --output_dir /path/to/save/results
```

The script will automatically detect and skip already processed files.

#### Force Reprocessing

To reprocess all files, even if they've been processed before:

```bash
python denoise_nifti.py --input_dir /path/to/nifti_files --output_dir /path/to/save/results --force_reprocess
```

#### Custom Model Checkpoint

```bash
python denoise_nifti.py --input_dir /path/to/nifti_files --checkpoint /path/to/your/model.keras
```

#### Performance Tuning

Adjust batch size for your GPU:

```bash
python denoise_nifti.py --input_dir /path/to/nifti_files --batch_size 16
```

- Smaller batch size: Use for GPUs with less memory
- Larger batch size: Potentially faster on high-end GPUs

## Project Structure

```
noise2void-mri-denoiser/
├── denoise_nifti.py       # Main script for denoising
├── setup_project.py       # Script to set up project structure
├── README.md              # This documentation file
├── models/                # Directory for model checkpoints
│   └── checkpoint_epoch_014.keras  # Pre-trained model
└── n2v/                   # Core library code
    ├── models/            # Neural network models
    │   └── network.py     # N2VNetwork implementation
    ├── data/              # Data handling
    │   ├── preprocessing.py  # MRI preprocessing 
    │   ├── generator.py      # Data generation
    │   └── augmentation.py   # Data augmentation
    └── config.py          # Configuration parameters
```

## About Noise2Void

### How It Works

Noise2Void is a self-supervised deep learning technique for image denoising that can be trained without clean reference data. The key idea is to use a special training approach where:

1. The network learns to predict the value of a pixel from its surrounding context
2. Some input pixels are randomly "masked" during training
3. The network is trained to predict these masked pixels from surrounding pixels
4. This forces the network to learn the underlying signal while ignoring noise

In this implementation:
- We use a 2.5D approach for brain MRI, using context slices in addition to the target slice
- A U-Net architecture with skip connections forms the backbone of the model
- The model handles slice-by-slice processing with proper context management

### Technical Details

**Network Architecture:**
- U-Net with 4 levels of depth
- 32 filters in the first layer, doubling at each level
- 2.5D approach with 5 input slices (1 central + 2 on each side)

**Processing Pipeline:**
1. Load NIFTI file
2. Preprocess and normalize slices
3. Process each slice with context information from neighboring slices
4. Blend together overlapping patches with weighted averaging
5. Reverse normalization and save denoised volume

**Performance Considerations:**
- GPU acceleration provides significant speedup
- Processing time depends on volume size and GPU capability
- Approximate processing time: 1-3 minutes per volume on a modern GPU

## Troubleshooting

### Common Issues

**Error loading model checkpoint:**
- Ensure the checkpoint file exists at the specified path
- Check if the TensorFlow version is compatible

**CUDA out of memory:**
- Reduce the batch size (`--batch_size`)
- Free up GPU memory from other processes

**Files not being processed:**
- Check if files are valid NIFTI format (.nii or .nii.gz)
- Ensure you have read/write permissions for the directories

### Logs and Debugging

The script creates several logs to help with debugging:
- Console output with timestamps
- `processing_progress.txt` in the output directory
- `processed_files.json` containing metadata about processed files

To get more detailed logs, run with:

```bash
python denoise_nifti.py --input_dir /path/to/nifti_files --log_level DEBUG
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```
Krull, A., Buchholz, T. O., & Jug, F. (2019). 
Noise2void-learning denoising from single noisy images. 
In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 2129-2137).
``` 