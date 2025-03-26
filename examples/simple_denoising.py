#!/usr/bin/env python3
"""
Simple example that demonstrates how to use the Noise2Void MRI denoiser.
This script denoises all NIFTI files in a directory and shows a before/after comparison.
"""

import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

# Add parent directory to path to import from main package
sys.path.insert(0, str(Path(__file__).parent.parent))

from n2v.models.network import N2VNetwork

def show_comparison(original_file, denoised_file):
    """Show a before/after comparison of the middle slice"""
    # Load the files
    orig_img = nib.load(original_file)
    denoised_img = nib.load(denoised_file)
    
    # Get the data
    orig_data = orig_img.get_fdata()
    denoised_data = denoised_img.get_fdata()
    
    # Get the middle slice for display
    mid_slice = orig_data.shape[2] // 2
    orig_slice = orig_data[:, :, mid_slice]
    denoised_slice = denoised_data[:, :, mid_slice]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display the original slice
    ax1.imshow(orig_slice, cmap='gray')
    ax1.set_title('Original')
    ax1.axis('off')
    
    # Display the denoised slice
    ax2.imshow(denoised_slice, cmap='gray')
    ax2.set_title('Denoised')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate Noise2Void MRI denoising')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing NIFTI files')
    parser.add_argument('--output_dir', type=str, default='./denoised', help='Output directory')
    parser.add_argument('--show_results', action='store_true', help='Show before/after comparison')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all NIFTI files
    input_dir = Path(args.input_dir)
    nifti_files = list(input_dir.glob('*.nii*'))
    
    if not nifti_files:
        print(f"No NIFTI files found in {input_dir}")
        return
    
    print(f"Found {len(nifti_files)} NIFTI files")
    
    # Load model
    model_path = Path(__file__).parent.parent / 'models' / 'checkpoint_epoch_014.keras'
    print(f"Loading model from {model_path}")
    model = N2VNetwork.load_pretrained(str(model_path))
    
    # Process each file
    for nifti_file in nifti_files:
        output_file = Path(args.output_dir) / f"{nifti_file.stem}_denoised{nifti_file.suffix}"
        print(f"Processing {nifti_file.name}...")
        
        # Denoise the file
        _ = model.denoise_nifti(
            str(nifti_file),
            output_path=str(output_file),
            show_progress=True
        )
        
        print(f"Saved denoised file to {output_file}")
        
        # Show comparison if requested
        if args.show_results:
            show_comparison(str(nifti_file), str(output_file))

if __name__ == "__main__":
    main() 