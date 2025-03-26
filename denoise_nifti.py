#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import traceback

import nibabel as nib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# Import project modules
from n2v.models.network import N2VNetwork
from n2v.data.preprocessing import BrainPreprocessor, PreprocessingParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_running_in_background():
    """
    Check if the script is running in the background by checking if stdout is connected to a terminal.
    
    Returns:
        bool: True if running in background, False otherwise
    """
    return not sys.stdout.isatty()

def load_processed_files_log(output_dir: Path) -> set:
    """
    Load the list of already processed files from the log file.
    
    Args:
        output_dir: Directory containing the log file
        
    Returns:
        set: Set of file stems that have been processed
    """
    log_file = output_dir / "processed_files.json"
    if not log_file.exists():
        return set()
    
    try:
        with open(log_file, 'r') as f:
            processed_data = json.load(f)
            return set(processed_data.keys())
    except (json.JSONDecodeError, FileNotFoundError) as e:
        logger.warning(f"Error loading processed files log: {e}")
        return set()

def update_processed_files_log(output_dir: Path, file_stem: str, metadata: dict = None):
    """
    Update the log file with a newly processed file.
    
    Args:
        output_dir: Directory containing the log file
        file_stem: Stem of the processed file
        metadata: Optional metadata to store with the file entry
    """
    log_file = output_dir / "processed_files.json"
    
    # Load existing data
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                processed_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            processed_data = {}
    else:
        processed_data = {}
    
    # Add new entry
    if metadata is None:
        metadata = {}
    
    # Add timestamp
    metadata['timestamp'] = datetime.now().isoformat()
    processed_data[file_stem] = metadata
    
    # Save updated data
    with open(log_file, 'w') as f:
        json.dump(processed_data, f, indent=2)

def verify_dimensions(nifti_file: Path, model: N2VNetwork) -> bool:
    """
    Verify if the NIFTI file dimensions are compatible with the model.

    Args:
        nifti_file: Path to NIFTI file
        model: Loaded N2VNetwork model

    Returns:
        bool: True if dimensions are compatible
    """
    try:
        img = nib.load(str(nifti_file))
        data = img.get_fdata()

        # Get expected patch size from model
        patch_size = model.input_shape[0]

        # Check if dimensions are sufficient for patch size
        if data.shape[0] < patch_size or data.shape[1] < patch_size:
            logger.error(f"File {nifti_file.name} dimensions {data.shape[:2]} too small for patch size {patch_size}")
            return False

        logger.info(f"File {nifti_file.name} dimensions: {data.shape}")
        return True

    except Exception as e:
        logger.error(f"Error checking dimensions for {nifti_file}: {str(e)}")
        return False

def test_inference(model: N2VNetwork, nifti_file: Path) -> bool:
    """
    Test inference on a single slice of the NIFTI file.

    Args:
        model: Loaded N2VNetwork model
        nifti_file: Path to NIFTI file

    Returns:
        bool: True if test inference succeeds
    """
    try:
        img = nib.load(str(nifti_file))
        data = img.get_fdata()

        # Try processing middle slice
        mid_slice = data.shape[2] // 2
        slice_stack = data[:, :, max(0, mid_slice-2):min(data.shape[2], mid_slice+3)]

        # Ensure we have enough context slices
        if slice_stack.shape[2] < 5:
            logger.error(f"Not enough slices for context in {nifti_file.name}")
            return False

        # Try denoising the slice
        _ = model.denoise_slice(slice_stack)
        logger.info(f"Test inference successful on {nifti_file.name}")
        return True

    except Exception as e:
        logger.error(f"Test inference failed for {nifti_file}: {str(e)}")
        return False

def process_files(
    model: N2VNetwork,
    input_files: list,
    output_dir: Path,
    processed_files: set = None,
    verify: bool = True,
    resume: bool = True,
    use_progress_bar: bool = True,
    batch_size: int = 32
):
    """
    Process a list of NIFTI files with the model, with resume capability.

    Args:
        model: Loaded N2VNetwork model
        input_files: List of input NIFTI files
        output_dir: Directory to save processed files
        processed_files: Optional set of already processed files to skip
        verify: Whether to verify dimensions and test inference
        resume: Whether to check for and skip already processed files
        use_progress_bar: Whether to display progress bars
        batch_size: Batch size for processing slices
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load already processed files if resuming
    if resume:
        if processed_files is None:
            processed_files = set()

        # Check output directory for completed files
        for completed_file in output_dir.glob("*_denoised.nii*"):
            base_name = completed_file.stem.replace("_denoised", "")
            processed_files.add(base_name)

        if processed_files:
            logger.info(f"Found {len(processed_files)} already processed files")

    # Create or append to progress log
    progress_log = output_dir / "processing_progress.txt"
    with open(progress_log, "a") as f:
        f.write(f"\nProcessing session started at {datetime.now()}\n")
        if processed_files:
            f.write(f"Resuming from previous session. {len(processed_files)} files already processed.\n")

    # Verify compatibility if requested
    if verify:
        logger.info("Verifying dataset compatibility...")
        # Find first unprocessed file for testing
        test_file = next((f for f in input_files if f.stem not in processed_files), None)
        if test_file is None:
            logger.warning("All files already processed!")
            return

        if not verify_dimensions(test_file, model):
            raise ValueError(f"Incompatible dimensions in {test_file}")

        if not test_inference(model, test_file):
            raise ValueError(f"Inference test failed on {test_file}")

        logger.info("Dataset verification successful")

    # Count total and remaining files
    total_files = len(input_files)
    remaining_files = len([f for f in input_files if f.stem not in processed_files])
    logger.info(f"Total files: {total_files}, Remaining: {remaining_files}")
    
    if remaining_files == 0:
        logger.info("No files to process. All files have been denoised.")
        return processed_files

    # Process files with progress tracking
    with open(progress_log, "a") as f:
        # Choose iterator based on progress bar preference
        if use_progress_bar:
            file_iterator = tqdm(
                [f for f in input_files if f.stem not in processed_files], 
                desc=f"Denoising NIFTI files",
                unit="file"
            )
        else:
            file_iterator = [f for f in input_files if f.stem not in processed_files]
            
        for nifti_path in file_iterator:
            try:
                # Skip if already processed
                if resume and nifti_path.stem in processed_files:
                    continue

                # Create output path
                output_path = output_dir / f"{nifti_path.stem}_denoised{nifti_path.suffix}"
                
                if not use_progress_bar:
                    logger.info(f"Processing {nifti_path.name}...")

                # Denoise and save
                _ = model.denoise_nifti(
                    nifti_path,
                    output_path=output_path,
                    show_progress=use_progress_bar,
                    batch_size=batch_size
                )

                # Log success
                metadata = {
                    'original_file': str(nifti_path),
                    'output_file': str(output_path),
                    'timestamp': datetime.now().isoformat()
                }
                update_processed_files_log(output_dir, nifti_path.stem, metadata)
                
                log_message = f"Successfully processed: {nifti_path.name} at {datetime.now()}\n"
                f.write(log_message)
                
                if not use_progress_bar:
                    logger.info(log_message.strip())

            except Exception as e:
                error_message = f"Error processing {nifti_path}: {str(e)}\n"
                f.write(error_message)
                logger.error(error_message.strip())
                continue

    # Write completion summary
    with open(progress_log, "a") as f:
        f.write(f"\nProcessing session completed at {datetime.now()}\n")
        f.write(f"Total files processed in this session: {len(processed_files)}\n")

    return processed_files

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Denoise NIFTI files using pre-trained Noise2Void model')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing NIFTI files to denoise')
    parser.add_argument('--output_dir', type=str, default='./denoised_output', help='Directory to save denoised files')
    parser.add_argument('--checkpoint', type=str, default='models/checkpoint_epoch_014.keras', help='Path to model checkpoint')
    parser.add_argument('--force_reprocess', action='store_true', help='Reprocess already processed files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar (for background jobs)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for denoising')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')
    parser.add_argument('--skip_verification', action='store_true', help='Skip dimension and inference verification')
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Convert paths to Path objects
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        checkpoint_path = Path(args.checkpoint)
        
        # Check if paths exist
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory {input_dir} does not exist")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist")
        
        # Check if running in background and decide whether to use progress bar
        is_background = is_running_in_background()
        use_progress_bar = not (is_background or args.no_progress_bar)
        
        if is_background:
            logger.info("Detected background execution - disabling progress bars")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load list of already processed files
        processed_files = set()
        if not args.force_reprocess:
            processed_files = load_processed_files_log(output_dir)
            
        # Find all NIFTI files in input directory
        input_files = sorted(list(input_dir.glob("*.nii*")))
        
        if not input_files:
            logger.error(f"No NIFTI files found in {input_dir}")
            return 1
        
        logger.info(f"Found {len(input_files)} NIFTI files in {input_dir}")
        
        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        model = N2VNetwork.load_pretrained(str(checkpoint_path))
        logger.info("Model loaded successfully")
        
        # Process files
        process_files(
            model=model,
            input_files=input_files,
            output_dir=output_dir,
            processed_files=processed_files,
            verify=not args.skip_verification,
            resume=not args.force_reprocess,
            use_progress_bar=use_progress_bar,
            batch_size=args.batch_size
        )
        
        logger.info("Denoising completed successfully!")
        logger.info(f"Denoised files saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error("Error during denoising:", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 