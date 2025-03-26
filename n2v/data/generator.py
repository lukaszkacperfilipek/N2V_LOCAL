import tensorflow as tf
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from threading import Lock
from .augmentation import SliceAugmenter

class N2VBrainGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        patch_size: int = 32,
        n_context_slices: int = 2,
        n_patches_per_slice: int = 2,
        pixels_per_patch: int = 100,
        validation: bool = False,
        seed: Optional[int] = None,
        preprocessor: Optional['BrainPreprocessor'] = None,
        **kwargs
    ):
        """
        Initialize the Noise2Void brain data generator with multiprocessing support.
        
        Additional kwargs supported:
        - workers: Number of worker processes
        - use_multiprocessing: Whether to enable multiprocessing
        - max_queue_size: Maximum size of the preprocessing queue
        """
        super().__init__(**kwargs)
        
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.n_context_slices = n_context_slices
        self.n_patches_per_slice = n_patches_per_slice
        self.pixels_per_patch = pixels_per_patch
        self.validation = validation
        self.seed = seed
        self.preprocessor = preprocessor
        
        # Thread safety locks
        self.slice_lock = Lock()
        self.batch_lock = Lock()
        
        # Initialize per-process RNG
        self.rng = np.random.RandomState(seed)
        
        # Initialize augmenter if not in validation mode
        self.augmenter = None if validation else SliceAugmenter(seed=seed)
        
        # Load dataset
        self.scans = self._load_dataset()
        self.total_slices = sum(img.shape[2] for _, img in self.scans)
        
        # For thread-safe slice tracking
        self.current_scan_idx = 0
        self.current_slice_idx = 0
        self.current_batch_patches = []
        
        # Pre-compute slice indices for better multiprocessing
        self.slice_indices = self._precompute_slice_indices()

    def _load_dataset(self) -> List[Tuple[Path, nib.Nifti1Image]]:
        """Load all .nii files from data directory."""
        nii_files = sorted(self.data_dir.glob("*.nii*"))
        if not nii_files:
            raise ValueError(f"No .nii files found in {self.data_dir}")
        
        scans = []
        for file_path in nii_files:
            img = nib.load(str(file_path))
            if len(img.shape) == 3:  # Basic validation
                scans.append((file_path, img))
            else:
                print(f"Skipping {file_path}: expected 3D image, got shape {img.shape}")
        
        if not scans:
            raise ValueError("No valid 3D .nii files found")
        return scans

    def _precompute_slice_indices(self) -> List[Tuple[int, int]]:
        """Pre-compute all valid slice indices for parallel processing."""
        indices = []
        for scan_idx, (_, img) in enumerate(self.scans):
            n_slices = img.shape[2]
            for slice_idx in range(n_slices):
                indices.append((scan_idx, slice_idx))
        return indices

    def _get_next_slice_indices(self):
        """Thread-safe method to get next slice indices."""
        with self.slice_lock:
            if self.current_scan_idx >= len(self.scans):
                self.current_scan_idx = 0
                self.current_slice_idx = 0
            
            _, img = self.scans[self.current_scan_idx]
            n_slices = img.shape[2]
            
            if self.current_slice_idx >= n_slices:
                self.current_scan_idx += 1
                self.current_slice_idx = 0
                if self.current_scan_idx >= len(self.scans):
                    self.current_scan_idx = 0
            
            if self.current_scan_idx < len(self.scans):
                scan_idx = self.current_scan_idx
                slice_idx = self.current_slice_idx
                self.current_slice_idx += 1
                return scan_idx, slice_idx
            
            return None, None

    def _get_context_slices(self, scan_idx: int, slice_idx: int) -> np.ndarray:
        """Get a slice and its context slices (thread-safe)."""
        try:
            _, img = self.scans[scan_idx]
            # Load data in a thread-safe manner
            with self.slice_lock:
                data = img.get_fdata().astype(np.float32)
            
            n_slices = data.shape[2]
            start_idx = max(0, slice_idx - self.n_context_slices)
            end_idx = min(n_slices, slice_idx + self.n_context_slices + 1)
            
            # Get slices
            slices = []
            for idx in range(start_idx, end_idx):
                slices.append(data[..., idx])
            
            # Pad if necessary
            while len(slices) < 2 * self.n_context_slices + 1:
                if len(slices) < slice_idx + self.n_context_slices + 1:
                    slices.append(slices[-1])
                else:
                    slices.insert(0, slices[0])
            
            slice_stack = np.stack(slices)
            
            # Apply preprocessing if available
            if self.preprocessor is not None:
                slice_stack = self.preprocessor.preprocess_slice_stack(
                    slice_stack,
                    file_path=self.scans[scan_idx][0]
                )
            
            # Apply augmentation if available and not in validation mode
            if self.augmenter is not None and not self.validation:
                slice_stack = self.augmenter.augment_context_slices(slice_stack)
            
            return slice_stack.astype(np.float32)
            
        except Exception as e:
            print(f"Error in _get_context_slices for scan {scan_idx}, slice {slice_idx}: {e}")
            return None

    def _extract_patches(self, slice_stack: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract random patches with proper validation handling.
        """
        slice_stack = slice_stack.astype(np.float32)
        n_slices, h, w = slice_stack.shape
        assert n_slices == 2 * self.n_context_slices + 1
        
        half_patch = self.patch_size // 2
        center_slice_idx = self.n_context_slices
        
        # Initialize arrays
        patches = np.zeros((self.n_patches_per_slice, self.patch_size, self.patch_size, n_slices), 
                        dtype=np.float32)
        targets = np.zeros((self.n_patches_per_slice, self.patch_size, self.patch_size, 1), 
                        dtype=np.float32)
        mask = np.zeros((self.n_patches_per_slice, self.patch_size, self.patch_size), 
                        dtype=bool)
        orig_vals = np.zeros((self.n_patches_per_slice, self.patch_size, self.patch_size), 
                            dtype=np.float32)
        
        # Valid ranges for patch centers
        y_min = half_patch
        y_max = h - half_patch
        x_min = half_patch
        x_max = w - half_patch
        
        for i in range(self.n_patches_per_slice):
            # Simple random sampling
            y = self.rng.randint(y_min, y_max)
            x = self.rng.randint(x_min, x_max)
            
            # Extract patch with context
            for j in range(n_slices):
                patches[i, :, :, j] = slice_stack[j,
                                                y - half_patch:y + half_patch,
                                                x - half_patch:x + half_patch]
            
            # Store original center slice values
            center_slice = patches[i, :, :, center_slice_idx].copy()
            targets[i, ..., 0] = center_slice
            orig_vals[i] = center_slice

            # For both training and validation, we need to mask pixels
            # Select pixels to mask
            n_pixels = self.patch_size * self.patch_size
            if self.validation:
                # For validation, use a regular grid
                stride = 8  # Controls density of validation masking
                y_coords, x_coords = np.meshgrid(
                    np.arange(0, self.patch_size, stride),
                    np.arange(0, self.patch_size, stride),
                    indexing='ij'
                )
                mask_y, mask_x = y_coords.flatten(), x_coords.flatten()
            else:
                # For training, random selection
                pixel_indices = self.rng.choice(n_pixels, 
                                            size=self.pixels_per_patch, 
                                            replace=False)
                mask_y, mask_x = np.unravel_index(pixel_indices, 
                                                (self.patch_size, self.patch_size))
            
            # Mark masked positions
            mask[i, mask_y, mask_x] = True
            
            # For each masked position
            for y, x in zip(mask_y, mask_x):
                # Select a random replacement position (not from masked positions)
                while True:
                    rand_y = self.rng.randint(0, self.patch_size)
                    rand_x = self.rng.randint(0, self.patch_size)
                    if not mask[i, rand_y, rand_x]:
                        break
                
                # Get the replacement value from center slice
                replacement_value = patches[i, rand_y, rand_x, center_slice_idx]
                
                # Replace the pixel in all slices (create blind spot)
                for slice_idx in range(n_slices):
                    patches[i, y, x, slice_idx] = replacement_value

        return patches, {
            'target': targets,
            'mask': mask,
            'original_values': orig_vals
        }

    def __len__(self):
        """Number of batches per epoch."""
        total_patches = self.total_slices * self.n_patches_per_slice
        return max(1, int(np.ceil(total_patches / self.batch_size)))

    def __getitem__(self, idx):
        """Get a batch of patches (thread-safe)."""
        # If we don't have enough patches for a batch, generate more
        with self.batch_lock:
            while len(self.current_batch_patches) < self.batch_size:
                scan_idx, slice_idx = self._get_next_slice_indices()
                if scan_idx is None:
                    break
                
                try:
                    # Get slice stack and extract patches
                    slice_stack = self._get_context_slices(scan_idx, slice_idx)
                    if slice_stack is not None:
                        patches, targets_dict = self._extract_patches(slice_stack)
                        
                        # Add individual patches to current batch
                        for i in range(patches.shape[0]):
                            self.current_batch_patches.append((
                                patches[i],
                                targets_dict['target'][i],
                                targets_dict['mask'][i],
                                targets_dict['original_values'][i]
                            ))
                except Exception as e:
                    print(f"Error processing slice {slice_idx} from scan {scan_idx}: {e}")
                    continue
            
            # Get up to batch_size patches
            batch_patches = []
            if self.current_batch_patches:
                n_patches = min(self.batch_size, len(self.current_batch_patches))
                batch_patches = [self.current_batch_patches.pop(0) for _ in range(n_patches)]
                
                # Pad if necessary
                while len(batch_patches) < self.batch_size:
                    batch_patches.append(batch_patches[0])
            else:
                # If we have no patches at all, create a dummy batch
                slice_stack = self._get_context_slices(0, 0)
                patches, targets_dict = self._extract_patches(slice_stack)
                batch_patches = [(
                    patches[0],
                    targets_dict['target'][0],
                    targets_dict['mask'][0],
                    targets_dict['original_values'][0]
                )] * self.batch_size
        
        # Unzip the batch data
        patches, targets, masks, orig_vals = zip(*batch_patches)
        
        # Stack and ensure float32 type
        return np.stack(patches).astype(np.float32), {
            'target': np.stack(targets).astype(np.float32),
            'mask': np.stack(masks),
            'original_values': np.stack(orig_vals).astype(np.float32)
        }

    def on_epoch_end(self):
        """Reset indices at epoch end (thread-safe)."""
        with self.slice_lock:
            self.current_scan_idx = 0
            self.current_slice_idx = 0
        with self.batch_lock:
            self.current_batch_patches = []