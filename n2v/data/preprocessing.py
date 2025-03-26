from pathlib import Path
import numpy as np
import nibabel as nib
from typing import Optional, Dict, NamedTuple, List, Tuple
import json
import h5py
from dataclasses import dataclass

@dataclass
class ScanStats:
    """Statistics for a single scan"""
    mean: float
    std: float
    min_val: float
    max_val: float
    percentiles: Dict[int, float]

class PreprocessingParams(NamedTuple):
    """Parameters for preprocessing"""
    normalize_mode: str = 'zscore'  # 'zscore' or 'minmax'
    target_range: Tuple[float, float] = (-1, 1)
    cache_preprocessed: bool = True

class BrainPreprocessor:
    """
    Preprocessor for brain MRI data with focus on 2.5D processing.
    Handles both individual slices and slice stacks with context.
    
    Features:
    - Supports z-score and min-max normalization
    - Caches preprocessed data and statistics
    - Handles 2.5D context slices consistently
    - Memory-efficient processing
    """
    def __init__(
        self,
        cache_dir: str,
        params: PreprocessingParams = PreprocessingParams(),
        force_recompute: bool = False
    ):
        self.cache_dir = Path(cache_dir)
        self.params = params
        self.force_recompute = force_recompute
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._stats_cache: Dict[str, ScanStats] = {}
        self._load_stats_cache()

    def _load_stats_cache(self):
        """Load previously computed statistics from cache."""
        cache_file = self.cache_dir / 'scan_stats.json'
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                for scan_id, stats in cached_data.items():
                    self._stats_cache[scan_id] = ScanStats(**stats)

    def _save_stats_cache(self):
        """Save computed statistics to cache."""
        cache_file = self.cache_dir / 'scan_stats.json'
        with open(cache_file, 'w') as f:
            cache_dict = {
                scan_id: {
                    'mean': stats.mean,
                    'std': stats.std,
                    'min_val': stats.min_val,
                    'max_val': stats.max_val,
                    'percentiles': stats.percentiles
                }
                for scan_id, stats in self._stats_cache.items()
            }
            json.dump(cache_dict, f)

    def compute_scan_statistics(self, file_path: Path) -> ScanStats:
        """
        Compute or retrieve scan statistics.
        
        Args:
            file_path: Path to the .nii file
            
        Returns:
            ScanStats object with computed statistics
        """
        scan_id = file_path.stem
        
        if not self.force_recompute and scan_id in self._stats_cache:
            return self._stats_cache[scan_id]
        
        img = nib.load(str(file_path))
        data = img.get_fdata()
        
        # Compute statistics
        stats = ScanStats(
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            min_val=float(np.min(data)),
            max_val=float(np.max(data)),
            percentiles={
                p: float(np.percentile(data, p))
                for p in [1, 99]  # Using 1st and 99th percentiles for robustness
            }
        )
        
        self._stats_cache[scan_id] = stats
        self._save_stats_cache()
        
        return stats

    def normalize_data(
        self,
        data: np.ndarray,
        stats: ScanStats
    ) -> np.ndarray:
        """
        Normalize data based on scan statistics.
        
        Args:
            data: Input data array
            stats: ScanStats object for the scan
            
        Returns:
            Normalized data array
        """
        if self.params.normalize_mode == 'zscore':
            normalized = (data - stats.mean) / (stats.std + 1e-8)
        else:  # minmax
            normalized = (data - stats.min_val) / (
                stats.max_val - stats.min_val + 1e-8
            )
        
        # Scale to target range
        if self.params.target_range != (-1, 1):
            range_min, range_max = self.params.target_range
            normalized = normalized * (range_max - range_min) + range_min
        
        return normalized.astype(np.float32)

    def preprocess_slice_stack(
        self,
        slice_stack: np.ndarray,
        file_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Preprocess a stack of slices (including context slices).
        
        Args:
            slice_stack: Array of shape (n_slices, height, width)
            file_path: Optional path to the original .nii file for statistics
            
        Returns:
            Preprocessed slice stack with same shape
        """
        # If file_path is not provided, normalize based on stack statistics
        if file_path is None:
            stats = ScanStats(
                mean=float(np.mean(slice_stack)),
                std=float(np.std(slice_stack)),
                min_val=float(np.min(slice_stack)),
                max_val=float(np.max(slice_stack)),
                percentiles={
                    p: float(np.percentile(slice_stack, p))
                    for p in [1, 99]
                }
            )
        else:
            stats = self.compute_scan_statistics(file_path)
        
        # Normalize all slices consistently
        normalized = self.normalize_data(slice_stack, stats)
        
        return normalized

    def get_preprocessed_data(
        self,
        file_path: Path,
        slice_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Get preprocessed data from cache or compute it.
        
        Args:
            file_path: Path to the .nii file
            slice_indices: Optional list of slice indices to load
            
        Returns:
            Preprocessed data array
        """
        if not self.params.cache_preprocessed:
            return self._preprocess_file(file_path, slice_indices)
        
        cache_file = self.cache_dir / f'{file_path.stem}_preprocessed.h5'
        
        if not self.force_recompute and cache_file.exists():
            with h5py.File(cache_file, 'r') as f:
                if slice_indices is None:
                    return f['data'][:]
                return f['data'][slice_indices]
        
        # Preprocess and cache
        processed = self._preprocess_file(file_path, slice_indices)
        
        if self.params.cache_preprocessed:
            with h5py.File(cache_file, 'w') as f:
                f.create_dataset('data', data=processed)
        
        return processed

    def _preprocess_file(
        self,
        file_path: Path,
        slice_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """Helper method to preprocess a single file."""
        img = nib.load(str(file_path))
        stats = self.compute_scan_statistics(file_path)
        
        if slice_indices is None:
            data = img.get_fdata()
            return self.normalize_data(data, stats)
        
        # Load and process only requested slices
        data = img.slicer[..., slice_indices].get_fdata()
        return self.normalize_data(data, stats)
    
    def reverse_normalization(
            self,
            normalized_data: np.ndarray,
            stats: Optional[ScanStats] = None
        ) -> np.ndarray:
            """
            Reverse the normalization process to get back to original value range.
            
            Args:
                normalized_data: Normalized data array
                stats: Optional ScanStats object. If None, uses last computed stats
                
            Returns:
                Data array in original value range
            """
            # If no stats provided, use the last computed stats
            if stats is None:
                if not self._stats_cache:
                    raise ValueError("No statistics available for denormalization")
                # Use the most recently computed stats
                stats = list(self._stats_cache.values())[-1]
            
            # First, rescale from target range if needed
            if self.params.normalize_mode == 'zscore':
                # For z-score normalization
                denormalized = normalized_data * stats.std + stats.mean
            else:  # minmax
                # For min-max normalization
                if self.params.target_range != (-1, 1):
                    # First rescale to [-1, 1]
                    range_min, range_max = self.params.target_range
                    normalized_data = (normalized_data - range_min) / (range_max - range_min) * 2 - 1
                
                # Then rescale to original range
                denormalized = (normalized_data + 1) / 2 * (stats.max_val - stats.min_val) + stats.min_val
            
            return denormalized.astype(np.float32)