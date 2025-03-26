import numpy as np
from typing import Optional, Tuple, NamedTuple
from scipy.ndimage import rotate, zoom

class AugmentationParams(NamedTuple):
    """Parameters for a single augmentation run."""
    do_flip: bool
    rotation_angle: float
    brightness_factor: float
    contrast_factor: float
    noise_magnitude: float

class SliceAugmenter:
    """
    Handles augmentation of brain MRI slices for Noise2Void training.
    Optimized for augmenting center slices and their context consistently.
    """
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-10, 10),
        brightness_range: Tuple[float, float] = (0.9, 1.1),
        contrast_range: Tuple[float, float] = (0.9, 1.1),
        noise_factor: float = 0.05,
        seed: Optional[int] = None
    ):
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_factor = noise_factor
        self.rng = np.random.RandomState(seed)

    def get_random_params(self) -> AugmentationParams:
        """Generate random augmentation parameters."""
        return AugmentationParams(
            do_flip=self.rng.random() < 0.5,
            rotation_angle=self.rng.uniform(*self.rotation_range),
            brightness_factor=self.rng.uniform(*self.brightness_range),
            contrast_factor=self.rng.uniform(*self.contrast_range),
            noise_magnitude=self.rng.random() * self.noise_factor
        )

    def augment_context_slices(
        self,
        slices: np.ndarray,
        params: Optional[AugmentationParams] = None
    ) -> np.ndarray:
        """
        Augment a stack of context slices consistently.
        
        Args:
            slices: Array of shape (n_slices, height, width)
            params: Optional pre-generated parameters
            
        Returns:
            Augmented slices with same shape as input
        """
        if params is None:
            params = self.get_random_params()

        # Convert to float32 if needed
        if slices.dtype != np.float32:
            slices = slices.astype(np.float32)

        # Store original range
        orig_min, orig_max = slices.min(), slices.max()
        
        # Normalize to [0, 1] for consistent augmentation
        slices = (slices - orig_min) / (orig_max - orig_min + 1e-8)

        # Apply augmentations consistently
        if params.do_flip:
            slices = slices[:, :, ::-1]  # Horizontal flip

        if params.rotation_angle != 0:
            rotated = np.zeros_like(slices)
            for i in range(len(slices)):
                rotated[i] = rotate(
                    slices[i],
                    params.rotation_angle,
                    reshape=False,
                    mode='reflect',
                    order=1
                )
            slices = rotated

        if params.brightness_factor != 1:
            slices = slices * params.brightness_factor

        if params.contrast_factor != 1:
            mean = slices.mean()
            slices = (slices - mean) * params.contrast_factor + mean

        if params.noise_magnitude > 0:
            sigma = params.noise_magnitude
            noise = self.rng.normal(0, sigma, slices.shape)
            slices = slices + noise

        # Clip and restore original range
        slices = np.clip(slices, 0, 1)
        slices = slices * (orig_max - orig_min) + orig_min

        return slices.astype(np.float32)