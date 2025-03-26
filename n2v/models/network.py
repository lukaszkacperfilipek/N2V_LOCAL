import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple, Optional, Dict, Any, Union, List, Callable
from pathlib import Path
import numpy as np
import nibabel as nib
import json
from tqdm import tqdm
import tempfile
import shutil
from ..data.preprocessing import BrainPreprocessor, PreprocessingParams

@tf.keras.utils.register_keras_serializable(package='N2VNetwork')
class N2VNetwork(Model):
    """
    Noise2Void network implementation for brain MRI denoising.
    Uses 2.5D approach with context slices and handles masked columns.
    
    Features:
    - U-Net architecture with skip connections
    - Proper handling of context slices
    - Masked loss computation for N2V training
    - Memory-efficient implementation
    
    Args:
        input_shape: Shape of input patches (height, width, n_slices)
        filters_base: Number of base filters (doubled in each layer)
        n_depth: Number of U-Net levels
        batch_norm: Whether to use batch normalization
    """
    """
    Noise2Void network implementation for brain MRI denoising.
    Uses 2.5D approach with context slices and handles masked columns.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (32, 32, 5),
        filters_base: int = 32,
        n_depth: int = 4,
        batch_norm: bool = True,
        **kwargs
    ):
        """Initialize N2VNetwork with the given parameters."""
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.filters_base = filters_base
        self.n_depth = n_depth
        self.batch_norm = batch_norm
        
        # Build model
        self.encoder_blocks = []
        self.decoder_blocks = []
        self.build_architecture()

    def build_architecture(self):
        """Build U-Net architecture."""
        # Initial convolution
        self.initial_conv = self._conv_block(self.filters_base, name="initial")
        
        # Encoder path
        for i in range(self.n_depth):
            filters = self.filters_base * (2 ** i)
            self.encoder_blocks.append(
                self._build_encoder_block(filters, name=f"encoder_{i}")
            )
        
        # Bridge
        bridge_filters = self.filters_base * (2 ** self.n_depth)
        self.bridge = self._conv_block(bridge_filters, name="bridge")
        
        # Decoder path
        for i in reversed(range(self.n_depth)):
            filters = self.filters_base * (2 ** i)
            self.decoder_blocks.append(
                self._build_decoder_block(filters, name=f"decoder_{i}")
            )
        
        # Final convolution
        self.final_conv = layers.Conv2D(1, 1, activation='linear', name="final")

    def _conv_block(self, filters: int, name: str) -> layers.Layer:
        """Create a convolution block."""
        block = tf.keras.Sequential(name=f"{name}_conv")
        
        for i in range(2):
            block.add(layers.Conv2D(filters, 3, padding='same'))
            if self.batch_norm:
                block.add(layers.BatchNormalization())
            block.add(layers.ReLU())
            
        return block

    @tf.keras.utils.register_keras_serializable()
    def _masked_mse_loss(self, y_true, y_pred, mask=None):
        """Custom MSE loss with Keras 3 compatibility."""
        if mask is None:
            return tf.keras.losses.mean_squared_error(y_true, y_pred)
        
        mask = tf.cast(mask, tf.float32)
        squared_error = tf.square(y_pred - y_true)
        masked_error = squared_error * mask
        n_masked_pixels = tf.reduce_sum(mask)
        return tf.reduce_sum(masked_error) / (n_masked_pixels + tf.keras.backend.epsilon())
    
    def compile(
        self,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
        loss: Optional[Union[str, Callable]] = None,
        metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None,
        **kwargs
    ):
        """Compile with Keras 3 compatibility."""
        if loss is None:
            loss = self._masked_mse_loss
        if metrics is None:
            metrics = ['mean_squared_error']  # Use string identifier for metric
            
        super().compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs
        )

    def _build_encoder_block(self, filters: int, name: str) -> Dict[str, layers.Layer]:
        """Create encoder block with convolutions and pooling."""
        return {
            'conv': self._conv_block(filters, name),
            'pool': layers.MaxPooling2D(2)
        }

    def _build_decoder_block(self, filters: int, name: str) -> Dict[str, layers.Layer]:
        """Create decoder block with upsampling and convolutions."""
        return {
            'upconv': layers.Conv2DTranspose(
                filters, 2, strides=2, padding='same'
            ),
            'conv': self._conv_block(filters, name)
        }

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the network.
        
        Args:
            inputs: Tensor of shape (batch_size, height, width, n_slices)
            training: Whether in training mode
            
        Returns:
            Tensor of shape (batch_size, height, width, 1)
        """
        # Initial convolution
        x = self.initial_conv(inputs)
        
        # Encoder path with skip connections
        skips = []
        for encoder in self.encoder_blocks:
            x = encoder['conv'](x)
            skips.append(x)
            x = encoder['pool'](x)
        
        # Bridge
        x = self.bridge(x)
        
        # Decoder path
        for decoder, skip in zip(self.decoder_blocks, reversed(skips)):
            x = decoder['upconv'](x)
            x = tf.concat([x, skip], axis=-1)
            x = decoder['conv'](x)
        
        # Final convolution
        return self.final_conv(x)

    def train_step(self, data: Tuple[tf.Tensor, Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        """
        Training step with improved loss calculation.
        """
        x, y_true_dict = data
        
        # Get mask and original values
        mask = y_true_dict['mask']
        target = y_true_dict['target']
        
        with tf.GradientTape() as tape:
            # Forward pass - predict entire patch
            y_pred = self(x, training=True)
            
            # Convert mask to float and expand dimensions
            mask_float = tf.cast(mask, tf.float32)
            mask_float = tf.expand_dims(mask_float, -1)
            
            # Calculate per-pixel squared error
            squared_error = tf.square(y_pred - target)
            
            # Apply mask and calculate mean loss
            masked_error = squared_error * mask_float
            n_masked_pixels = tf.reduce_sum(mask_float) + 1e-8  # Avoid division by zero
            loss = tf.reduce_sum(masked_error) / n_masked_pixels
            
            # Optional: Add regularization losses if any
            reg_loss = tf.reduce_sum(self.losses) if self.losses else 0.0
            total_loss = loss + reg_loss
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Clip gradients to prevent explosion
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics
        self.compiled_metrics.update_state(target, y_pred, sample_weight=mask_float)
        
        # Prepare results dict
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = total_loss
        
        return results

    
    def test_step(self, data: Tuple[tf.Tensor, Dict[str, tf.Tensor]]) -> Dict[str, tf.Tensor]:
        """
        Custom test step with updated metrics handling for Keras 3.
        """
        x, y_true_dict = data
        
        # Forward pass
        y_pred = self(x, training=False)
        
        # Convert mask to float and expand dimensions
        mask = tf.cast(y_true_dict['mask'], tf.float32)
        mask = tf.expand_dims(mask, -1)
        
        # Calculate masked MSE loss
        squared_error = tf.square(y_pred - y_true_dict['target'])
        masked_error = squared_error * mask
        n_masked_pixels = tf.reduce_sum(mask)
        loss = tf.reduce_sum(masked_error) / (n_masked_pixels + tf.keras.backend.epsilon())
        
        # Update metrics using new approach
        for metric in self.metrics:
            if metric.name == 'loss':
                metric.update_state(loss)
            else:
                metric.update_state(y_true_dict['target'], y_pred, sample_weight=mask)
        
        return {m.name: m.result() for m in self.metrics}

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'filters_base': self.filters_base,
            'n_depth': self.n_depth,
            'batch_norm': self.batch_norm
        })
        return config

    def get_compile_config(self):
        """Return the compilation configuration."""
        config = {
            'optimizer': tf.keras.optimizers.serialize(self.optimizer),
            'metrics': ['mean_squared_error'],  # Using string identifier instead of metric instance
            'loss': 'mse',  # Using string identifier for built-in loss
            'weighted_metrics': None,
            'run_eagerly': self.run_eagerly
        }
        return config

    def compile_from_config(self, config):
        """Compile the model from configuration."""
        optimizer = tf.keras.optimizers.deserialize(config['optimizer'])
        
        self.compile(
            optimizer=optimizer,
            loss=self._masked_mse_loss,  # Always use our custom loss
            metrics=[tf.keras.metrics.MeanSquaredError()],
            run_eagerly=config.get('run_eagerly', None)
        )

    @classmethod
    def from_config(cls, config):
        """Create model instance from configuration dictionary.
        
        Args:
            config: Dictionary containing model configuration
            
        Returns:
            N2VNetwork instance
        """
        # Extract only the parameters needed for __init__
        init_config = {
            'input_shape': config['input_shape'],
            'filters_base': config['filters_base'],
            'n_depth': config['n_depth'],
            'batch_norm': config['batch_norm']
        }
        
        # Add any remaining Keras-specific parameters
        for key in ['name', 'trainable', 'dtype']:
            if key in config:
                init_config[key] = config[key]
        
        return cls(**init_config)

    @classmethod
    def load_pretrained(cls, path: str) -> 'N2VNetwork':
        """
        Load pretrained model with Keras 3 compatibility.
        """
        path = Path(path)
        if not str(path).endswith('.keras'):
            path = path.with_suffix('.keras')
        
        if not path.exists():
            raise ValueError(f"Model file not found: {path}")

        # Use global custom object scope
        custom_objects = {
            'N2VNetwork': cls,
            '_masked_mse_loss': cls._masked_mse_loss
        }
        
        with tf.keras.utils.custom_object_scope(custom_objects):
            try:
                model = tf.keras.models.load_model(path)
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                # Fallback loading method
                model = tf.keras.models.load_model(
                    path,
                    custom_objects=custom_objects,
                    compile=False
                )
                # Recompile the model
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss=model._masked_mse_loss,
                    metrics=[tf.keras.metrics.MeanSquaredError()]
                )
        
        return model

    def save_model(self, path: str, include_optimizer: bool = True) -> None:
        """
        Save model with Keras 3 compatibility.
        """
        path = Path(path)
        if not str(path).endswith('.keras'):
            path = path.with_suffix('.keras')
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Simple save without extra parameters
        self.save(path)


    def denoise_slice(
        self,
        slice_stack: np.ndarray,
        preprocessor: Optional['BrainPreprocessor'] = None,
        overlap: int = 16,
        use_gaussian_weights: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Denoise a single slice using improved patch blending.
        Uses overlap-tile strategy with central region extraction.
        """
        if len(slice_stack.shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {slice_stack.shape}")
        
        # Handle different input formats
        if slice_stack.shape[-1] != 2*self.n_context_slices + 1:
            slice_stack = np.transpose(slice_stack, (1, 2, 0))
        slice_stack = slice_stack.astype(np.float32)
        
        # Store original range
        orig_min = slice_stack.min()
        orig_max = slice_stack.max()
        
        # Apply preprocessing if provided
        if preprocessor is not None:
            slice_stack = preprocessor.preprocess_slice_stack(slice_stack)
        
        height, width = slice_stack.shape[:2]
        patch_size = self.input_shape[0]
        
        # Use larger overlap for better blending
        overlap = patch_size // 2  # 50% overlap
        stride = patch_size - overlap
        
        # Calculate effective region size (central region of each patch)
        effective_size = patch_size - 2 * overlap
        
        # Calculate grid positions ensuring full coverage
        y_positions = list(range(0, height - patch_size + 1, stride))
        if y_positions[-1] + patch_size < height:
            y_positions.append(height - patch_size)
            
        x_positions = list(range(0, width - patch_size + 1, stride))
        if x_positions[-1] + patch_size < width:
            x_positions.append(width - patch_size)
        
        # Create accumulation arrays
        denoised = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        
        # Create blending weights for patch (high weight in center, low at edges)
        y, x = np.meshgrid(
            np.linspace(-1, 1, patch_size),
            np.linspace(-1, 1, patch_size),
            indexing='ij'
        )
        radius = np.sqrt(x*x + y*y)
        weight_patch = np.clip(1 - radius, 0, 1)
        weight_patch = np.power(weight_patch, 2)  # Sharper falloff at edges
        
        # Process patches
        patches = []
        positions = []
        
        for y in y_positions:
            for x in x_positions:
                # Extract patch with context
                patch = np.zeros((patch_size, patch_size, slice_stack.shape[-1]), dtype=np.float32)
                
                # Handle boundary conditions with reflection padding
                for i in range(slice_stack.shape[-1]):
                    patch[:, :, i] = np.pad(
                        slice_stack[
                            max(0, y):min(height, y + patch_size),
                            max(0, x):min(width, x + patch_size),
                            i
                        ],
                        [
                            (abs(min(0, y)), abs(min(0, height - (y + patch_size)))),
                            (abs(min(0, x)), abs(min(0, width - (x + patch_size))))
                        ],
                        mode='reflect'
                    )
                
                patches.append(patch)
                positions.append((y, x))
                
                if len(patches) >= batch_size:
                    # Process batch
                    batch_predictions = self.predict(np.stack(patches), verbose=0)
                    
                    # Apply predictions with blending
                    for (py, px), pred in zip(positions, batch_predictions):
                        pred = pred[..., 0]  # Remove channel dimension
                        
                        # Calculate valid region for this patch
                        y_start, y_end = py, py + patch_size
                        x_start, x_end = px, px + patch_size
                        
                        # Add weighted prediction
                        denoised[y_start:y_end, x_start:x_end] += pred * weight_patch
                        weights[y_start:y_end, x_start:x_end] += weight_patch
                    
                    patches = []
                    positions = []
        
        # Process remaining patches
        if patches:
            batch_predictions = self.predict(np.stack(patches), verbose=0)
            for (py, px), pred in zip(positions, batch_predictions):
                pred = pred[..., 0]
                y_start, y_end = py, py + patch_size
                x_start, x_end = px, px + patch_size
                denoised[y_start:y_end, x_start:x_end] += pred * weight_patch
                weights[y_start:y_end, x_start:x_end] += weight_patch
        
        # Normalize by weights
        valid_mask = weights > 1e-8
        denoised[valid_mask] /= weights[valid_mask]
        
        # Handle potential boundary artifacts
        boundary_mask = weights < 1e-8
        if np.any(boundary_mask):
            # Fill in any remaining holes using nearest neighbor
            from scipy.ndimage import distance_transform_edt
            
            dist, indices = distance_transform_edt(
                boundary_mask,
                return_indices=True
            )
            
            denoised[boundary_mask] = denoised[
                indices[0][boundary_mask],
                indices[1][boundary_mask]
            ]
        
        # Reverse preprocessing
        if preprocessor is not None:
            denoised = preprocessor.reverse_normalization(denoised)
        else:
            denoised = (denoised - denoised.min()) / (denoised.max() - denoised.min())
            denoised = denoised * (orig_max - orig_min) + orig_min
        
        return denoised

    def denoise_nifti(
            self,
            nifti_path: Union[str, Path],
            output_path: Optional[Union[str, Path]] = None,
            batch_size: int = 32,
            overlap: int = 4,  # Reduced overlap to 4 pixels (12.5% of patch size)
            use_gaussian_weights: bool = True,
            show_progress: bool = True
        ) -> nib.Nifti1Image:
            """
            Denoise an entire NIFTI file while preserving metadata.
            """
            # Load NIFTI and preserve metadata
            nifti_img = nib.load(str(nifti_path))
            header = nifti_img.header.copy()
            affine = nifti_img.affine.copy()
            data = nifti_img.get_fdata()
            
            # Create temporary directory for preprocessor cache
            import tempfile
            temp_cache_dir = tempfile.mkdtemp()
            
            try:
                # Create preprocessor for this specific file
                preprocessor = BrainPreprocessor(
                    cache_dir=temp_cache_dir,
                    params=PreprocessingParams(normalize_mode='zscore')
                )
                
                # Compute statistics for this specific file
                preprocessor.compute_scan_statistics(Path(nifti_path))
                
                # Process slices with proper progress bar
                denoised_data = np.zeros_like(data)
                
                # Create progress bar that updates in-place
                from tqdm.auto import tqdm
                pbar = tqdm(range(data.shape[2]), 
                        desc='Denoising slices',
                        leave=True,
                        position=0,
                        dynamic_ncols=True)
                
                for slice_idx in pbar:
                    # Get context slices with padding
                    start_idx = max(0, slice_idx - self.n_context_slices)
                    end_idx = min(data.shape[2], slice_idx + self.n_context_slices + 1)
                    slice_stack = data[:, :, start_idx:end_idx]
                    
                    if slice_stack.shape[2] < 2*self.n_context_slices + 1:
                        pad_width = 2*self.n_context_slices + 1 - slice_stack.shape[2]
                        pad_config = ((0,0), (0,0), 
                                    (pad_width if start_idx == 0 else 0,
                                    pad_width if end_idx == data.shape[2] else 0))
                        slice_stack = np.pad(slice_stack, pad_config, mode='reflect')
                    
                    # Denoise slice
                    denoised_slice = self.denoise_slice(
                        slice_stack,
                        preprocessor=preprocessor,
                        overlap=overlap,  # Using smaller overlap
                        batch_size=batch_size,
                        use_gaussian_weights=use_gaussian_weights
                    )
                    
                    denoised_data[:, :, slice_idx] = denoised_slice
                
                # Preserve data type from original NIFTI
                original_dtype = header.get_data_dtype()
                denoised_data = denoised_data.astype(original_dtype)
                
                # Create new NIFTI with preserved metadata
                denoised_nifti = nib.Nifti1Image(denoised_data, affine, header)
                
                # Save if requested
                if output_path is not None:
                    nib.save(denoised_nifti, str(output_path))
                
                return denoised_nifti
                
            finally:
                # Clean up temporary directory
                import shutil
                shutil.rmtree(temp_cache_dir)

    @property
    def n_context_slices(self) -> int:
        """Number of context slices on each side based on input shape."""
        return (self.input_shape[-1] - 1) // 2