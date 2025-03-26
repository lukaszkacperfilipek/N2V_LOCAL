# Training parameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
INPUT_SHAPE = (200, 256, 5)
VALIDATION_SPLIT = 0.2
NUM_CONTEXT_SLICES = 2

# Data parameters
DATA_DIR = '/home/luks/NIO/ALL_NIFTI'
CACHE_DIR = 'preprocessing_cache'

# Model parameters
FILTERS_INITIAL = 64
DEPTH = 4
KERNEL_SIZE = 3
