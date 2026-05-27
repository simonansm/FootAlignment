"""Configuration for foot alignment classification."""

# Data configuration
BATCH_SIZE = 32
NUM_CLASSES = 3
NUM_EPOCHS = 1000
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-3

# Model configuration
MODEL_TYPE = "inception"  # Options: "inception", "resnet", "vit", "efficientnet"
NUM_FEATURES = 19

# Morphological features used in the models
MORPHOLOGICAL_FEATURES = [
    "gender", "side", "len_foot", "len_arch", "len_mm_med", "len_mm_lat",
    "len_latdors", "len_arch_lat", "len_arch_med", "wid_fore", "wid_heel",
    "wid_instep", "wid_meta", "size_eu", "arch_med", "arch_reg", "arch_lat",
    "arch_index", "pron_angle",
]

# Training configuration
EARLY_STOPPING_PATIENCE = 150
MAX_SAVED_MODELS = 5

# Bootstrap evaluation
N_BOOTSTRAP = 100
BOOTSTRAP_SEED = 0
CI_PERCENT = 95
