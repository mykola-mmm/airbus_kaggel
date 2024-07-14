import os

# model configs
NUM_SAMPLES = 3000  # not a total number of samples, but max number of samples with the same 'ship_count' 
GAUSSIAN_NOISE = 0.1
NB_EPOCHS = 50
BATCH_SIZE = 64
PATCH_SIZE = 256
INPUT_DATA_DIM = (PATCH_SIZE, PATCH_SIZE, 3)

# env_configs
BASE_DIR = 'airbus-ship-detection'
TEST_IMG_DIR = os.path.join(BASE_DIR,'test_v2')
TRAIN_IMG_DIR = os.path.join(BASE_DIR,'train_v2')
TRAIN_DATASET_CSV = os.path.join(BASE_DIR,'train_ship_segmentations_v2.csv')

# WEIGHTS_DIR = 'weights'
# WEIGHTS_FILE = 'model.{epoch:02d}-{val_loss:.2f}.weights.h5'
# WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, WEIGHTS_FILE)

MODEL_DIR = 'model'
MODEL_FILE = 'model.{epoch:02d}-{val_loss:.2f}.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

# model testing configs
MODEL_TO_TEST = ''
MODEL_TO_TEST_PATH = os.path.join(MODEL_DIR, MODEL_TO_TEST)
