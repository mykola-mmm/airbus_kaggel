import os
import random
import argparse
from tensorflow import keras
from utils.losses import *
from utils.model import *
from utils.utils import *


def parse_args():
   parser = argparse.ArgumentParser(description='Train a U-Net model for ship detection.')
   parser.add_argument('--model_path', type=str, help='Path to the trained model')
   parser.add_argument('--test_data_path', type=str, help='Path to the test data directory')
   parser.add_argument('--num_test_images', type=int, default=100, help='Number of test images to be predicted')
   parser.add_argument('--image_names', nargs='+', type=str, help='List of image names in test folder to be predicted')
   parser.add_argument('--patch_size', type=int, default=768, help='Size of the patches used for model training, should be the same value as used while the model training')

   return parser.parse_args()


if __name__ == "__main__":
   args = parse_args()
   MODEL_PATH = args.model_path
   TEST_DATA_PATH = args.test_data_path
   NUM_TEST_IMAGES = args.num_test_images
   IMAGE_NAMES = args.image_names
   PATCH_SIZE = args.patch_size

   model = keras.models.load_model(MODEL_PATH, compile=False)

   if IMAGE_NAMES:
      test_imgs = IMAGE_NAMES
   else:
      file_names = os.listdir(TEST_DATA_PATH)
      test_imgs = random.sample(file_names, NUM_TEST_IMAGES)

   for img in test_imgs:
      visualise_prediction(model, PATCH_SIZE, TEST_DATA_PATH, img)
