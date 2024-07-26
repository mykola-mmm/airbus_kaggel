import os
import random
from tensorflow import keras
from utils.losses import *
from utils.model import *
from utils.utils import *
import argparse





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

   # Load the trained model
   model = keras.models.load_model(MODEL_PATH, compile=False)
   # model = keras.models.load_model(MODEL_PATH, custom_objects={'dice_bce_loss': dice_bce_loss, "dice_score": dice_score})

   # model = keras.models.load_model(MODEL_PATH)
   # Handpicked images for nice results :)
   handpicked_test_imgs = ['ba6fe8a25.jpg','8dd7d6368.jpg','534e10e90.jpg','711be1fce.jpg','8e6b39bb1.jpg','45d43380c.jpg','2868e5640.jpg','a00886458.jpg','0c2848844.jpg','d9b95022a.jpg','ba9c3d11a.jpg','3566fb758.jpg','abb672b82.jpg','acb7dd8d2.jpg', '30e126c21.jpg']

   # choose 100 random images from test directory for unbiased results
   file_names = os.listdir(TEST_DATA_PATH)
   test_imgs = random.sample(file_names, 20)


   # for i in handpicked_test_imgs:
   #    visualise_prediction(model, 256, TEST_DATA_PATH, i) 

   for img in test_imgs:
      visualise_prediction(model, 256, TEST_DATA_PATH, img)

# # Handpicked images for nice results :)
# handpicked_test_imgs = ['ba6fe8a25.jpg','8dd7d6368.jpg','534e10e90.jpg','711be1fce.jpg','8e6b39bb1.jpg','45d43380c.jpg','2868e5640.jpg','a00886458.jpg','0c2848844.jpg','d9b95022a.jpg','ba9c3d11a.jpg','3566fb758.jpg','abb672b82.jpg','acb7dd8d2.jpg', '30e126c21.jpg']

# # choose 100 random images from test directory for unbiased results
# file_names = os.listdir(TEST_IMG_DIR)
# test_imgs = random.sample(file_names, 33)

# print(f"MODEL_TO_TEST_PATH - {MODEL_TO_TEST_PATH}")

# model = keras.models.load_model(MODEL_TO_TEST_PATH, custom_objects={'dice_bce_loss': dice_bce_loss, "dice_score": dice_score})

# for i in handpicked_test_imgs:
#    visualise_prediction(model, TEST_IMG_DIR, i) 

# for img in test_imgs:
#     visualise_prediction(model, TEST_IMG_DIR, img)

