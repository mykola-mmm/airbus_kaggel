from config import *
from tensorflow import keras
from utils.losses import *
from utils.model import *
from utils.utils import *

print(f"MODEL_TO_TEST_PATH - {MODEL_TO_TEST_PATH}")
model = keras.models.load_model(MODEL_TO_TEST_PATH, custom_objects={'dice_bce_loss': dice_bce_loss, "dice_score": dice_score})
test_imgs = ['8e6b39bb1.jpg','45d43380c.jpg','2868e5640.jpg','a00886458.jpg','0c2848844.jpg','d9b95022a.jpg','ba9c3d11a.jpg','3566fb758.jpg','abb672b82.jpg','acb7dd8d2.jpg', '30e126c21.jpg']

# for img in test_imgs:
#     prediction = generate_prediction(model, TEST_IMG_DIR, img)
#     fig.add_subplot(rows, columns, 1)

visualise_prediction(model, TEST_IMG_DIR, '8e6b39bb1.jpg')