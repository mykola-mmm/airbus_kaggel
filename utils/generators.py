from config import *
from .utils import create_mask

import numpy as np

from PIL import Image
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def img_gen(input_df, batch_size = BATCH_SIZE, patch_size = PATCH_SIZE):
    # shuffle the dataset
    input_df = input_df.sample(frac=1, random_state=42).reset_index(drop=True)
    out_rgb = []
    out_mask = []
    while True:
        for index, row in input_df.iterrows():
            rgb_path = os.path.join(TRAIN_IMG_DIR, row.ImageId)
            rgb = Image.open(rgb_path)
            rgb = np.array(rgb)/255.0
            rgb = resize(rgb, (patch_size, patch_size), anti_aliasing=True)
            mask = create_mask(row.AllEncodedPixels)
            mask = resize(mask, (patch_size, patch_size), anti_aliasing=True)
#             the next line is 'kostyl' to address min/max mask values beeing equal to 0.0/3.051851e-05
            mask = np.where(mask > 0, 1, 0)
            mask = np.expand_dims(mask, -1)
            
            for i in range(0, rgb.shape[0], patch_size):
                for j in range(0, rgb.shape[1], patch_size):
                    single_mask_patch = mask[i:i+patch_size, j:j+patch_size]
                    if (single_mask_patch.max()):
                        single_rgb_patch = rgb[i:i+patch_size, j:j+patch_size]
                        out_rgb += [single_rgb_patch]
                        out_mask += [single_mask_patch]
                    if len(out_rgb)>=batch_size:
                        yield np.stack(out_rgb, 0), np.stack(out_mask, 0).astype(np.float32)
                        out_rgb, out_mask=[], []

# arhuments for augmentation image generator
data_gen_args = dict(rotation_range = 90,
                       horizontal_flip = True,
                       vertical_flip = True,
                       data_format = 'channels_last')

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# data augmentation generator
def augmentation_generator(input_gen, seed = None):
    random_seed = np.random.randint(0, 10000)
    for input_x, input_y in input_gen:
        augmented_x = image_datagen.flow(
            input_x*255,
            batch_size=input_x.shape[0],
            seed=random_seed
        )

        augmented_y = mask_datagen.flow(
            input_y,
            batch_size=input_y.shape[0],
            seed=random_seed
        )

        yield next(augmented_x)/255.0, next(augmented_y)
