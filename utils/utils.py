import os
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt

#utility functions
def rle_to_mask(starts, lengths, height, width):
    # Create an empty array of zeros of shape (height, width)
    mask = np.zeros(height * width, dtype=np.uint8)
    
    # For each start and length, set the corresponding values in the mask to 1
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    
    # Reshape the mask into the desired dimensions
    mask = mask.reshape((height, width))
    mask = mask.T
    return mask

def create_mask(mask_array, width, height):
    masks = np.zeros((width, height), dtype=np.int16)
    # if element == element:
    if isinstance(mask_array, str):
        split = mask_array.split()
        startP, lengthP = [np.array(x, dtype=int) for x in (split[::2], split[1::2])]
        masks += (rle_to_mask(startP, lengthP, width, height))
    return masks

def generate_prediction(model, patch_size, img_dir, img_name):
    img = os.path.join(img_dir, img_name)
    img = Image.open(img)
    img = np.array(img)
    img = resize(img, (patch_size, patch_size), anti_aliasing=True)
    img = tf.expand_dims(img, axis=0)
    pred = model.predict(img)
    print(f"prediction shape - {pred.shape}")
    return pred, img

def visualise_prediction(model, patch_size, img_dir, img_name):
    pred, img = generate_prediction(model, patch_size, img_dir, img_name)
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img[0])
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(pred[0])
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.show()