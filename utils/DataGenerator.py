import tensorflow as tf
import numpy as np
import pandas as pd
import cv2  # For image processing
import os

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe, image_dir, batch_size=32, default_image_size=(768, 768), training_image_size=(768, 768), shuffle=True):
        """
        Initialization
        :param dataframe: pandas dataframe with columns 'ImageId' and 'AllEncodedPixels'
        :param image_dir: directory where images are stored
        :param batch_size: size of the batches
        :param default_image_size: size of the original images
        :param training_image_size: size to which images and masks are resized for training
        :param shuffle: whether to shuffle the data after each epoch
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.default_image_size = default_image_size
        self.training_image_size = training_image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of the batch
        :return: tuple (X, y) of input and output
        """
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.dataframe.iloc[indices]
        X, y = self.__data_generation(batch_data)
        return X, y

    def on_epoch_end(self):
        """
        Updates indices after each epoch
        """
        self.indices = np.arange(len(self.dataframe))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_data):
        """
        Generates data containing batch_size samples
        :param batch_data: dataframe containing batch data
        :return: tuple of numpy arrays (X, y)
        """
        X = np.empty((self.batch_size, *self.training_image_size, 3), dtype=np.float32)
        y = np.empty((self.batch_size, *self.training_image_size, 1), dtype=np.float32)
        for i, (_, row) in enumerate(batch_data.iterrows()):
            image_path = os.path.join(self.image_dir, row['ImageId'])
            image = self.__load_image(image_path)
            mask = self.__rle_decode(row['AllEncodedPixels'])
            X[i,] = image
            y[i,] = mask

        return X, y

    def __load_image(self, image_path):
        """
        Load and preprocess the image
        :param image_path: path to the image file
        :return: preprocessed image
        """
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.training_image_size)
        image = image / 255.0  # Normalize to [0, 1]
        return image

    def __rle_decode(self, rle):
        """
        Decode run-length encoded mask
        :param rle: run-length encoded mask
        :return: decoded mask resized to self.training_image_size
        """
        if pd.isnull(rle):
            return np.zeros(shape=(*self.training_image_size, 1), dtype=np.float32)
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(self.default_image_size[0] * self.default_image_size[1], dtype=np.float32)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        mask = img.reshape(self.default_image_size).T
        
        # Resize mask to the training image size
        mask_resized = cv2.resize(mask, self.training_image_size)
        mask_resized = mask_resized[:, :, np.newaxis]
        return mask_resized

# Usage example:
# df = pd.read_csv('path/to/your/data.csv')
# image_dir 
