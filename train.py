from config import *
from utils.generators import *
from utils.losses import *
from utils.model import *
from utils.utils import *

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau

# read dataset
df = pd.read_csv(TRAIN_DATASET_CSV)

# add info about ship beeing present in the image
df['has_ship'] = df['EncodedPixels'].apply(lambda x: 0 if pd.isna(x) else 1)

# add info about total number of ships at the image
df['ship_count'] = df.groupby('ImageId')['EncodedPixels'].transform('count')

# concat all EncodedPixels into AllEncodedPixels
df['AllEncodedPixels'] = df.groupby('ImageId')['EncodedPixels'].transform(
    lambda x: np.nan if x.isna().all() else ' '.join(filter(None, x))
)

# remove repeating images
df = df.drop_duplicates(subset='ImageId', keep='first')

# delete EncodedPixels column
df = df.drop(columns=['EncodedPixels'])

# reset indexes
df = df.reset_index(drop=True)

value_counts = df['ship_count'].value_counts()

balanced_df = pd.DataFrame()

# create 'balanced' dataset
for value in value_counts.index:
    subset = df[df['ship_count'] == value]
    number_samples = NUM_SAMPLES if NUM_SAMPLES < len(subset) else len(subset)
    resampled_subset = resample(subset, replace=False, n_samples=number_samples, random_state=42)
    balanced_df = pd.concat([balanced_df, resampled_subset])

# drop images without ships
balanced_df = balanced_df[balanced_df['ship_count'] > 0]


# split train data into train and validation
train_ids, validation_ids = train_test_split(
    balanced_df, test_size = 0.1, stratify = balanced_df['ship_count'])

train_df = pd.merge(balanced_df, train_ids)
validation_df = pd.merge(balanced_df, validation_ids)

print(f"train_df:\n {train_df.sample(5)}")
print(f"validation_df:\n {validation_df.sample(5)}")

train_gen = img_gen(train_df)


# callbacks
tensorboard = TensorBoard(log_dir='logs')

earlystopping = EarlyStopping(
    monitor="val_dice_score", 
    mode="max", 
    patience=15) 

checkpoint = ModelCheckpoint(
    filepath='model.{epoch:02d}-{val_loss:.2f}.weights.h5',
    monitor='val_dice_score',
    verbose=1,
    mode='max',
    save_weights_only = True)

reduceLR = ReduceLROnPlateau(
    monitor='val_dice_score',
    factor=0.2, 
    patience=3, 
    verbose=1,
    mode='max',
    min_delta=0.0001,
    cooldown=2,
    min_lr=1e-6)

callbacks = [tensorboard, earlystopping, checkpoint, reduceLR]





STEP_COUNT = train_df.shape[0]//BATCH_SIZE


model_fit_gen = augmentation_generator(img_gen(train_df, BATCH_SIZE, PATCH_SIZE))

# create validation set
# TODO move to the begining of the file
VALIDATION_SET_SIZE = (balanced_df.shape[0] - train_df.shape[0])
validation_x, validation_y = next(img_gen(validation_df, VALIDATION_SET_SIZE, PATCH_SIZE))



print(f"The size of training set - {train_df.shape[0]}")
print(f"The size of validation set - {VALIDATION_SET_SIZE}")
print(f"Steps/Epoch - {STEP_COUNT}")

# Create a MirroredStrategy.
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope.
with strategy.scope():

    model = unet(INPUT_DATA_DIM, optimizer='adam', loss=dice_bce_loss, metrics=[dice_score])
    model.summary()
    # Train the model on all available devices.
    model.fit(
        model_fit_gen,
        steps_per_epoch=STEP_COUNT,
        epochs=NB_EPOCHS,
        validation_data=(validation_x, validation_y),
        callbacks=callbacks)
