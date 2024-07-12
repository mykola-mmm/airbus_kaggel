import os
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


# Read dataset
df = pd.read_csv(TRAIN_DATASET_CSV)

# Add info about whether a ship is present in the image
df['has_ship'] = df['EncodedPixels'].apply(lambda x: 0 if pd.isna(x) else 1)

# Add info about the total number of ships in the image
df['ship_count'] = df.groupby('ImageId')['EncodedPixels'].transform('count')

# Concatenate all EncodedPixels into AllEncodedPixels
df['AllEncodedPixels'] = df.groupby('ImageId')['EncodedPixels'].transform(
    lambda x: np.nan if x.isna().all() else ' '.join(filter(None, x))
)

# Remove duplicate images
df = df.drop_duplicates(subset='ImageId', keep='first')

# Delete EncodedPixels column
df = df.drop(columns=['EncodedPixels'])

# Reset indexes
df = df.reset_index(drop=True)

# Create a DataFrame to store the balanced dataset
balanced_df = pd.DataFrame()

# Create a balanced dataset
value_counts = df['ship_count'].value_counts()
for value in value_counts.index:
    subset = df[df['ship_count'] == value]
    number_samples = NUM_SAMPLES if NUM_SAMPLES < len(subset) else len(subset)
    resampled_subset = resample(subset, replace=False, n_samples=number_samples, random_state=42)
    balanced_df = pd.concat([balanced_df, resampled_subset])

# Drop images without ships
balanced_df = balanced_df[balanced_df['ship_count'] > 0]

# Split the balanced dataset into train and validation sets
train_ids, validation_ids = train_test_split(
    balanced_df, test_size=0.1, stratify=balanced_df['ship_count'])

train_df = pd.merge(balanced_df, train_ids)
validation_df = pd.merge(balanced_df, validation_ids)

print(f"train_df:\n {train_df.sample(5)}")
print(f"validation_df:\n {validation_df.sample(5)}")

# Create a generator for training data
train_gen = img_gen(train_df)

# Define callbacks for training
tensorboard = TensorBoard(log_dir='logs')

earlystopping = EarlyStopping(
    monitor="val_dice_score",
    mode="max",
    patience=15)

# Check if WEIGHTS_DIR exists, if not create it
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_path = MODEL_PATH
checkpoint = ModelCheckpoint(
    filepath=model_path,
    monitor='val_dice_score',
    verbose=1,
    mode='max',
    save_weights_only=False)

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

# Calculate the number of steps per epoch
STEP_COUNT = train_df.shape[0] // BATCH_SIZE

# Create an augmented generator for model fitting
model_fit_gen = augmentation_generator(img_gen(train_df, BATCH_SIZE, PATCH_SIZE))

# Create a validation set
VALIDATION_SET_SIZE = (balanced_df.shape[0] - train_df.shape[0])
validation_x, validation_y = next(img_gen(validation_df, VALIDATION_SET_SIZE, PATCH_SIZE))

print(f"The size of the training set: {train_df.shape[0]}")
print(f"The size of the validation set: {VALIDATION_SET_SIZE}")
print(f"Steps/Epoch: {STEP_COUNT}")

# Create a MirroredStrategy for distributed training
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# Open a strategy scope
with strategy.scope():
    # Create the model using the unet function
    model = unet(INPUT_DATA_DIM, optimizer='adam', loss=dice_bce_loss, metrics=[dice_score])
    model.summary()
    
    # Train the model on all available devices
    loss_history = [model.fit(
                    model_fit_gen,
                    steps_per_epoch=STEP_COUNT,
                    epochs=NB_EPOCHS,
                    validation_data=(validation_x, validation_y),
                    callbacks=callbacks)]
