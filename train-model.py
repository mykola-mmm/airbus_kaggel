import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.generators import *
from utils.losses import *
from utils.model import *
from utils.utils import *
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


# Function to do final processing of the dataset and balance it
def prepare_balanced_dataset():
    # Read preprocessed dataset
    df = pd.read_csv(CSV_FILE)

    # Concatenate all EncodedPixels into AllEncodedPixels
    df['AllEncodedPixels'] = df.groupby('ImageId')['EncodedPixels'].transform(
        lambda x: np.nan if x.isna().all() else ' '.join(filter(None, x))
    )

    # Drop EncodedPixels column
    df = df.drop(columns=['EncodedPixels'])

    # Grouping by 'ImageId' and aggregating columns
    df = df.groupby('ImageId').agg({
        'AllEncodedPixels': 'first',
        'has_ship': 'first',
        'ship_count': 'first',
        'width': 'first',
        'height': 'first',
        'mask_size': 'max',
        'mask_size_percentage': 'max'
    }).reset_index()

    # Divide mask_size into bins with equal number of samples
    num_bins_mask_size = 100
    df['mask_size_bins'] = pd.qcut(df['mask_size'], num_bins_mask_size, labels=False, duplicates='drop')

    # Calculate the total number of unique ship counts and mask sizes
    ship_count_total = df['ship_count'].unique()
    mask_size_total = df['mask_size_bins'].unique()

    num_bins = len(ship_count_total) * len(mask_size_total)
    samples_per_bin = MAX_NUMBER_OF_SAMPLES // num_bins

    samples = []

    # Sample data from each bin combination
    for i in ship_count_total:
        for j in mask_size_total:
            bin_df = df[(df['ship_count'] == i) & (df['mask_size_bins'] == j)]
            if len(bin_df) >= samples_per_bin:
                samples.append(bin_df.sample(n=samples_per_bin, replace=False))
            else:
                samples.append(bin_df.sample(n=len(bin_df), replace=False))

    balanced_df = pd.concat(samples).reset_index(drop=True)
    return balanced_df

# Function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Train a U-Net model for ship detection.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--max_number_of_samples', type=int, default=5000, help='Max number of samples')
    parser.add_argument('--validation_test_size', type=float, default=0.2, help='Validation test set size, expected value - float from [0: 1]')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout coefficient')
    parser.add_argument('--gaussian_noise', type=float, default=0.1, help='Standard deviation of Gaussian noise')
    parser.add_argument('--num_filters', type=int, default=16, help='Number of filters for convolutional layers')
    parser.add_argument('--dataset_path', type=str, default='airbus-ship-detection/train_v2', help='Path to the dataset')
    parser.add_argument('--csv_file', type=str, default='airbus-ship-detection/dataset.csv', help='Path to the CSV file')
    parser.add_argument('--patch_size', type=int, default=768, help='Size to which training images will be resized')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory where models will be saved')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type')
    parser.add_argument('--loss_function', type=str, default='dice_loss', help='Loss function for training')
    return parser.parse_args()

if __name__ == '__main__':
    # Create variables for each argument
    args = parse_args()
    BATCH_SIZE = args.batch_size
    PATCH_SIZE = args.patch_size
    INPUT_DATA_DIM = (PATCH_SIZE, PATCH_SIZE, 3)
    EPOCHS = args.epochs
    MAX_NUMBER_OF_SAMPLES = args.max_number_of_samples
    VALIDATION_TEST_SIZE = args.validation_test_size
    DROPOUT = args.dropout
    GAUSSIAN_NOISE = args.gaussian_noise
    DATASET_PATH = args.dataset_path
    CSV_FILE = args.csv_file
    LOSS_FUNCTION = args.loss_function
    MODEL_DIR = args.model_dir
    NUM_FILTERS = args.num_filters
    LEARNING_RATE = args.learning_rate
    MIN_LEARNING_RATE = args.min_learning_rate
    OPTIMIZER = args.optimizer

    # Create a MirroredStrategy for distributed training
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope
    with strategy.scope():

        # Check if the chosen optimizer is available
        if OPTIMIZER == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        elif OPTIMIZER == 'nadam':
            optimizer = tf.keras.optimizers.Nadam(learning_rate=LEARNING_RATE)
        elif OPTIMIZER == 'rms':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
        else:
            raise ValueError(f"Invalid optimizer type. Available optimizers are 'adam', 'nadam', and 'rms'.")

        # Check if the chosen loss function is available
        if LOSS_FUNCTION == 'dice_loss':
            loss_function = dice_loss
        elif LOSS_FUNCTION == 'bce_loss':
            loss_function = bce_loss
        elif LOSS_FUNCTION == 'dice_bce_loss':
            loss_function = dice_bce_loss
        elif LOSS_FUNCTION == 'ioe_loss':
            loss_function = iou_loss
        else:
            raise ValueError(f"Invalid loss function. Available loss functions are 'dice_loss', 'bce_loss', 'dice_bce_loss', and 'ioe_loss'.")

        balanced_df = prepare_balanced_dataset()
        train_ids, validation_ids = train_test_split(balanced_df, test_size=VALIDATION_TEST_SIZE, stratify=balanced_df['ship_count'])

        train_df = pd.merge(balanced_df, train_ids)
        validation_df = pd.merge(balanced_df, validation_ids)

        # Create an augmented generator for model fitting
        model_fit_gen = augmentation_generator(img_gen(train_df, BATCH_SIZE, PATCH_SIZE, train_img_dir=DATASET_PATH))

        # Create a validation set
        validation_test_size = (balanced_df.shape[0] - train_df.shape[0])
        validation_x, validation_y = next(img_gen(validation_df, validation_test_size, PATCH_SIZE, train_img_dir=DATASET_PATH))

        # Calculate the number of steps per epoch
        STEP_COUNT = train_df.shape[0] // BATCH_SIZE

        # Define callbacks for training
        tensorboard = TensorBoard(log_dir='logs')

        earlystopping = EarlyStopping(
            monitor="val_dice_score",
            mode="max",
            patience=15)

        # Check if WEIGHTS_DIR exists, if not create it
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        # MODEL_FILE = 'model_{PATCH_SIZE}x{PATCH_SIZE}.epoch{epoch:02d}-val_dice_score{val_dice_score:.3f}.keras'

        MODEL_FILE = 'model.epoch{epoch:02d}-val_dice_score{val_dice_score:.3f}.keras'

        MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILE)

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
            min_lr=MIN_LEARNING_RATE)

        callbacks = [tensorboard, earlystopping, checkpoint, reduceLR]

        # Create a MirroredStrategy for distributed training
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        print(f"The size of the training set: {train_df.shape[0]}")
        print(f"The size of the validation set: {validation_test_size}")
        print(f"Steps/Epoch: {STEP_COUNT}")

        # Create the model using the unet function
        model = unet(INPUT_DATA_DIM, optimizer=optimizer, loss=loss_function, metrics=['accuracy', dice_score], gaussian_noise=GAUSSIAN_NOISE, dropout=DROPOUT, num_filters=NUM_FILTERS)
        model.summary()
        
        # Train the model on all available devices
        model.fit(model_fit_gen,
                  steps_per_epoch=STEP_COUNT,
                  epochs=EPOCHS,
                  validation_data=(validation_x, validation_y),
                  callbacks=callbacks)
