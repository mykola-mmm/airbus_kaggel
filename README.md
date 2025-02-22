# airbus_kaggel
This repository, `airbus_kaggel`, contains code for training and running a U-Net model. The purpose of this model is to solve the image segmentation problem of ship detection from satellite imagery. The U-Net architecture is widely used in computer vision tasks and has shown great performance in segmenting objects from images. By utilizing this code, you can train and run a U-Net model specifically designed for ship detection, enabling accurate identification of ships in satellite images.

## Data Analysis
Please, see the `notebooks/eda.ipynb`.

## U-net model description
The model is defined in `utils/model.py` file. It is typical U-net model consisting from the following layers:
- **Input Layer**: Adds Gaussian noise and batch normalization.
- **Encoder**:
  - 4 convolution blocks with filter number doubling each block.
  - Each block: Conv2D (ReLU, He normal), Dropout, MaxPooling2D.
- **Bottleneck**:
  - 2 Conv2D layers and Dropout.
- **Decoder**:
  - 4 upsampling blocks with filter number halving each block.
  - Each block: Conv2DTranspose, concatenate with encoder, Conv2D (ReLU), Dropout.
- **Output Layer**: Conv2D with 1 filter and sigmoid activation function.

## Preparing the Environment
To get started, follow these steps to prepare the environment:

1. Manually download the dataset from [here](https://www.kaggle.com/competitions/airbus-ship-detection/data) and unzip it into the directory `airbus-ship-detection`.
2. Install the required dependencies by running the following command:
    ```
    pip install -r ./requirements
    ```
3. To run the model training, execute the following command:
    ```bash
    python ./train_model.py 
    ```
    During the invocation of the training script the following arguments could be specified:
    ```bash
    --batch_size: Batch size for training
    --epochs: Number of training epochs
    --max_number_of_samples: Max number of samples
    --validation_test_size: Validation test set size, expected value - float from [0: 1]
    --dropout: Dropout coefficient
    --gaussian_noise: Standard deviation of Gaussian noise
    --num_filters: Number of filters for convolutional layers
    --dataset_path: Path to the dataset
    --csv_file: Path to the CSV file
    --patch_size: Size to which training images will be resized
    --model_dir: Directory where models will be saved
    --learning_rate: Learning rate
    --optimizer: Optimizer type, suppoerted values are 'adam', 'nadam' and 'rms'
    --loss_function: Loss function for training, supported values are 'dice_loss', 'bce_loss', 'dice_bce_loss', and 'ioe_loss'
    --early_stopping_patience: Patience for early stopping
    --reduce_lr_factor: Factor by which the learning rate will be reduced
    --reduce_lr_patience: Number of epochs with no improvement after which learning rate will be reduced
    --reduce_lr_min_delta: Minimum change in the monitored quantity to qualify as an improvement
    --reduce_lr_cooldown: Number of epochs to wait before resuming normal operation after lr has been reduced
    --reduce_lr_min_lr: Lower bound on the learning rate
    ```
    
    After each epoch the model is saved into the `model` directory in format `model.epoch{epoch:02d}-val_dice_score{val_dice_score:.3f}.keras`, so after the training it would be possible to choose the model with the smallest validation loss
4. For testing the model, execute the following command, it is important to specify the patch size (should be the same as used for model training) and .keras model path:
    ```bash
    python ./test_model.py --model_path ./models/model_256x256.epoch71-val_dice_score0.650.keras --num_test_images=100 --test_data_path ./airbus-ship-detection/test_v2 --patch_size=256
    ```
    All arguments available for the test script:
    ```bash
    --model_path: Path to the trained model
    --test_data_path: Path to the test data directory
    --num_test_images: Number of test images to be predicted
    --image_names: List of image names in test folder to be predicted
    --patch_size: Size of the patches used for model training, should be the same value as used while the model training
    ```
    In order to select for what images the mask would be generated - use `--image_names` that is followed by the image names seperated with space, example:
    ```bash
    python ./test_model.py --model_path ./models/model_256x256.epoch71-val_dice_score0.650.keras --image_names img1.jpg img2.jpg img3.jpg --test_data_path ./airbus-ship-detection/test_v2 --patch_size=256
    ```

## Results
Results can be found in `notebooks/test-model.ipynb`. The final model performs reasonably well in detecting ships, but there are many false positives in images containing clouds, land, or structures (piers, bridges, etc.). To reduce false positives, it is feasible to implement another model to first predict the presence of ships in the image. If no ships are detected, an empty mask is generated. If ships are detected, the second model performs segmentation to create the ship mask.
