import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import seaborn as sns

def get_img_dimensions(img_path):
    with Image.open(img_path) as img:
        return img.size

def calc_mask_size(pixelStr: str):
    if (pixelStr == pixelStr):
        split = pixelStr.split(" ")
        lengthP = np.array(split[1::2], dtype=int)
        return lengthP.sum()
    else:
        return 0

class AirbusDataSet:
    def __init__(self, csv_file_path, train_data_path, random_seed = None):
        self.dataframe = pd.read_csv(csv_file_path)
        self.processed_df = self.process_dataframe()
        self.train_data_path = train_data_path
        self.random_seed = random_seed if random_seed is not None else np.random.randint(0, 10000)

    def process_dataframe(self):
        df = self.dataframe
        # Add info about ship beeing present in the image
        df['has_ship'] = df['EncodedPixels'].apply(lambda x: 0 if pd.isna(x) else 1)

        # Count the number of ships in the image
        df['ship_count'] = df.groupby('ImageId')['EncodedPixels'].transform('count')

        # # Add image size info
        # df['width'], df['height'] = zip(*df['ImageId'].apply(lambda x: get_img_dimensions(os.path.join(self.train_data_path, x))))

        # # Add image size info, the values are constant for this dataset
        # df['width'] = 768
        # df['height'] = 768

        # Calculate mask size
        df['mask_size'] = df['EncodedPixels'].apply(lambda x: calc_mask_size(x))

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
                                    # 'width': 'first',
                                    # 'height': 'first',
                                    'mask_size': 'mean'
                                }).reset_index()

        # Rename column
        df = df.rename(columns={'mask_size': 'mean_mask_size'})

        return df

    def get_balanced_data(self, total_samples):
        df = self.processed_df

        # Divide mask_size into bins with equal number of samples
        num_bins_mask_size = 100
        df['bin_mean_mask_size'] = pd.qcut(df['mean_mask_size'], num_bins_mask_size, labels=False, duplicates='drop')

        # Calculate the total number of unique ship counts and mask sizes
        ship_count_total = df['ship_count'].unique()
        mask_size_total = df['bin_mean_mask_size'].unique()

        print("Total number of unique ship counts: ", len(ship_count_total))
        print("Total number of unique mask sizes: ", len(mask_size_total))

        # Calculate the number of samples per unique mask size
        samples_per_mask_size = total_samples // len(mask_size_total)
        # Create a new dataframe to store the balanced data
        balanced_df = pd.DataFrame()

        # Iterate over each unique bin_mean_mask_size
        for bin_size in df['bin_mean_mask_size'].unique():
            # Get samples_per_mask_size samples for the current bin_mean_mask_size
            samples = df[df['bin_mean_mask_size'] == bin_size].sample(samples_per_mask_size, random_state=self.random_seed)
            
            # Append the samples to the balanced dataframe
            balanced_df = balanced_df._append(samples)

        # Reset the index of the balanced dataframe
        balanced_df = balanced_df.reset_index(drop=True)

        # Return the balanced dataframe
        self.balanced_df = balanced_df


    def get_train_validation_split(self, total_samples, validation_fraction):
        self.get_balanced_data(total_samples)
        df = self.balanced_df


        # Display the number of samples in the balanced dataframe
        print("Number of samples in balanced dataframe:", len(df))



        return df


        # train_size = int(total_samples * (1 - validation_fraction))
        # validation_size = total_samples - train_size

        # train_data, validation_data = train_test_split(self.balanced_df, train_size=len(self.balanced_df), test_size=validation_fraction, random_state=self.random_seed, stratify=self.balanced_df['mean_mask_size'])

        # return train_data, validation_data

# # Example usage
# csv_file_path = 'path_to_your_csv_file.csv'

# dataset = AirbusDataSet(csv_file_path)
# train_data, validation_data = dataset.get_train_validation_split(total_samples=4, validation_fraction=0.25)

# print("Train Data:")
# print(train_data)
# print("\nValidation Data:")
# print(validation_data)
