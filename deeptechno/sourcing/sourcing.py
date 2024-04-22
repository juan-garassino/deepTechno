import tensorflow as tf
import pathlib
import os
import csv
import os
import tensorflow as tf
from deepTechno.manager.manager import Manager
import pandas as pd
import random

def download_maestro_dataset(data_dir):
    """
    Download the MAESTRO dataset if it doesn't exist locally.

    Args:
    - data_dir (str): Directory to save the dataset.

    Returns:
    - str: Path to the downloaded dataset.
    """
    # Check if the data directory exists
    if not os.path.exists(data_dir):
        # Download and extract the dataset
        dataset_path = tf.keras.utils.get_file(
            'maestro-v2.0.0-midi.zip',
            origin='https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0-midi.zip',
            extract=True,
            cache_dir='.',
            cache_subdir='dataset'
        )
        return dataset_path
    else:
        print("Dataset already exists in", data_dir)
        dataset_path = os.path.join(data_dir, "maestro-v2.0.0-midi")
        if os.path.exists(dataset_path):
            return dataset_path
        else:
            print("Error: Dataset not found in the specified directory:", data_dir)
            return None

def find_midi_files(dataset_folder, csv_file):
    """
    Find all MIDI files within subfolders of the dataset folder and save their directories to a CSV file.

    Args:
    - dataset_folder (str): Path to the dataset folder.
    - csv_file (str): Path to the CSV file to save the directories.

    Returns:
    - list: List of directories of the found MIDI files.
    """
    # List to store the directories of MIDI files
    midi_files = []

    # Recursively search for MIDI files within subfolders
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".midi") or file.endswith(".mid"):
                midi_path = os.path.join(root, file)
                midi_files.append(midi_path)

    # Create the directories to the CSV file if it doesn't exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    # Save the directories to a CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["midi_file"])  # Write header
        for midi_file in midi_files:
            writer.writerow([midi_file])

    return midi_files, csv_file

def split_csv(csv_file, train_csv, val_csv, test_csv, train_p=0.6, val_p=0.2, test_p=0.2):
    """
    Split the CSV file into three separate CSV files for train, validation, and test sets.

    Args:
    - csv_file (str): Path to the input CSV file.
    - train_csv (str): Path to save the CSV file for the train set.
    - val_csv (str): Path to save the CSV file for the validation set.
    - test_csv (str): Path to save the CSV file for the test set.
    - train_p (float): Proportion of data to include in the train set.
    - val_p (float): Proportion of data to include in the validation set.
    - test_p (float): Proportion of data to include in the test set.
    """
    # Read the original CSV file
    df = pd.read_csv(csv_file)

    # Shuffle the rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Calculate the number of samples for each split
    num_samples = len(df)
    num_train = int(num_samples * train_p)
    num_val = int(num_samples * val_p)
    num_test = num_samples - num_train - num_val

    # Split the data into train, validation, and test sets
    train_set = df.iloc[:num_train]
    val_set = df.iloc[num_train:num_train + num_val]
    test_set = df.iloc[num_train + num_val:]

    # Save the split datasets to CSV files
    train_set.to_csv(train_csv, index=False)
    val_set.to_csv(val_csv, index=False)
    test_set.to_csv(test_csv, index=False)

    print("Split data into train, validation, and test sets successfully.")

if __name__ == "__main__":
    
    data_dir = "./dataset/maestro-v2.0.0"  # Change this to your desired directory
    
    dataset_path = download_maestro_dataset(data_dir)
    
    print("Dataset path:", dataset_path)

    csv_file = os.path.join(data_dir, "maestro.csv")  # Construct the CSV file path
    
    print("CSV file path:", csv_file)

    print("First few rows of the CSV file:")
    
    midi_files, csv_file = find_midi_files(data_dir, csv_file)

    split_csv(csv_file, "train.csv", "val.csv", "test.csv", train_p=0.6, val_p=0.2, test_p=0.2)

    print("first",  midi_files[1])