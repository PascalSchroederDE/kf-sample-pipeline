import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import logging

def read_file(path):
    f = open(path, "r")
    content = f.read()
    f.close()
    return content

def write_file(path, content):
    f = open(path, "w")
    f.write(content)
    f.close()

def load_data(path):
    return pd.read_csv(path)

def split_label_and_img(df):
    images = df.drop([col for col in df.columns if 'pixel' not in col ], axis='columns')
    labels = df.drop([col for col in df.columns if 'pixel' in col ], axis='columns')

    return images, labels

def main():
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument('--dataset_location' , type=str, help='Location of dataset to be used')
    parser.add_argument('--test_size', type=float, help='Size of test set')
    parser.add_argument('--random_state', type=int, help='Random state of train-test-split')
    parser.add_argument('--output_train_img' , type=str, help='Target location of dataset to be generated')
    parser.add_argument('--output_train_label' , type=str, help='Target location of dataset to be generated')
    parser.add_argument('--output_test_img' , type=str, help='Target location of dataset to be generated')
    parser.add_argument('--output_test_label' , type=str, help='Target location of dataset to be generated')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Loading data...")
    df = load_data(read_file(args.dataset_location))

    image_df, label_df = split_label_and_img(df)

    images_train, images_test, labels_train, labels_test = train_test_split(image_df, label_df.values, test_size=args.test_size, random_state=args.random_state)

    logging.info("Writing resulting dataframes to csvs")
    images_train.to_csv(args.output_train_img, index=False)
    labels_train.to_csv(args.output_train_label, index=False)
    images_test.to_csv(args.output_test_img, index=False)
    labels_test.to_csv(args.output_test_label, index=False)

    write_file("/trainimg.txt", args.output_train_img)
    write_file("/trainlabel.txt", args.output_train_label)
    write_file("/testimg.txt", args.output_test_img)
    write_file("/testlabel.txt", args.output_test_label)