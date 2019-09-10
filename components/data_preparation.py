import pandas as pd
import numpy as np
import argparse
import logging

CATEGORIES = ['top', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boot']

def write_file(path, content):
    f = open(path, "w")
    f.write(content)
    f.close()

def load_data(path):
    return pd.read_csv(path)

def unify_datatypes(dataset):
    for col in dataset.select_dtypes("int64"):
        dataset[col] = dataset[col].astype("float64")
    return dataset

def remove_mis_values(dataset):
    dataset.replace('', np.nan, inplace=True)
    dataset = dataset.dropna()
    return dataset

def remove_faulty_values(dataset):
    dataset = dataset[dataset.label.isin(CATEGORIES)]
    return dataset

def normalize_values(dataset):
    for col in (col for col in dataset.columns if col not in ["label"]):
        dataset[col] = dataset[col] / 255.0
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument('--dataset_location' , type=str, help='Location of dataset to be used')
    parser.add_argument('--output' , type=str, help='Target location of dataset to be generated')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Loading data...")
    df = load_data(args.dataset_location)

    logging.info("Unify dataset (Integers to Float)...")
    df = unify_datatypes(df)

    logging.info("Remove rows with missing values...")
    df = remove_mis_values(df)

    logging.info("Remove rows with faulty values...")
    df = remove_faulty_values(df)

    logging.info("Normalize values (every cell should have a value between 0 and 1)")
    df = normalize_values(df)

    logging.info("Writing resulting dataframe to csv")
    df.to_csv(args.output, index=False)

    write_file("/prepdf_output.txt", args.output)

if __name__ == '__main__':
    main()