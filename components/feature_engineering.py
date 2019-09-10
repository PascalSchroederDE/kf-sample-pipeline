import pandas as pd
import argparse
import logging

def write_file(path, content):
    f = open(path, "w")
    f.write(content)
    f.close()

def load_data(path):
    return pd.read_csv(path)

def one_hot_encoding(dataset):
    one_hot = pd.get_dummies(dataset['label'])
    dataset = dataset.drop('label',axis = 1)
    dataset = dataset.join(one_hot) 
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument('--dataset_location' , type=str, help='Location of dataset to be used')
    parser.add_argument('--output' , type=str, help='Target location of dataset to be generated')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Loading data...")
    df = load_data(args.dataset_location)

    logging.info("Applying one hot encoding...")
    df = one_hot_encoding(df)    

    logging.info("Writing resulting dataframe to csv")
    df.to_csv(args.output, index=False)

    write_file("/findf_output.txt", args.output)

if __name__ == '__main__':
    main()