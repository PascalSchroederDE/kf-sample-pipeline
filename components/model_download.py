import pandas as pd
import tensorflow as tf
from tensorflow import keras
import argparse
import logging

def write_file(path, content):
    f = open(path, "w")
    f.write(content)
    f.close()
    
def get_activation_func(shorthand):
    return {
        "relu": tf.nn,
        "softmax": tf.nn.softmax
    }[shorthand]

def load_data(path):
    return pd.read_csv(path)

def download_model(input_shape):
    return tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                               include_top=False,
                                               weights='imagenet')

def main():
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument('--input_shape_height', type=int, help='Heigt of input images')
    parser.add_argument('--input_shape_width', type=int, help='Width of input images')
    parser.add_argument('--output', type=str, help='Output location for model to be build')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Downloading model...")
    model = download_model((args.input_shape_height, args.input_shape_width))

    logging.info("Saving model...")
    model.save(args.output)

    write_file("/model.txt", args.output)

if __name__ == '__main__':
    main()
    