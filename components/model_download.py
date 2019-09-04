import pandas as pd
import tensorflow as tf
from tensorflow import keras
import argparse
import logging

def get_activation_func(shorthand):
    return {
        "relu": tf.nn,
        "softmax": tf.nn.softmax
    }[shorthand]

def load_data(path):
    return pd.read_csv(path)

def build_model(input_shape):
    return tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
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
    model = build_model((args.input_shape_height, args.input_shape_width), args.num_units, args.num_outputs, get_activation_func(args.activation_l2), get_activation_func(args.activation_l3))

    logging.info("Saving model...")
    model.save(args.output)
    