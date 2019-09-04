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

def build_model(input_shape, num_units, num_outputs, activation_l2, activation_l3):
    return keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(num_units, activation=activation_l2),
        keras.layers.Dense(num_outputs, activation=activation_l3)
    ])

def main():
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument('--input_shape_height', type=int, help='Heigt of input images')
    parser.add_argument('--input_shape_width', type=int, help='Width of input images')
    parser.add_argument('--num_units', type=int, help='Number of nodes in neural net')
    parser.add_argument('--num_outputs', type=int, help='Number of output nodes in neural ned')
    parser.add_argument('--activation_l2', type=str, help='Activation function for layer 2 as String')
    parser.add_argument('--activation_l3', type=str, help='Activation function for layer 3 as String')
    parser.add_argument('--optimizer', type=str, help='Optimizer function to improve model')
    parser.add_argument('--loss', type=str, help='Loss function which should be minimized')
    parser.add_argument('--metrics', type=str, help='Metrics with which should be measured')
    parser.add_argument('--output', type=str, help='Output location for model to be build')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Building model...")
    model = build_model((args.input_shape_height, args.input_shape_width), args.num_units, args.num_outputs, get_activation_func(args.activation_l2), get_activation_func(args.activation_l3))

    logging.info("Compile model...")
    model.compile(optimizer=args.optimizer,
        loss=args.loss,
        metrics=[args.metrics])

    logging.info("Saving model...")
    model.save(args.output)