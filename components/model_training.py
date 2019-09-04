import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import argparse
import logging


def load_data(path):
    return pd.read_csv(path)

def load_model(path):
    return keras.models.load_model(path)

def prepare_image_shape(imageset, shape_height, shape_width):
    return np.array([img.reshape(shape_height,shape_width) for img in imageset])

def main():
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument('--input_train_img' , type=str, help='Target location of dataset to be generated')
    parser.add_argument('--input_train_label' , type=str, help='Target location of dataset to be generated')
    parser.add_argument('--input_shape_height', type=int, help='Heigt of input images')
    parser.add_argument('--input_shape_width', type=int, help='Width of input images')
    parser.add_argument('--model_location', type=str, help='Created model location')
    parser.add_argument('--epochs', type=int, help='Number of epochs to be executed')
    parser.add_argument('--output', type=str, help='Output location for trained model')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Loading data...")
    train_img = load_data(args.input_train_img)
    train_label = load_data(args.input_train_label)

    logging.info("Preparing images...")
    train_img = prepare_image_shape(train_img.values, args.input_shape_height, args.input_shape_width)

    logging.info("Loading model...")
    model = load_model(args.model_location)

    logging.info("Training model...")
    model.fit(train_img, train_label, epochs=args.epochs)

    logging.info("Saving model weights...")
    model.save(args.output)