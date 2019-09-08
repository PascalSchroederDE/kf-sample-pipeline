import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
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

def load_model(path):
    return keras.models.load_model(path)

def prepare_image_shape(imageset, shape_height, shape_width):
    return np.array([img.reshape(shape_height,shape_width) for img in imageset])

def store_loss_acc(file, loss, acc):
    with open (file,'w') as f:
        f.write("Loss: {}".format(loss))
        f.write("Accuracy: {}".format(acc))

def main():
    parser = argparse.ArgumentParser(description="Feature engineering")
    parser.add_argument('--input_test_img' , type=str, help='Target location of dataset to be generated')
    parser.add_argument('--input_test_label' , type=str, help='Target location of dataset to be generated')
    parser.add_argument('--input_shape_height', type=int, help='Heigt of input images')
    parser.add_argument('--input_shape_width', type=int, help='Width of input images')
    parser.add_argument('--model_location', type=str, help='Created model location')
    parser.add_argument('--output', type=str, help='Output location for trained model')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info("Loading data...")
    test_img = load_data(read_file(args.input_test_img))
    test_label = load_data(read_file(args.input_test_label))

    logging.info("Preparing images...")
    test_img = prepare_image_shape(test_img.values, args.input_shape_height, args.input_shape_width)

    logging.info("Loading model...")
    model = load_model(read_file(args.model_location))

    logging.info("Evaluate model...")
    loss, acc = model.evaluate(test_img, test_label)
    logging.info("Evaluation loss: {}".format(loss))
    logging.info("Evaluation accuracy: {}".format(acc))

    logging.info("Saving loss and accuracy...")
    store_loss_acc(args.output, loss, acc)

    write_file("/trained_model.txt", args.output)