"""Train on combined public and private data."""

import os
import pandas as pd
from model_definitions import define_model_cnn
from train import train_nn_models


def main():
    # load data
    data_folder = "data"
    solar_train = pd.read_csv(os.path.join(data_folder, "public", "solar_wind.csv"))
    dst_train = pd.read_csv(os.path.join(data_folder, "public", "dst_labels.csv"))
    sunspots_train = pd.read_csv(os.path.join(data_folder, "public", "sunspots.csv"))
    solar_test = pd.read_csv(os.path.join(data_folder, "private", "solar_wind.csv"))
    dst_test = pd.read_csv(os.path.join(data_folder, "private", "dst_labels.csv"))
    sunspots_test = pd.read_csv(os.path.join(data_folder, "private", "sunspots.csv"))

    # combine test and train
    solar = pd.concat([solar_train, solar_test], axis=1)
    sunspots = pd.concat([sunspots_train, sunspots_test], axis=1)
    dst = pd.concat([dst_train, dst_test], axis=1)

    # train and save models
    output_folder = os.path.join("trained_models", "cnn", "all_data")
    os.makedirs(output_folder, exist_ok=True)
    train_nn_models(solar, sunspots, dst, define_model_cnn, 5, output_folder)


if __name__ == "__main__":
    main()
