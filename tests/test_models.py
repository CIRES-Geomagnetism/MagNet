import pytest
import pandas as pd
import os
from nn_functions import prepare_data


@pytest.fixture
def test_data():
    data_folder = os.path.join("..", "test_data")
    solar_train = pd.read_csv(os.path.join(data_folder, "public", "solar_wind.csv"))
    dst_train = pd.read_csv(os.path.join(data_folder, "public", "dst_labels.csv"))
    sunspots_train = pd.read_csv(os.path.join(data_folder, "public", "sunspots.csv"))
    return solar_train, sunspots_train, dst_train


def test_no_nulls_in_prepared_data(test_data, tmpdir):
    solar, sunspots, dst = test_data
    df, train_cols = prepare_data(solar, sunspots, dst, output_folder=tmpdir)
    assert df[train_cols + ["target", "target_shift"]].notnull().all().all()
