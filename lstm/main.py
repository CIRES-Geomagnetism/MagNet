import pandas as pd


def read_data(path:str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    dst = pd.read_csv(path + "dst_labels.csv")
    solar_wind = pd.read_csv(path + "solar_wind.csv")
    sunspots = pd.read_csv(path + "sunspots.csv")

    return dst, solar_wind, sunspots

def get_training_dataframe(path:str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    dst, solar_wind, sunspots = read_data(path)

    # Data indexing according to the timestamps
    dst.timedelta = pd.to_timedelta(dst.timedelta)
    dst.set_index(["period", "timedelta"], inplace=True)

    sunspots.timedelta = pd.to_timedelta(sunspots.timedelta)
    sunspots.set_index(["period", "timedelta"], inplace=True)

    solar_wind.timedelta = pd.to_timedelta(solar_wind.timedelta)
    solar_wind.set_index(["period", "timedelta"], inplace=True)

    return dst, solar_wind, sunspots

def main():


    data_folder = "data/public"
    dst, solar_wind, sunspots = get_training_dataframe(data_folder)

    data_config = {
        "timesteps": 128,
        "batch_size": 128,
    }




if __name__ == "__main__":

    main()

