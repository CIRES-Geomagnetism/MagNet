"""Make a small dataset for testing."""

import os
import datetime as dt
import pandas as pd

os.makedirs(os.path.join("test_data", "public"), exist_ok=True)
os.makedirs(os.path.join("test_data", "private"), exist_ok=True)

for df_name in ["dst_labels", "solar_wind", "sunspots"]:
    for folder in ["public", "private"]:
        df = pd.read_csv(os.path.join("data", folder, f"{df_name}.csv"))
        df["timedelta"] = pd.to_timedelta(df["timedelta"])
        ind = df["timedelta"] <= dt.timedelta(days=28)
        df.loc[ind].to_csv(
            os.path.join("test_data", folder, f"{df_name}.csv"),
            header=True,
            index=False,
        )
