"""Process old data to produce files similar to competition format."""

import os
import numpy as np
import pandas as pd
from spacepy import coordinates as coords
from spacepy.time import Ticktock
import datetime as dt

# process solar wind data
solar = pd.read_csv(
    os.path.join("data", "old", "NASA_OMNI_Bowshock_SW_Dst_1963_1997_Hourly.csv")
)
solar["period"] = "train_d"
year = solar["Decimal year"].astype(int)
year_timestamp = pd.to_datetime(
    pd.DataFrame({"year": year, "month": np.ones(len(year)), "day": np.ones(len(year))})
)
year_plus_one_timestamp = pd.to_datetime(
    pd.DataFrame(
        {"year": year + 1, "month": np.ones(len(year)), "day": np.ones(len(year))}
    )
)
hours_in_year = (year_plus_one_timestamp - year_timestamp).dt.total_seconds() // 3600
hours = np.round((solar["Decimal year"] - year) * hours_in_year, 0)
solar["timestamp"] = year_timestamp + pd.to_timedelta(hours, unit="hour")
solar["timedelta"] = solar["timestamp"] - solar["timestamp"].min()

# fill missing values with nan
for c in solar.columns:
    solar.loc[solar[c] == 999.9, c] = np.nan

# add gsm coords
old_coords = coords.Coords(solar[["bx_gse", "by_gse", "bz_gse"]].values, "GSE", "car")
unix_time = (solar["timestamp"] - dt.datetime(1970, 1, 1)).dt.total_seconds()
old_coords.ticks = Ticktock(unix_time.values, "UNX")
new_coords = old_coords.convert("GSM", "car")
solar[["bx_gsm", "by_gsm", "bz_gsm"]] = new_coords.data

# rename columns and save
solar.rename(columns={"Dst": "dst", "source_imf": "source"}, inplace=True)
output_cols = [
    "period",
    "timedelta",
    "bx_gse",
    "by_gse",
    "bz_gse",
    "bx_gsm",
    "by_gsm",
    "bz_gsm",
    "density",
    "speed",
    "source",
]
solar[output_cols].to_csv(os.path.join("data", "old", "solar_wind.csv"), index=False)

# save dst separately
solar[["period", "timedelta", "dst"]].to_csv(
    os.path.join("data", "old", "dst_labels.csv")
)


# process sunspot data, from Royal Observatory of Belgium
# https://wwwbis.sidc.be/silso/datafiles#total
ssn = pd.read_csv(
    os.path.join("data", "old", "SN_ms_tot_V2.0.csv"), sep=";", header=None
)
# decimal year is the value for the middle of the month
ssn.columns = ["year", "month", "decimal_year", "ssn", "std_dev", "num_obs", "quality"]
year = ssn["decimal_year"].astype(int)
year_timestamp = pd.to_datetime(
    pd.DataFrame({"year": year, "month": np.ones(len(year)), "day": np.ones(len(year))})
)
year_plus_one_timestamp = pd.to_datetime(
    pd.DataFrame(
        {"year": year + 1, "month": np.ones(len(year)), "day": np.ones(len(year))}
    )
)
days_in_year = (year_plus_one_timestamp - year_timestamp).dt.total_seconds() // (
    3600 * 24
)
days = np.round((ssn["decimal_year"] - year) * days_in_year, 0)
ssn["timestamp"] = year_timestamp + pd.to_timedelta(days, unit="day")
# calculate timedelta relative to same start time as solar wind data
ssn["timedelta"] = ssn["timestamp"] - solar["timestamp"].min()
ssn = ssn.loc[ssn["timestamp"] >= solar["timestamp"].min()]
ssn["period"] = "train_d"
ssn.rename(columns={"ssn": "smoothed_ssn"}, inplace=True)
output_cols = ["period", "timedelta", "smoothed_ssn"]
ssn[output_cols].to_csv(os.path.join("data", "old", "sunspots.csv"), index=False)
