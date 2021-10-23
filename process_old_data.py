"""Process old data to produce files similar to competition format."""

import os
import numpy as np
import pandas as pd

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
solar.to_csv(os.path.join("data", "old", "solar_wind.csv"))

# save dst separately
solar.rename(columns={"Dst": "dst"}, inplace=True)
solar[["period", "timedelta", "dst"]].to_csv(
    os.path.join("data", "old", "dst_labels.csv")
)

# fill missing values with nan
for c in solar.columns:
    solar.loc[solar[c] == 999.9, c] = np.nan

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
ssn["period"] = "train_d"
ssn.to_csv(os.path.join("data", "old", "sunspots.csv"))
