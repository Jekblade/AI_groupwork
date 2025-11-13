import numpy as np
import pandas as pd
import glob
import os
from datetime import timedelta

def load_all_csvs(data_dir=None):
    """
    Load all CSVs, merge by datetime, and resample to 30-minute bins.
    Rules:
      - All numeric sensors → mean within 30 min.
      - Occupancy (binary) → keep original events without averaging.
        * If multiple occupancy changes occur in one bin:
            - if within first 15 min of bin → assign to current bin
            - if after 15 min of bin → assign to next bin so we keep all data
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # --- Load all sensors ---
    sensor_dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, parse_dates=['datetime'], infer_datetime_format=True)
        except Exception:
            df = pd.read_csv(
                f, header=None, names=['datetime', 'value'],
                parse_dates=[0], infer_datetime_format=True, sep=None, engine='python'
            )
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')

        name = os.path.splitext(os.path.basename(f))[0]
        df = df[['datetime', 'value']].rename(columns={'value': name})
        df.set_index('datetime', inplace=True)
        sensor_dfs.append(df)

    combined = pd.concat(sensor_dfs, axis=1, join='outer').sort_index()

    # --- Separate occupancy ---
    if '2020-05_lgh1_presence_livingroom' in combined.columns:
        occupancy = combined['2020-05_lgh1_presence_livingroom'].dropna().copy()
        other = combined.drop(columns=['2020-05_lgh1_presence_livingroom'])
    else:
        raise ValueError("No 'occupancy' column found in data.")

    # --- Resample other sensors every 30 minutes (mean) ---
    resampled_others = other.resample('30T').mean()

    # --- Handle occupancy manually ---
    resampled_occupancy = pd.Series(index=resampled_others.index, dtype='float64')

    for t, val in occupancy.items():
        # find which 30-min bin this timestamp belongs to
        bin_start = t.floor('30T')
        delta = (t - bin_start).total_seconds() / 60  # minutes since bin start
        if delta > 15:
            # move to next bin if near the end of the interval
            bin_start = bin_start + timedelta(minutes=30)
        # assign occupancy to that bin (keep latest if conflict)
        resampled_occupancy.loc[bin_start] = val

    # Combine back into final dataset
    combined_30min = resampled_others.copy()
    combined_30min['occupancy'] = resampled_occupancy
    combined_30min.sort_index(inplace=True)

    # Reset index for saving
    combined_30min.reset_index(inplace=True)
    combined_30min.rename(columns={'index': 'datetime'}, inplace=True)

    return combined_30min


if __name__ == '__main__':
    combined_30min_df = load_all_csvs()

    output_path = os.path.join(os.getcwd(), 'combined_resampled_30min.csv')
    combined_30min_df.to_csv(output_path, index=False)

    print(f"30-minute resampled CSV exported to: {output_path}")
    print(f"Columns: {list(combined_30min_df.columns)}")
