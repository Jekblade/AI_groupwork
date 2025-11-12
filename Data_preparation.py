import numpy as np
import pandas as pd
import glob
import os

def load_all_csvs(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
    files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    sensor_dfs = []
    for f in files:
        # try reading with a header first, fall back to headerless
        try:
            df = pd.read_csv(f, parse_dates=['datetime'], infer_datetime_format=True)
        except Exception:
            df = pd.read_csv(
                f, 
                header=None, 
                names=['datetime','value'], 
                parse_dates=[0], 
                infer_datetime_format=True, 
                sep=None, 
                engine='python'
            )
        
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce', infer_datetime_format=True)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        sensor_name = os.path.splitext(os.path.basename(f))[0]
        df = df[['datetime','value']].rename(columns={'value': sensor_name})
        df.set_index('datetime', inplace=True)
        sensor_dfs.append(df)

    # Merge all sensors into one combined dataframe (outer join to include all timestamps)
    combined = pd.concat(sensor_dfs, axis=1, join='outer')
    combined.sort_index(inplace=True)

    # --- Resample to 5-minute bins ---
    combined_5min = combined.resample('1T').mean()  # 5T = 5 minutes (30T = 30 minutes) CHANGE if needed

    # Keep only data starting from the first actual timestamp
    combined_5min = combined_5min[combined_5min.index >= combined.index.min()]

    # Reset index for export
    combined_5min.reset_index(inplace=True)

    return combined_5min


if __name__ == '__main__':
    combined_5min_df = load_all_csvs()

    # Export to CSV
    output_path = os.path.join(os.getcwd(), 'combined_resampled_1min.csv') # CHANGE NAME AS WELL
    combined_5min_df.to_csv(output_path, index=False)

    print(f"✅ 5-minute resampled CSV exported to: {output_path}")
    print(f"Columns: {list(combined_5min_df.columns)}")
