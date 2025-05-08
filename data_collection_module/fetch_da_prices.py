import os
import glob
import pandas as pd
import requests
import polars as pl


def get_day_ahead_prices(start, end, price_area="DK_2"):
    
    base_url = "https://api.energidataservice.dk/dataset/Elspotprices"
    
    start_str = start.strftime('%Y-%m-%dT%H:%M')
    end_str = end.strftime('%Y-%m-%dT%H:%M')
    
    params = {
        "offset": 0,
        "start": start_str,
        "end": end_str,
        "sort": "HourUTC ASC"
    }
    
    response = requests.get(base_url, params=params)
    response.raise_for_status() 
    
    data = response.json()
    
    
    df = pd.json_normalize(data['records'])
    
    if price_area == "DK_2":
        df = df[df["PriceArea"] == "DK2"]
    elif price_area == "DK_1":
        df = df[df["PriceArea"] == "DK1"]
    
    df = df[['HourUTC', 'SpotPriceEUR']].rename(columns={"HourUTC":"time_utc", "SpotPriceEUR":"da_price"})
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True)
    df = df.set_index("time_utc")
    
    return df





def load_and_process_boiler_data(folder_path):
    # Get all CSV files
    csv_files_list = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Expected columns and mapping
    EXPECTED_COLS = ['UTC', 'Heat load forecast [MW]', 'Spotprice Prediction [DKK/MWh]']
    column_rename_map = {
        'UTC': 'time_utc',
        'Heat load forecast [MW]': 'heat_load_forecast',
        'Spotprice Prediction [DKK/MWh]': 'da_preds'
    }
    
    dfs = []
    for csv_file in csv_files_list:
        try:
            # Read and clean each file
            df_temp = pd.read_csv(csv_file)
            df_temp.columns = df_temp.columns.str.strip()
            
            # Check if expected columns exist
            missing_cols = [col for col in EXPECTED_COLS if col not in df_temp.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} in file {csv_file}")
                continue
                
            dfs.append(df_temp[EXPECTED_COLS])
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    if not dfs:
        raise ValueError("No valid CSV files found with expected columns")
    
    # Combine all DataFrames
    df = pd.concat(dfs, ignore_index=True)
    
    # Process data
    df["UTC"] = pd.to_datetime(df["UTC"].str.strip(), utc=True)
    df = df.rename(columns=column_rename_map)
    
    # Handle numeric conversions
    df['da_preds'] = pd.to_numeric(df['da_preds'].ffill(), errors='coerce') * 0.13
    df['heat_load_forecast'] = pd.to_numeric(df['heat_load_forecast'].ffill(), errors='coerce')
    
    # Set index and handle duplicates
    df = df.set_index("time_utc")
    df = df[~df.index.duplicated(keep='first')]
    
    # Sort index for proper slicing
    df = df.sort_index()
    
    return df




