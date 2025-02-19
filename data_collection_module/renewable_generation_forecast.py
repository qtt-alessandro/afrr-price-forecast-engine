import pandas as pd
import requests

def get_raw_forecasts_data(start, end):
    """Fetch raw forecast data from the energy data service API."""
    start_utc = start.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M')
    end_utc = end.tz_convert('UTC').strftime('%Y-%m-%dT%H:%M')
    
    url = "https://api.energidataservice.dk/dataset/Forecasts_Hour"
    params = {
        'offset': 0,
        'start': start_utc,
        'end': end_utc,
        'sort': 'HourUTC DESC'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    df = pd.DataFrame(data['records'])
    df['HourUTC'] = pd.to_datetime(df['HourUTC'], utc=True)  # Add UTC awareness here
    return df

def transform_forecast_types(df):
    """
    Transform forecast types into separate columns for each forecast horizon.
    Includes DayAhead and Current forecasts.
    """
    forecast_columns = [
        'ForecastDayAhead',
        'ForecastCurrent'
    ]
    
    dfs = []
    for fcst_col in forecast_columns:
        pivot_df = df.pivot(
            index=['HourUTC', 'PriceArea'],
            columns='ForecastType',
            values=fcst_col
        )
        
        # Remove ForecastType name from columns
        pivot_df.columns.name = None
        
        suffix = fcst_col.replace('Forecast', '').lower()
        pivot_df = pivot_df.rename(columns={
            'Offshore Wind': f'wind_offshore_{suffix}',
            'Onshore Wind': f'wind_onshore_{suffix}',
            'Solar': f'solar_{suffix}'
        })
        
        dfs.append(pivot_df)
    
    result = pd.concat(dfs, axis=1)
    result = result.reset_index()
    result = result.loc[:, ~result.columns.duplicated()]
    
    return result

def get_wind_solar_data(start, end, price_area="DK_2"):
    """
    Fetch and transform energy data, filtering for specified price area.
    
    Parameters:
    -----------
    start : pd.Timestamp
        Start datetime with timezone
    end : pd.Timestamp
        End datetime with timezone
    price_area : str
        Price area to filter for ("DK_1" or "DK_2")
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with wind and solar forecasts
    """
    raw_df = get_raw_forecasts_data(start, end)
    processed_df = transform_forecast_types(raw_df)
    
    # Filter for price area
    area_code = "DK1" if price_area == "DK_1" else "DK2"
    processed_df = processed_df[processed_df["PriceArea"] == area_code]
    
    # Set index to HourUTC
    processed_df = processed_df.set_index("HourUTC")
    processed_df.index.name = "time_utc"
    processed_df.index = pd.to_datetime(processed_df.index)
    return processed_df
