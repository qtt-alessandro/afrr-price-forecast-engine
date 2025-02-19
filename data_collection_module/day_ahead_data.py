import pandas as pd
import requests

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
