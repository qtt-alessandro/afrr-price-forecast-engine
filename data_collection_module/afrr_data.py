import pandas as pd
import requests

def get_afrr_data(start, end, price_area="DK_2"):
    base_url = "https://api.energidataservice.dk/dataset/AfrrReservesNordic"
    
    # Convert timestamps to simple string format YYYY-MM-DDTHH:00
    params = {
        'offset': 0,
        'start': start.strftime('%Y-%m-%dT%H:%M'),
        'end': end.strftime('%Y-%m-%dT%H:%M'),
        'sort': 'HourUTC DESC'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        df = pd.DataFrame(data['records'])
        df['HourUTC'] = pd.to_datetime(df['HourUTC'], utc=True)
        df.set_index('HourUTC', inplace=True)
        
        if price_area == "DK_2":
            df = df[df["PriceArea"] == "DK2"]
        elif price_area == "DK_1":
            df = df[df["PriceArea"] == "DK1"]
            
        df = df.drop(columns=["PriceArea", "HourDK", "aFRR_DownCapPriceDKK", "aFRR_UpCapPriceDKK"])
        df.index = df.index.rename("time_utc")
        df = df.sort_index(ascending=True)
        
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None