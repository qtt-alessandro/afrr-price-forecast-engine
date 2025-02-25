import os
import polars as pl 
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from entsoe import EntsoePandasClient
from renewable_generation_forecast import get_wind_solar_data
from afrr_data import get_afrr_data
from power_demand_forecast import get_load_forecasts
from day_ahead_data import get_day_ahead_prices

load_dotenv(override=True)  
entsoe_key = os.getenv('ENTSOE_API_KEY')

client = EntsoePandasClient(api_key=entsoe_key)
country_code = 'DK_1'
start = pd.Timestamp('20240101', tz='Europe/Copenhagen')
end = pd.Timestamp('20250228', tz='Europe/Copenhagen')

renewable_forecasts = get_wind_solar_data(start, end, price_area=country_code)
afrr_data = get_afrr_data(start, end, price_area=country_code)
power_demand_forecast = get_load_forecasts(client, country_code, start, end)
da_ahead = get_day_ahead_prices(start, end, country_code)
data = renewable_forecasts.join([afrr_data, power_demand_forecast, da_ahead]).dropna()
data["prod_unbalance"] = data[["wind_offshore_dayahead", "wind_onshore_dayahead", "solar_dayahead"]].sum(axis=1) - data["load_forecasts"]

df = data[['wind_offshore_dayahead', 'wind_onshore_dayahead',
       'solar_dayahead','aFRR_DownCapPriceEUR', 'aFRR_UpCapPriceEUR', 'load_forecasts',
       'da_price', 'prod_unbalance']]

df.to_parquet("./data/afrr_price.parquet")
print(df)