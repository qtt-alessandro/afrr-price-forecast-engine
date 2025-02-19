import pandas as pd 

def get_load_forecasts(client, country_code, start, end):

    load_forecasts = pd.DataFrame(
        client.query_load_forecast(country_code, start=start, end=end)
    ).rename(columns={"Forecasted Load": "load_forecasts"})
    
    load_forecasts.index.name = "time_utc"
    load_forecasts.index = load_forecasts.index.tz_convert('UTC')
    
    return load_forecasts




#day_ahead_price = pd.DataFrame(client.query_day_ahead_prices(country_code, start, end)).rename(columns={0: "da_price"})
#wind_solar_forecasts = pd.DataFrame(client.query_wind_and_solar_forecast(country_code, start=start, end=end, psr_type=None)).rename(columns={"Solar":"solar", "Wind Offshore":"wind_solar", "Wind Onshore":"wind_offshore"})
#generation_forecasts = pd.DataFrame(client.query_generation_forecast(country_code, start=start, end=end).rename("generation_forecast"))
#load_forecasts = pd.DataFrame(client.query_load_forecast(country_code, start=start, end=end)).rename(columns={"Forecasted Load":"load_forecasts"})

#data = pd.concat([day_ahead_price, 
#           wind_solar_forecasts, 
#           generation_forecasts, 
#           load_forecasts], 
#          axis=1).dropna()

#data.index = data.index.rename("time_utc")
#data.index = data.index.tz_convert('UTC')