from datetime import datetime
import pandas as pd
import requests


class PicassoPrices():
    """
    Class to fetch prices from the Energidata API for the aFRR market in the DK1/DK2 price area
    """

    def __init__(self, start_date, end_date, price_area):
        """
        Args:
        start_date  (str): Start date in the format 'YYYY-MM-DD HH:MM:SS' in the UTC timezone
        end_date    (str): End date in the format 'YYYY-MM-DD HH:MM:SS' in the UTC timezone
        price_area  (str): Price area for which the prices should be fetched. Should be 'DK1' or 'DK2'
        """
        self.start_str= start_date
        self.end_str = end_date

        self.start_dt = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        self.end_dt = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')

        self.start_api = self.start_dt.strftime('%Y-%m-%d')
        self.end_api = (self.end_dt+pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        self.area = price_area

    def fetch_prices_energidata(self):
        response_data = requests.get(
            url = f'https://api.energidataservice.dk/dataset/AfrrEnergyActivated?start={self.start_api}&end={self.end_api}&filter={{"PriceArea":["{self.area}"]}}&timezone=UTC')
        results_data = response_data.json()
        records_data = results_data.get('records', [])
        df_picasso = pd.DataFrame(records_data)[['ActivationTime','aFRR_DownActivatedPriceEUR','aFRR_UpActivatedPriceEUR']]
        df_picasso['ActivationTime'] = pd.to_datetime(df_picasso['ActivationTime'], format = '%Y-%m-%dT%H:%M:%S', utc = True)
        df_picasso.set_index('ActivationTime', inplace = True)
        df_picasso = df_picasso.resample('4S').mean()
        return df_picasso
    
    def prices_per_MTU(self): 
        df_picasso = self.fetch_prices_energidata()
        N = 15*60 # number of prices per MTU

        df_MTU = pd.DataFrame()
        df_MTU['aFRR_DownActivatedPrice_AvgMTU_EUR'] = df_picasso['aFRR_DownActivatedPriceEUR'].resample('15T').mean()
        df_MTU['aFRR_UpActivatedPrice_AvgMTU_EUR'] = df_picasso['aFRR_UpActivatedPriceEUR'].resample('15T').mean()
        df_MTU['aFRR_DownActivatedPrice_MaxMTU_EUR'] = df_picasso['aFRR_DownActivatedPriceEUR'].resample('15T').max()
        df_MTU['aFRR_UpActivatedPrice_MaxMTU_EUR'] = df_picasso['aFRR_UpActivatedPriceEUR'].resample('15T').max()
        df_MTU['aFRR_DownActivatedPrice_MinMTU_EUR'] = df_picasso['aFRR_DownActivatedPriceEUR'].resample('15T').min()
        df_MTU['aFRR_UpActivatedPrice_MinMTU_EUR'] = df_picasso['aFRR_UpActivatedPriceEUR'].resample('15T').min()
        df_MTU['aFRR_UpActivationFreq'] = df_picasso['aFRR_UpActivatedPriceEUR'].resample('15T').count()/N
        df_MTU['aFRR_DownActivationFreq'] = df_picasso['aFRR_DownActivatedPriceEUR'].resample('15T').count()/N
        return df_MTU

if __name__ == '__main__':
    start = '2025-01-01 00:00:00'
    end = '2025-01-07 23:59:59'
    picasso = PicassoPrices(start, end, 'DK1')
    print(picasso.prices_per_MTU())