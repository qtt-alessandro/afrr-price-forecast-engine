import polars as pl 
from dotenv import load_dotenv
import os 
import glob 

#define folder path and file path
folder_path = "/home/alqua/data/boiler_data/daily_data"

#load all the .cvs files in the folder 
csv_files_list = glob.glob(os.path.join(folder_path, "*.csv"))

dfs = []
#split the 
for csv_file in csv_files_list:
    # read the data
    data = pl.read_csv(csv_file)
    #load the data
    data = data.with_columns(
        pl.col("UTC").str.to_datetime("%Y-%m-%d %H:%M:%S+00:00").alias("UTC")
    )
    data = data.with_columns(pl.all().exclude("UTC").cast(pl.Float64))
    #data = data.slice(0,97)
    dfs.append(data)

data = pl.concat(dfs)
data = data.sort("UTC")

last_measurement = str(data.select("UTC").tail(n=1).cast(pl.Date).item().strftime("%Y_%m_%d"))

