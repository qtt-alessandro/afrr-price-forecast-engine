import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import os 
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly_resampler import FigureResampler, FigureWidgetResampler

from utils.afrr_preprocessing import preprocess_afrr_data
from ensamble_forecast.lr_forecaster import run_lr_pipeline
from ensamble_forecast.gp_forecaster import run_gp_pipeline
from ensamble_forecast.xgb_forecaster import run_xgb_pipeline

from utils.forecast_utils import get_forecast_params
from darts.timeseries import concatenate


data_path = "../data/afrr_price.parquet"
hyper_params_path = "../data/results/lr_hp_results.json"
train_start = "2024-10-01 22:00:00"
test_start = "2025-01-09 22:00:00"
test_end = "2025-03-20 22:00:00"


# Get forecast parameters
forecast_params = get_forecast_params()
output_chunk_length = forecast_params['output_chunk_length']
forecast_horizon = forecast_params['forecast_horizon']
stride = forecast_params['stride']
quantiles = forecast_params['quantiles']

# Run the pipeline with all parameters defined at the end
lr_models, lr_results = run_lr_pipeline(
    data_path=data_path,
    hyper_params_path=hyper_params_path,
    train_start=train_start,
    test_start=test_start,
    test_end=test_end,
    output_chunk_length=output_chunk_length,
    forecast_horizon=forecast_horizon,
    stride=stride,
    quantiles=quantiles
)

hyper_params_path_gp = "../data/results/gp_hp_results.json"

forecast_params = get_forecast_params()
output_chunk_length = forecast_params['output_chunk_length']
forecast_horizon = forecast_params['forecast_horizon']
stride = forecast_params['stride']
quantiles = forecast_params['quantiles']

# Run the pipeline with all parameters defined at the end
gp_model, gp_results = run_gp_pipeline(
    data_path=data_path,
    hyper_params_path=hyper_params_path_gp,
    train_start=train_start,
    test_start=test_start,
    test_end=test_end,
    output_chunk_length=output_chunk_length,
    forecast_horizon=forecast_horizon,
    stride=stride,
    quantiles=quantiles
)


print(gp_results)



hyper_params_path = "../data/results/xgb_hp_results.json"


xgb_model, xgb_results = run_xgb_pipeline(
    data_path=data_path,
    hyper_params_path=hyper_params_path,
    train_start=train_start,
    test_start=test_start,
    test_end=test_end,
    output_chunk_length=output_chunk_length,
    forecast_horizon=forecast_horizon,
    stride=stride
)
