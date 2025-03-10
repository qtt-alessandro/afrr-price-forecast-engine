import json
import pandas as pd
from darts import TimeSeries
from darts.timeseries import concatenate
from afrr_preprocessing import preprocess_afrr_data


def load_data(data_path="../data/afrr_price.parquet"):
    """
    Load and preprocess AFRR data
    
    Returns:
        Tuple of preprocessed data components
    """
    return preprocess_afrr_data(data_path)

def load_hyperparameters(lr_file='results/lr_hp_results.json', gp_file='results/gp_hp_results.json'):
    """
    Load hyperparameters from JSON files for both models
    
    Returns:
        Tuple of (lr_opt_params, gp_opt_params)
    """
    # Load the JSON file with the best parameters for LR
    with open(lr_file, 'r') as f:
        lr_hyper_opt_params_dict = json.load(f)
    
    # Load the JSON file with the best parameters for GP
    with open(gp_file, 'r') as f:
        gp_hyper_opt_params_dict = json.load(f)

    # Extract the parameters from the loaded dictionary
    lr_opt_params = lr_hyper_opt_params_dict["parameters"]
    gp_opt_params = gp_hyper_opt_params_dict["parameters"]
    
    return lr_opt_params, gp_opt_params

def get_forecast_params():
    """
    Return common forecasting parameters used by both models
    """
    return {
        'output_chunk_length': 24,
        'forecast_horizon': 24, 
        'stride': 24,
        'start_idx': 24,
        'quantiles': [0.1, 0.5, 0.9]
    }