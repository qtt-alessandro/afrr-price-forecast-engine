#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for aFRR Price Forecasting
"""

import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from darts.timeseries import concatenate
from darts.metrics import rmse, mape
from utils.afrr_preprocessing import preprocess_afrr_data

def save_model_results(model_type, best_params, metrics, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {
        "model_type": model_type,
        "parameters": best_params,
        "metrics": metrics,
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    }
    
    # Generate filename
    filename = f"{model_type}_hp_results.json"
    filepath = os.path.join(output_dir, filename)
    
    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {filepath}")
    return filepath


def generate_historical_forecasts(model, model_type, afrr_pr_ts_scl_test, exog_ts_scl_test, afrr_pr_scaler, horizon, target_col):
    """
    Generate historical forecasts using the trained model.
    
    Args:
        model: Trained model
        model_type (str): Type of model ("gp", "lr", or "xgb")
        afrr_pr_ts_scl_test (TimeSeries): Test target data
        exog_ts_scl_test (TimeSeries): Test exogenous data
        afrr_pr_scaler (Pipeline): Scaler for transforming predictions back to original scale
        horizon (int): Forecast horizon
        
    Returns:
        TimeSeries: Historical forecasts
    """
    print(f'Historical Forecast Using Fitted {model_type.upper()} Model...')
    
    kwargs = {}
    if model_type == "gp":
        kwargs["predict_likelihood_parameters"] = False
    
    hist_forecasts = model.historical_forecasts(
        series=afrr_pr_ts_scl_test,
        past_covariates=exog_ts_scl_test,
        start=0.05,
        forecast_horizon=horizon,
        last_points_only=False,
        stride=24,
        retrain=False,
        verbose=True,
        **kwargs
    )
    
    hist_forecasts = concatenate(hist_forecasts)
    if target_col == 'aFRR_UpCapPriceEUR':
        hist_forecasts = hist_forecasts.with_columns_renamed('aFRR_UpCapPriceEUR_cl', f'afrr_up_cap_price_{model_type}_hat')
        hist_forecasts = afrr_pr_scaler.inverse_transform(hist_forecasts)
    elif target_col == 'aFRR_DownCapPriceEUR':
        hist_forecasts = hist_forecasts.with_columns_renamed('aFRR_DownCapPriceEUR_cl', f'afrr_down_cap_price_{model_type}_hat')
        hist_forecasts = afrr_pr_scaler.inverse_transform(hist_forecasts)
        
    return hist_forecasts


def plot_results(afrr_pr_ts_orig_test, hist_forecasts, model_type):
    """
    Plot the test data and forecasts and calculate metrics.
    
    Args:
        afrr_pr_ts_orig_test (TimeSeries): Original test data
        hist_forecasts (TimeSeries): Forecasts to plot
        model_type (str): Type of model for labeling
    
    Returns:
        dict: Dictionary with RMSE and MAPE metrics
    """
    plt.figure(figsize=(12, 6))
    afrr_pr_ts_orig_test.plot(label='Actual')
    hist_forecasts.plot(label=f'{model_type.upper()} Forecast')
    plt.title(f'aFRR Up-Capacity Price Forecasting ({model_type.upper()} Model)')
    plt.xlabel('Time')
    plt.ylabel('EUR')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate error metrics
    error_rmse = rmse(afrr_pr_ts_orig_test, hist_forecasts)
    #error_mape = mape(afrr_pr_ts_orig_test.clip_lower(0), hist_forecasts)
    print(f"RMSE: {error_rmse:.4f}")
    #print(f"MAPE: {error_mape:.4f}")
    
    # Return metrics as dictionary
    return {
        "rmse": float(error_rmse),
        "mape": None
    }
    
def load_data(data_path):
    """
    Load and preprocess AFRR data
    
    Returns:
        Tuple of preprocessed data components
    """
    return preprocess_afrr_data(data_path)

def load_hyperparameters(file_path):

    with open(file_path, 'r') as f:
        model_hyper_opt_params_dict = json.load(f)

    model_opt_params = model_hyper_opt_params_dict["parameters"]
    
    return model_opt_params

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
