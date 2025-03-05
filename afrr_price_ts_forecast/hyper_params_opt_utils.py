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


def save_model_results(model_type, best_params, metrics, output_dir="afrr_price_ts_forecast/results"):
    """
    Save model parameters and performance metrics to a JSON file.
    
    Args:
        model_type (str): Type of model (gp, lr, xgb)
        best_params (dict): Best hyperparameters from optimization
        metrics (dict): Performance metrics (RMSE, MAPE)
        output_dir (str): Directory to save results
    
    Returns:
        str: Path to the saved JSON file
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a results dictionary
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


def generate_historical_forecasts(model, model_type, afrr_pr_ts_scl_test, exog_ts_scl_test, afrr_pr_scaler, horizon=24):
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
    
    # Handle model-specific keyword arguments
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
    hist_forecasts = hist_forecasts.with_columns_renamed('aFRR_UpCapPriceEUR_cl', f'afrr_up_cap_price_{model_type}_hat')
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
    error_mape = mape(afrr_pr_ts_orig_test, hist_forecasts)
    print(f"RMSE: {error_rmse:.4f}")
    print(f"MAPE: {error_mape:.4f}")
    
    # Return metrics as dictionary
    return {
        "rmse": float(error_rmse),
        "mape": float(error_mape)
    }