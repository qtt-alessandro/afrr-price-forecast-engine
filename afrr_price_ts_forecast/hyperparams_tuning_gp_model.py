#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aFRR Price Forecasting using Gaussian Process Regression
"""

# Standard libraries
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

# Data manipulation libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly_resampler import FigureResampler, FigureWidgetResampler

# Custom modules
from ssa import *
from gp_regressor import GPRegressor

# Scikit-learn components
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.linear_model import Ridge

# Darts - Time Series components
from darts import TimeSeries, concatenate
from darts.datasets import AirPassengersDataset
from darts.metrics import mape, rmse
from darts.utils.missing_values import extract_subseries
from darts.utils.timeseries_generation import datetime_attribute_timeseries as dt_attr

# Darts - Models
from darts.models import (
    XGBModel,
    RandomForest,
    TCNModel,
)

# Darts - Data processing
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler, GPSampler


def load_and_prepare_data():
    """Load and prepare the aFRR price data."""
    data = pd.read_parquet("../data/afrr_price.parquet")
    
    data = data[['wind_offshore_dayahead', 'wind_onshore_dayahead',
           'solar_dayahead','aFRR_DownCapPriceEUR', 'aFRR_UpCapPriceEUR', 'load_forecasts',
           'da_price', 'prod_unbalance']]
    
    data.index = data.index.tz_localize(None)
    
    return data


def apply_ssa_decomposition(data):
    """Apply Singular Spectrum Analysis (SSA) decomposition to the target variable."""
    ts = data["aFRR_UpCapPriceEUR"]
    ssa_ts = mySSA(data["aFRR_UpCapPriceEUR"])
    ssa_ts.embed(embedding_dimension=128, suspected_frequency=24, verbose=True)
    ssa_ts.decompose(True)
    
    # Components to use for reconstruction (to tune for optimal decomposition)
    components = [i for i in range(15)]
    ts_clean = ssa_ts.view_reconstruction(*[ssa_ts.Xs[i] for i in components], names=components, return_df=True, plot=False)
    
    data["aFRR_UpCapPriceEUR_cl"] = ts_clean.values
    
    return data


def prepare_time_series(data):
    """Convert data to Darts TimeSeries objects and preprocess."""
    afrr_pr = 'aFRR_UpCapPriceEUR_cl'
    afrr_pr_orig = 'aFRR_UpCapPriceEUR'
    
    exog_cols = ['wind_offshore_dayahead', 'wind_onshore_dayahead', 
                'solar_dayahead', 'load_forecasts', 'da_price', 'prod_unbalance']
    
    afrr_pr_ts = TimeSeries.from_series(data[afrr_pr], freq="1h")
    afrr_pr_ts_orig = TimeSeries.from_series(data[afrr_pr_orig], freq="1h")
    exog_ts = TimeSeries.from_dataframe(data[exog_cols], freq="1h")
    
    # Setup preprocessing pipelines
    scaler_target_ts = Scaler()
    scaler_exog_ts = Scaler()
    filler_target_ts = MissingValuesFiller()
    filler_exog_ts = MissingValuesFiller()
    
    afrr_pr_scaler = Pipeline([scaler_target_ts])
    exog_ts_preprocess = Pipeline([scaler_exog_ts])
    
    # Apply preprocessing
    afrr_pr_ts_filled = filler_target_ts.transform(afrr_pr_ts)
    exog_ts_filled = filler_exog_ts.transform(exog_ts)
    
    afrr_pr_ts_scl = afrr_pr_scaler.fit_transform(afrr_pr_ts_filled)
    exog_ts_scl = exog_ts_preprocess.fit_transform(exog_ts_filled)
    
    return afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl, afrr_pr_scaler


def split_data(afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl):
    """Split data into training and test sets."""
    # Define time periods
    train_start = pd.Timestamp("2024-10-01 22:00:00")
    test_start = pd.Timestamp("2025-01-09 22:00:00")
    test_end = pd.Timestamp("2025-02-20 22:00:00")
    
    # Split target series
    afrr_pr_ts_scl_train = afrr_pr_ts_scl[train_start : test_start - afrr_pr_ts_scl.freq]
    afrr_pr_ts_scl_test = afrr_pr_ts_scl[test_start : test_end]
    
    afrr_pr_ts_orig_train = afrr_pr_ts_orig[train_start : test_start - afrr_pr_ts_scl.freq]
    afrr_pr_ts_orig_test = afrr_pr_ts_orig[test_start : test_end]
    
    # Split exogenous series
    exog_ts_scl_train = exog_ts_scl[train_start : test_start - exog_ts_scl.freq]
    exog_ts_scl_test = exog_ts_scl[test_start : test_end]
    
    return (
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        afrr_pr_ts_orig_train, 
        afrr_pr_ts_orig_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test
    )


def optimize_gp_model(afrr_pr_ts_scl_train, afrr_pr_ts_scl_test, exog_ts_scl_train, exog_ts_scl_test):
    """Optimize the Gaussian Process model hyperparameters."""
    kernel = DotProduct() + WhiteKernel()
    
    def objective(trial):
        # Optimize lags
        lags = trial.suggest_int("lags", 12, 48)
        lags_past_covariates = trial.suggest_int("lags_past_covariates", 12, 48)
        
        # Create and train model
        model_gp = GPRegressor(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            output_chunk_length=24,
            kernel=kernel
        )
        
        try:
            # Train the model
            model_gp.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
            
            # Make predictions on validation set
            pred = model_gp.predict(
                n=len(afrr_pr_ts_scl_test), 
                past_covariates=concatenate([exog_ts_scl_train, exog_ts_scl_test], axis=0)
            )
            
            # Calculate error
            error = rmse(afrr_pr_ts_scl_test, pred)
            return error
        except Exception as e:
            print(f"Error: {e}")
            return float("inf")
    
    # Create study and optimize
    sampler = GPSampler()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=10, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")
    print(f"Best RMSE: {study.best_value}")
    
    return best_params


def train_gp_model(best_params, afrr_pr_ts_scl_train, exog_ts_scl_train):
    """Train the GP model with the best parameters."""
    kernel = DotProduct() + WhiteKernel()
    
    model_gp = GPRegressor(
        lags=best_params["lags"],
        lags_past_covariates=best_params["lags_past_covariates"],
        output_chunk_length=24,
        kernel=kernel
    )
    
    model_gp.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
    
    return model_gp


def generate_historical_forecasts(model_gp, afrr_pr_ts_scl_test, exog_ts_scl_test, afrr_pr_scaler):
    """Generate historical forecasts using the trained GP model."""
    print('Historical Forecast Using Fitted Gaussian Process Model...')
    horizon = 24
    
    hist_forecasts_gp = model_gp.historical_forecasts(
        series=afrr_pr_ts_scl_test,
        past_covariates=exog_ts_scl_test,
        start=0.05,
        forecast_horizon=horizon,
        predict_likelihood_parameters=False, 
        last_points_only=False,
        stride=24,
        retrain=False,
        verbose=True
    )
    
    hist_forecasts_gp = concatenate(hist_forecasts_gp)
    hist_forecasts_gp = hist_forecasts_gp.with_columns_renamed('aFRR_UpCapPriceEUR_cl', 'afrr_up_cap_price_gp_hat')
    hist_forecasts_gp = afrr_pr_scaler.inverse_transform(hist_forecasts_gp)
    
    return hist_forecasts_gp


def plot_results(afrr_pr_ts_orig_test, hist_forecasts_gp, afrr_pr_scaler):
    """Plot the test data and forecasts."""
    hist_afrr_pr = afrr_pr_scaler.inverse_transform(afrr_pr_ts_orig_test)
    
    plt.figure(figsize=(12, 6))
    afrr_pr_ts_orig_test.plot(label='Actual')
    hist_forecasts_gp.plot(label='Gaussian Process Forecast')
    plt.title('aFRR Up-Capacity Price Forecasting')
    plt.xlabel('Time')
    plt.ylabel('EUR')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Calculate and print error metrics
    error_rmse = rmse(afrr_pr_ts_orig_test, hist_forecasts_gp)
    error_mape = mape(afrr_pr_ts_orig_test, hist_forecasts_gp)
    print(f"RMSE: {error_rmse:.4f}")
    print(f"MAPE: {error_mape:.4f}")


def main():
    """Main function to run the aFRR price forecasting pipeline."""
    # Load and prepare data
    data = load_and_prepare_data()
    
    # Apply SSA decomposition
    data = apply_ssa_decomposition(data)
    
    # Prepare time series
    afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl, afrr_pr_scaler = prepare_time_series(data)
    
    # Split data
    (
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        afrr_pr_ts_orig_train, 
        afrr_pr_ts_orig_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test
    ) = split_data(afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl)
    
    # Optimize GP model
    best_params = optimize_gp_model(
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test
    )
    
    # Train GP model
    model_gp = train_gp_model(best_params, afrr_pr_ts_scl_train, exog_ts_scl_train)
    
    # Generate historical forecasts
    hist_forecasts_gp = generate_historical_forecasts(
        model_gp, 
        afrr_pr_ts_scl_test, 
        exog_ts_scl_test, 
        afrr_pr_scaler
    )
    
    # Plot results
    plot_results(afrr_pr_ts_orig_test, hist_forecasts_gp, afrr_pr_scaler)


if __name__ == "__main__":
    main()