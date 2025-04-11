#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear Regression model for aFRR Price Forecasting

This standalone module handles optimization and training for Linear Regression models
using a validation set for hyperparameter tuning.
"""
import optuna
from optuna.samplers import TPESampler
from darts.timeseries import concatenate
from darts.models import LinearRegressionModel
from utils.afrr_preprocessing import preprocess_afrr_data
from utils.forecast_utils import save_model_results, generate_historical_forecasts, plot_results
from darts.metrics import rmse


def optimize_model(afrr_pr_ts_scl_train, afrr_pr_ts_scl_val, exog_ts_scl_train, exog_ts_scl_val, output_chunk_length, n_trials):
    """
    Optimize Linear Regression model hyperparameters using validation set.
    
    Args:
        afrr_pr_ts_scl_train (TimeSeries): Training target data
        afrr_pr_ts_scl_val (TimeSeries): Validation target data
        exog_ts_scl_train (TimeSeries): Training exogenous data
        exog_ts_scl_val (TimeSeries): Validation exogenous data
        output_chunk_length (int): Output chunk length
        n_trials (int): Number of optimization trials
        
    Returns:
        dict: Best hyperparameters
    """
    def objective(trial):
        # Optimize lags
        lags_max = trial.suggest_int("lags_max", 100, 400)
        lags_past_covariates_max = trial.suggest_int("lags_past_covariates_max", 50, 150)
        
        # Create and train model
        model = LinearRegressionModel(
            output_chunk_length=output_chunk_length,
            lags=list(range(-1, -lags_max, -1)),
            lags_past_covariates=list(range(-1, -lags_past_covariates_max, -1)),
        )
        
        try:
            # Train the model on training data
            model.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
            
            # Make predictions on validation set
            pred = model.predict(
                n=len(afrr_pr_ts_scl_val), 
                past_covariates=concatenate([exog_ts_scl_train, exog_ts_scl_val], axis=0)
            )
            
            # Calculate error on validation set using Darts' RMSE function
            error = rmse(afrr_pr_ts_scl_val, pred)
            return error
        except Exception as e:
            print(f"Error: {e}")
            return float("inf")
    
    sampler = TPESampler()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    print(f"Best parameters for LR model: {best_params}")
    print(f"Best validation RMSE: {study.best_value}")
    
    return best_params


def train_model(best_params, afrr_pr_ts_scl_train, exog_ts_scl_train, output_chunk_length):
    """
    Train Linear Regression model with the best parameters.
    
    Args:
        best_params (dict): Best hyperparameters from optimization
        afrr_pr_ts_scl_train (TimeSeries): Training target data
        exog_ts_scl_train (TimeSeries): Training exogenous data
        output_chunk_length (int): Output chunk length
        
    Returns:
        LinearRegressionModel: Trained model
    """
    model = LinearRegressionModel(
        output_chunk_length=output_chunk_length,
        lags=list(range(-1, -best_params["lags_max"], -1)),
        lags_past_covariates=list(range(-1, -best_params["lags_past_covariates_max"], -1))
    )
    
    model.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
    
    return model


def train_final_model(best_params, afrr_pr_ts_scl_train, afrr_pr_ts_scl_val, exog_ts_scl_train, exog_ts_scl_val, output_chunk_length):
    """
    Train the final model on combined training and validation data with best parameters.
    
    Args:
        best_params (dict): Best hyperparameters from optimization
        afrr_pr_ts_scl_train (TimeSeries): Training target data
        afrr_pr_ts_scl_val (TimeSeries): Validation target data
        exog_ts_scl_train (TimeSeries): Training exogenous data
        exog_ts_scl_val (TimeSeries): Validation exogenous data
        output_chunk_length (int): Output chunk length
        
    Returns:
        LinearRegressionModel: Trained model on combined data
    """
    # Combine training and validation data
    combined_train_data = concatenate([afrr_pr_ts_scl_train, afrr_pr_ts_scl_val], axis=0)
    combined_exog_data = concatenate([exog_ts_scl_train, exog_ts_scl_val], axis=0)
    
    model = LinearRegressionModel(
        output_chunk_length=output_chunk_length,
        lags=list(range(-1, -best_params["lags_max"], -1)),
        lags_past_covariates=list(range(-1, -best_params["lags_past_covariates_max"], -1))
    )
    
    model.fit(combined_train_data, past_covariates=combined_exog_data)
    
    return model


def main(data_path, output_chunk_length, horizon, n_trials, save_results, output_dir):
    """
    Main function to run the complete LR model pipeline for aFRR price forecasting.
    
    Args:
        data_path (str): Path to the parquet file containing aFRR price data
        output_chunk_length (int): Output chunk length
        horizon (int): Forecast horizon
        n_trials (int): Number of optimization trials
        save_results (bool): Whether to save results to JSON
        output_dir (str): Directory to save results
        
    Returns:
        tuple: Tuple containing trained model, forecasts, best parameters, and metrics
    """
    # Process data with validation split
    (
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_val,
        afrr_pr_ts_scl_test, 
        afrr_pr_ts_orig_train,
        afrr_pr_ts_orig_val, 
        afrr_pr_ts_orig_test, 
        exog_ts_scl_train,
        exog_ts_scl_val, 
        exog_ts_scl_test,
        afrr_pr_scaler
    ) = preprocess_afrr_data(
        data_path=data_path,
        train_start="2024-10-10 00:00:00",
        val_start="2025-01-01 00:00:00",
        test_start="2025-03-01 00:00:00",
        test_end="2025-04-09 23:59:59",
        use_validation=True
    )
    
    # Find best hyperparameters using validation set
    best_params = optimize_model(
        afrr_pr_ts_scl_train=afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_val=afrr_pr_ts_scl_val,
        exog_ts_scl_train=exog_ts_scl_train, 
        exog_ts_scl_val=exog_ts_scl_val,
        output_chunk_length=output_chunk_length,
        n_trials=n_trials
    )
    
    # Train model on combined training+validation data for final evaluation
    final_model = train_final_model(
        best_params=best_params,
        afrr_pr_ts_scl_train=afrr_pr_ts_scl_train,
        afrr_pr_ts_scl_val=afrr_pr_ts_scl_val,
        exog_ts_scl_train=exog_ts_scl_train,
        exog_ts_scl_val=exog_ts_scl_val,
        output_chunk_length=output_chunk_length
    )
    
    # Generate historical forecasts on test set
    hist_forecasts = generate_historical_forecasts(
        model=final_model, 
        model_type="lr",
        afrr_pr_ts_scl_test=afrr_pr_ts_scl_test, 
        exog_ts_scl_test=exog_ts_scl_test, 
        afrr_pr_scaler=afrr_pr_scaler,
        horizon=horizon
    )
    
    # Evaluate on test set
    metrics = plot_results(afrr_pr_ts_orig_test, hist_forecasts, "lr")
    print(f"Test set metrics: {metrics}")
    
    if save_results:
        save_model_results("lr", best_params, metrics, output_dir)
    
    return final_model, hist_forecasts, best_params, metrics


if __name__ == "__main__":
    default_data_path = "../data/afrr_price.parquet"
    default_output_dir = "../data/results/"
    default_output_chunk_length = 24
    default_horizon = 24
    default_n_trials = 10
    default_save_results = True
    
    main(
        data_path=default_data_path,
        output_chunk_length=default_output_chunk_length,
        horizon=default_horizon,
        n_trials=default_n_trials,
        save_results=default_save_results,
        output_dir=default_output_dir
    )
