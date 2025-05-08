#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost model for aFRR Price Forecasting

This standalone module handles optimization and training for XGBoost models
using a validation set for hyperparameter tuning.
"""
import optuna
from optuna.samplers import TPESampler
from darts import concatenate
from darts.metrics import rmse
from darts.models import XGBModel
from utils.afrr_preprocessing import preprocess_afrr_data
from utils.forecast_utils import save_model_results, generate_historical_forecasts, plot_results


def optimize_model(afrr_pr_ts_scl_train, afrr_pr_ts_scl_val, exog_ts_scl_train, exog_ts_scl_val, output_chunk_length, n_trials):
    """
    Optimize XGBoost model hyperparameters using validation set.
    
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
    # Define time series encoders for XGBoost
    ts_encoders = {
        'cyclic': {'future': ['month']},
        'datetime_attribute': {'future': ['hour', 'dayofweek']},
        'position': {'past': ['relative'], 'future': ['relative']},
        'tz': 'UTC'
    }
    
    def objective(trial):
        # Optimize lags
        lags = trial.suggest_int("lags", 24, 96)
        lags_past_covariates = trial.suggest_int("lags_past_covariates", 12, 48)
        
        # Optimize XGBoost specific parameters
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        
        # Create and train model
        model = XGBModel(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            add_encoders=ts_encoders,
            output_chunk_length=output_chunk_length,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators
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
    
    # Create study and optimize
    sampler = TPESampler()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters for XGB model: {best_params}")
    print(f"Best validation RMSE: {study.best_value}")
    
    return best_params


def train_model(best_params, afrr_pr_ts_scl_train, exog_ts_scl_train, output_chunk_length):
    """
    Train XGBoost model with the best parameters.
    
    Args:
        best_params (dict): Best hyperparameters from optimization
        afrr_pr_ts_scl_train (TimeSeries): Training target data
        exog_ts_scl_train (TimeSeries): Training exogenous data
        output_chunk_length (int): Output chunk length
        
    Returns:
        XGBModel: Trained model
    """
    # Define time series encoders for XGBoost
    ts_encoders = {
        'cyclic': {'future': ['month']},
        'datetime_attribute': {'future': ['hour', 'dayofweek']},
        'position': {'past': ['relative'], 'future': ['relative']},
        'tz': 'UTC'
    }
    
    model = XGBModel(
        lags=best_params["lags"],
        lags_past_covariates=best_params["lags_past_covariates"],
        add_encoders=ts_encoders,
        output_chunk_length=output_chunk_length,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"]
    )
    
    # Train the model
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
        XGBModel: Trained model on combined data
    """
    # Define time series encoders for XGBoost
    ts_encoders = {
        'cyclic': {'future': ['month']},
        'datetime_attribute': {'future': ['hour', 'dayofweek']},
        'position': {'past': ['relative'], 'future': ['relative']},
        'tz': 'UTC'
    }
    
    # Combine training and validation data
    combined_train_data = concatenate([afrr_pr_ts_scl_train, afrr_pr_ts_scl_val], axis=0)
    combined_exog_data = concatenate([exog_ts_scl_train, exog_ts_scl_val], axis=0)
    
    model = XGBModel(
        lags=best_params["lags"],
        lags_past_covariates=best_params["lags_past_covariates"],
        add_encoders=ts_encoders,
        output_chunk_length=output_chunk_length,
        max_depth=best_params["max_depth"],
        learning_rate=best_params["learning_rate"],
        n_estimators=best_params["n_estimators"]
    )
    
    # Train the model on combined data
    model.fit(combined_train_data, past_covariates=combined_exog_data)
    
    return model


def main(data_path, output_chunk_length, horizon, n_trials, save_results, target_col, output_dir):
    """
    Main function to run the complete XGB model pipeline for aFRR price forecasting.
    
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
        use_validation=True, 
        target_col=target_col
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
        model_type="xgb",
        afrr_pr_ts_scl_test=afrr_pr_ts_scl_test, 
        exog_ts_scl_test=exog_ts_scl_test, 
        afrr_pr_scaler=afrr_pr_scaler,
        horizon=horizon,
        target_col=target_col
    )
    
    # Clip forecasts to ensure positive values before evaluation
    hist_forecasts_clipped = hist_forecasts
    
    # Plot results and get metrics
    metrics = plot_results(afrr_pr_ts_orig_test, hist_forecasts_clipped, "xgb")
    print(f"Test set metrics: {metrics}")
    
    # Save results if requested
    if save_results:
        save_model_results("xgb", best_params, metrics, output_dir)
    
    return final_model, hist_forecasts_clipped, best_params, metrics


if __name__ == "__main__":
    default_data_path = "./data/afrr_price.parquet"
    default_output_dir = "./data/results/"
    default_output_chunk_length = 24
    default_horizon = 24
    default_n_trials = 1
    default_save_results = True
    
    main(
        data_path=default_data_path,
        output_chunk_length=default_output_chunk_length,
        horizon=default_horizon,
        n_trials=default_n_trials,
        target_col='aFRR_UpCapPriceEUR',
        save_results=default_save_results,
        output_dir=default_output_dir
    )
