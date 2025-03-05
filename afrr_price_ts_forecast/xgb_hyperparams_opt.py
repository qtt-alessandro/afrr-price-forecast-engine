#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XGBoost model for aFRR Price Forecasting

This standalone module handles optimization and training for XGBoost models.
"""

import optuna
from optuna.samplers import TPESampler
from darts import concatenate
from darts.models import XGBModel

# Import utility functions
from afrr_price_ts_forecast.hyper_params_opt.hyper_params_opt_utils import save_model_results, generate_historical_forecasts, plot_results


def optimize_model(afrr_pr_ts_scl_train, afrr_pr_ts_scl_test, exog_ts_scl_train, exog_ts_scl_test, output_chunk_length=24, n_trials=10):
    """
    Optimize XGBoost model hyperparameters.
    
    Args:
        afrr_pr_ts_scl_train (TimeSeries): Training target data
        afrr_pr_ts_scl_test (TimeSeries): Test target data
        exog_ts_scl_train (TimeSeries): Training exogenous data
        exog_ts_scl_test (TimeSeries): Test exogenous data
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
            # Train the model
            model.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
            
            # Make predictions on validation set
            pred = model.predict(
                n=len(afrr_pr_ts_scl_test), 
                past_covariates=concatenate([exog_ts_scl_train, exog_ts_scl_test], axis=0)
            )
            
            # Calculate error using Darts' RMSE function
            from darts.metrics import rmse
            error = rmse(afrr_pr_ts_scl_test, pred)
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
    print(f"Best RMSE: {study.best_value}")
    
    return best_params


def train_model(best_params, afrr_pr_ts_scl_train, exog_ts_scl_train, output_chunk_length=24):
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


def main(data_path="../data/afrr_price.parquet", output_chunk_length=24, horizon=24, n_trials=10, save_results=True, output_dir="results"):
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
    # Import here to avoid circular imports
    from afrr_preprocessing import preprocess_afrr_data
    
    # Preprocess data
    (
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        afrr_pr_ts_orig_train, 
        afrr_pr_ts_orig_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test,
        afrr_pr_scaler
    ) = preprocess_afrr_data(data_path)
    
    # Optimize model
    best_params = optimize_model(
        afrr_pr_ts_scl_train=afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test=afrr_pr_ts_scl_test, 
        exog_ts_scl_train=exog_ts_scl_train, 
        exog_ts_scl_test=exog_ts_scl_test,
        output_chunk_length=output_chunk_length,
        n_trials=n_trials
    )
    
    # Train model
    model = train_model(
        best_params=best_params, 
        afrr_pr_ts_scl_train=afrr_pr_ts_scl_train, 
        exog_ts_scl_train=exog_ts_scl_train,
        output_chunk_length=output_chunk_length
    )
    
    # Generate historical forecasts
    hist_forecasts = generate_historical_forecasts(
        model=model, 
        model_type="xgb",
        afrr_pr_ts_scl_test=afrr_pr_ts_scl_test, 
        exog_ts_scl_test=exog_ts_scl_test, 
        afrr_pr_scaler=afrr_pr_scaler,
        horizon=horizon
    )
    
    # Plot results and get metrics
    metrics = plot_results(afrr_pr_ts_orig_test, hist_forecasts, "xgb")
    
    # Save results if requested
    if save_results:
        save_model_results("xgb", best_params, metrics, output_dir)
    
    return model, hist_forecasts, best_params, metrics


if __name__ == "__main__":
    main()