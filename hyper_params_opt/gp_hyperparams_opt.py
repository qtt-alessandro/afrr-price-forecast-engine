#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optuna
from optuna.samplers import GPSampler
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from darts import concatenate
from darts.metrics import rmse

from models.gp_regressor import GPRegressor
from utils.afrr_preprocessing import preprocess_afrr_data
from utils.forecast_utils import save_model_results, generate_historical_forecasts, plot_results


def optimize_model(afrr_pr_ts_scl_train, afrr_pr_ts_scl_val, exog_ts_scl_train, exog_ts_scl_val, output_chunk_length, n_trials):

    kernel = DotProduct() + WhiteKernel()
    
    def objective(trial):
        # Optimize lags
        lags = trial.suggest_int("lags", 12, 48)
        lags_past_covariates = trial.suggest_int("lags_past_covariates", 12, 48)
        
        # Create and train model
        model = GPRegressor(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            output_chunk_length=output_chunk_length,
            kernel=kernel
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
    sampler = GPSampler()
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters for GP model: {best_params}")
    print(f"Best validation RMSE: {study.best_value}")
    
    return best_params


def train_model(best_params, afrr_pr_ts_scl_train, exog_ts_scl_train, output_chunk_length):

    kernel = DotProduct() + WhiteKernel()
    
    model = GPRegressor(
        lags=best_params["lags"],
        lags_past_covariates=best_params["lags_past_covariates"],
        output_chunk_length=output_chunk_length,
        kernel=kernel
    )
    
    # Train the model
    model.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
    
    return model


def train_final_model(best_params, afrr_pr_ts_scl_train, afrr_pr_ts_scl_val, exog_ts_scl_train, exog_ts_scl_val, output_chunk_length):

    # Combine training and validation data
    combined_train_data = concatenate([afrr_pr_ts_scl_train, afrr_pr_ts_scl_val], axis=0)
    combined_exog_data = concatenate([exog_ts_scl_train, exog_ts_scl_val], axis=0)
    
    kernel = DotProduct() + WhiteKernel()
    
    model = GPRegressor(
        lags=best_params["lags"],
        lags_past_covariates=best_params["lags_past_covariates"],
        output_chunk_length=output_chunk_length,
        kernel=kernel
    )
    
    model.fit(combined_train_data, past_covariates=combined_exog_data)
    
    return model


def main(data_path, output_chunk_length, horizon, n_trials, target_col, save_results, output_dir):

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
    
    best_params = optimize_model(
        afrr_pr_ts_scl_train=afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_val=afrr_pr_ts_scl_val,
        exog_ts_scl_train=exog_ts_scl_train, 
        exog_ts_scl_val=exog_ts_scl_val,
        output_chunk_length=output_chunk_length,
        n_trials=n_trials
    )
    
    final_model = train_final_model(
        best_params=best_params,
        afrr_pr_ts_scl_train=afrr_pr_ts_scl_train,
        afrr_pr_ts_scl_val=afrr_pr_ts_scl_val,
        exog_ts_scl_train=exog_ts_scl_train,
        exog_ts_scl_val=exog_ts_scl_val,
        output_chunk_length=output_chunk_length
    )
    
    hist_forecasts = generate_historical_forecasts(
        model=final_model, 
        model_type="gp",
        afrr_pr_ts_scl_test=afrr_pr_ts_scl_test, 
        exog_ts_scl_test=exog_ts_scl_test, 
        afrr_pr_scaler=afrr_pr_scaler,
        horizon=horizon,
        target_col=target_col
    )
    
    hist_forecasts_clipped = hist_forecasts
    
    metrics = plot_results(afrr_pr_ts_orig_test, hist_forecasts_clipped, "gp")
    print(f"Test set metrics: {metrics}")
    
    if save_results:
        save_model_results("gp_" + str(target_col), best_params, metrics, output_dir)
    
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
        target_col='aFRR_DownCapPriceEUR',
        save_results=default_save_results,
        output_dir=default_output_dir
    )
