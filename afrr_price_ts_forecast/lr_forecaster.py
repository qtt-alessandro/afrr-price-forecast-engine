import pandas as pd
from sklearn.linear_model import QuantileRegressor
from darts import TimeSeries
from darts.timeseries import concatenate
from darts.models import RegressionModel

from utils import load_data, load_hyperparameters, get_forecast_params

def train_lr_models(afrr_pr_ts_scl_train, exog_ts_scl_train, lr_params, output_chunk_length, quantiles):
    """
    Train Linear Regression models for each quantile
    
    Args:
        afrr_pr_ts_scl_train: Scaled training data for AFRR price
        exog_ts_scl_train: Scaled training data for exogenous variables
        lr_params: Hyperparameters for LR models
        output_chunk_length: Length of forecast output chunks
        quantiles: List of quantiles to calculate
        
    Returns:
        Dictionary of trained LR models
    """
    lr_quantile_models = {}
    
    for q in quantiles:
        quantile_regressor = QuantileRegressor(
            alpha=0,
            quantile=q,
            solver='highs'
        )
        
        lr_quantile_models[q] = RegressionModel(
            output_chunk_length=output_chunk_length,
            lags=list(range(-1, -lr_params["lags_max"], -1)),
            lags_past_covariates=list(range(-1, -lr_params["lags_past_covariates_max"], -1)), 
            model=quantile_regressor
        )
        
        lr_quantile_models[q].fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
    
    return lr_quantile_models

def generate_lr_forecasts(
    lr_models, 
    afrr_pr_ts_scl_train, 
    afrr_pr_ts_scl_test, 
    exog_ts_scl_train, 
    exog_ts_scl_test, 
    forecast_horizon, 
    stride, 
    quantiles, 
    afrr_pr_scaler
):
    """
    Generate forecasts using the LR models
    
    Args:
        lr_models: Dictionary of trained LR models
        afrr_pr_ts_scl_train, afrr_pr_ts_scl_test: Scaled training and test data for AFRR price
        exog_ts_scl_train, exog_ts_scl_test: Scaled training and test data for exogenous variables
        forecast_horizon: Horizon for forecasting
        stride: Stride for historical forecasts
        quantiles: List of quantiles to calculate
        afrr_pr_scaler: Scaler for inverse transformation
        
    Returns:
        DataFrame with forecast results
    """
    lr_backtest_forecasts = {}
    
    for q in quantiles:
        lr_backtest_forecasts[q] = lr_models[q].historical_forecasts(
            series=concatenate([afrr_pr_ts_scl_train, afrr_pr_ts_scl_test], axis=0),
            past_covariates=concatenate([exog_ts_scl_train, exog_ts_scl_test], axis=0),
            start=1,  
            forecast_horizon=forecast_horizon,
            enable_optimization=True,
            stride=stride,
            retrain=False,
            last_points_only=False,
            verbose=True
        )
    
    # Combine forecasts for all quantiles
    dfs = [
        afrr_pr_scaler.inverse_transform(
            concatenate(lr_backtest_forecasts[q]).with_columns_renamed(
                ['aFRR_UpCapPriceEUR_cl'], 
                col_names_new=[f'lr_afrr_up_cap_price_{q}']
            )
        ).pd_dataframe()
        for q in quantiles
    ]

    lr_df_hat = pd.concat(dfs, axis=1)
    
    return lr_df_hat

def run_lr_pipeline():
    """
    Run the full Linear Regression pipeline
    """
    # Load data
    (
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        afrr_pr_ts_orig_train, 
        afrr_pr_ts_orig_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test,
        afrr_pr_scaler
    ) = load_data()
    
    # Load hyperparameters
    lr_opt_params, _ = load_hyperparameters()
    
    # Get common forecast parameters
    forecast_params = get_forecast_params()
    output_chunk_length = forecast_params['output_chunk_length']
    forecast_horizon = forecast_params['forecast_horizon']
    stride = forecast_params['stride']
    quantiles = forecast_params['quantiles']
    
    # Train LR models
    lr_models = train_lr_models(
        afrr_pr_ts_scl_train, 
        exog_ts_scl_train, 
        lr_opt_params, 
        output_chunk_length,
        quantiles
    )
    
    # Generate forecasts
    lr_results = generate_lr_forecasts(
        lr_models,
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test,
        forecast_horizon,
        stride,
        quantiles,
        afrr_pr_scaler
    )
    
    return lr_models, lr_results

if __name__ == "__main__":
    lr_models, lr_results = run_lr_pipeline()
    print("LR forecasting completed. Results shape:", lr_results.shape)