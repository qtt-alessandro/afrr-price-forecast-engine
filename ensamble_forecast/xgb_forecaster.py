import pandas as pd
from darts import TimeSeries
from darts.timeseries import concatenate
from darts.models import XGBModel
from utils.forecast_utils import load_hyperparameters, get_forecast_params
from utils.afrr_preprocessing import preprocess_afrr_data


def train_xgb_model(afrr_pr_ts_scl_train, exog_ts_scl_train, xgb_params, output_chunk_length):
    """Train XGBoost model with quantile regression"""
    ts_encoders = {
        'cyclic': {'future': ['month']},
        'datetime_attribute': {'future': ['hour', 'dayofweek']},
        'position': {'past': ['relative'], 'future': ['relative']},
        'tz': 'UTC'
    }
    
    xgb_model = XGBModel(
        lags=xgb_params["lags"],
        lags_past_covariates=xgb_params["lags_past_covariates"],
        add_encoders=ts_encoders,
        output_chunk_length=output_chunk_length,
        max_depth=xgb_params["max_depth"],
        learning_rate=xgb_params["learning_rate"],
        n_estimators=xgb_params["n_estimators"],
        likelihood="quantile",
        quantiles=[0.1, 0.5, 0.9],
    )

    xgb_model.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
    
    return xgb_model


def unscale_quantile_forecasts(quantile_dfs, scaler):
    """Unscale quantile forecasts using the provided scaler"""
    unscaled_dfs = []
    
    for df in quantile_dfs:
        # Create empty DataFrame to store unscaled results
        df_unscaled = pd.DataFrame(index=df.index)
        
        # Unscale each quantile column individually
        for col in df.columns:
            # Create TimeSeries from the quantile column
            ts = TimeSeries.from_series(df[col])
            
            # Inverse transform using the scaler
            ts_unscaled = scaler.inverse_transform(ts)
            
            # Store back in DataFrame
            df_unscaled[col] = ts_unscaled.pd_series()
        
        unscaled_dfs.append(df_unscaled)
    
    return unscaled_dfs


def generate_xgb_forecasts(
    xgb_model, 
    afrr_pr_ts_scl_train, 
    afrr_pr_ts_scl_test, 
    exog_ts_scl_train, 
    exog_ts_scl_test, 
    forecast_horizon, 
    stride, 
    afrr_pr_scaler
):
    """
    Generate forecasts with quantile outputs and return a single DataFrame with unscaled quantiles
    """
    # Generate backtests
    xgb_backtest_forecasts = xgb_model.historical_forecasts(
        series=concatenate([afrr_pr_ts_scl_train, afrr_pr_ts_scl_test], axis=0),
        past_covariates=concatenate([exog_ts_scl_train, exog_ts_scl_test], axis=0),
        start=1,  
        forecast_horizon=forecast_horizon,
        stride=stride,
        retrain=False,
        last_points_only=False,  
        verbose=True
    )
    
    # Convert forecasts to DataFrame
    forecast_df = xgb_backtest_forecasts.pd_dataframe()
    
    # Unscale the quantile forecasts
    unscaled_dfs = []
    for col in forecast_df.columns:
        ts = TimeSeries.from_series(forecast_df[col])
        ts_unscaled = afrr_pr_scaler.inverse_transform(ts)
        unscaled_dfs.append(ts_unscaled.pd_series())
    
    # Combine into final DataFrame
    final_df = pd.concat(unscaled_dfs, axis=1)
    final_df.columns = ['quantile_0.1', 'quantile_0.5', 'quantile_0.9']
    
    return final_df

def run_xgb_pipeline(data_path, hyper_params_path, train_start, test_start, 
                    test_end, output_chunk_length, forecast_horizon, stride):
    """Run the complete XGBoost forecasting pipeline"""
    
    # Preprocess data
    (afrr_pr_ts_scl_train, 
     afrr_pr_ts_scl_test, 
     afrr_pr_ts_orig_train, 
     afrr_pr_ts_orig_test, 
     exog_ts_scl_train, 
     exog_ts_scl_test,
     afrr_pr_scaler) = preprocess_afrr_data(data_path, train_start, test_start, test_end)
    
    # Load hyperparameters
    xgb_opt_params = load_hyperparameters(file_path=hyper_params_path)
    
    # Train XGBoost model
    xgb_model = train_xgb_model(
        afrr_pr_ts_scl_train, 
        exog_ts_scl_train, 
        xgb_opt_params, 
        output_chunk_length
    )
    
    # Generate and unscale forecasts
    xgb_results = generate_xgb_forecasts(
        xgb_model,
        afrr_pr_ts_scl_train,
        afrr_pr_ts_scl_test,
        exog_ts_scl_train,
        exog_ts_scl_test,
        forecast_horizon,
        stride,
        afrr_pr_scaler
    )
    
    return xgb_model, xgb_results


if __name__ == "__main__":
    # Define all parameters here at the end
    data_path = "./data/afrr_price.parquet"
    hyper_params_path = "./data/results/xgb_hp_results.json"
    train_start = "2024-10-01 22:00:00"
    test_start = "2025-01-09 22:00:00"
    test_end = "2025-03-20 22:00:00"
    
    # Get forecast parameters
    forecast_params = get_forecast_params()
    output_chunk_length = forecast_params['output_chunk_length']
    forecast_horizon = forecast_params['forecast_horizon']
    stride = forecast_params['stride']
    
    # Run the pipeline with all parameters defined at the end
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
    
    # Print information about the results
    print("XGBoost forecasting completed.")
    print(f"Number of forecast windows: {len(xgb_results)}")
    print("Example forecast window (unscaled quantiles):")
    print(xgb_results[0].head())