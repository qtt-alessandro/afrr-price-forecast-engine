import pandas as pd
from darts import TimeSeries
from darts.timeseries import concatenate
from darts.models import XGBModel
from utils.forecast_utils import load_hyperparameters, get_forecast_params
from utils.afrr_preprocessing import preprocess_afrr_data


def train_xgb_model(afrr_pr_ts_scl_train, exog_ts_scl_train, xgb_params, output_chunk_length):
    """
    Train XGBoost model
    
    Args:
        afrr_pr_ts_scl_train: Scaled training data for AFRR price
        exog_ts_scl_train: Scaled training data for exogenous variables
        xgb_params: Hyperparameters for XGBoost model
        output_chunk_length: Length of forecast output chunks
        
    Returns:
        Trained XGBoost model
    """
    # Define time series encoders for XGBoost
    ts_encoders = {
        'cyclic': {'future': ['month']},
        'datetime_attribute': {'future': ['hour', 'dayofweek']},
        'position': {'past': ['relative'], 'future': ['relative']},
        'tz': 'UTC'
    }
    
    # Create XGBoost model with the loaded parameters
    xgb_model = XGBModel(
        lags=xgb_params["lags"],
        lags_past_covariates=xgb_params["lags_past_covariates"],
        add_encoders=ts_encoders,
        output_chunk_length=output_chunk_length,
        max_depth=xgb_params["max_depth"],
        learning_rate=xgb_params["learning_rate"],
        n_estimators=xgb_params["n_estimators"]
    )

    xgb_model.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
    
    return xgb_model


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
    Generate forecasts using the XGBoost model
    
    Args:
        xgb_model: Trained XGBoost model
        afrr_pr_ts_scl_train, afrr_pr_ts_scl_test: Scaled training and test data for AFRR price
        exog_ts_scl_train, exog_ts_scl_test: Scaled training and test data for exogenous variables
        forecast_horizon: Horizon for forecasting
        stride: Stride for historical forecasts
        afrr_pr_scaler: Scaler for inverse transformation
        
    Returns:
        DataFrame with forecast results
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

    xgb_df_hat = afrr_pr_scaler.inverse_transform(concatenate(xgb_backtest_forecasts)).pd_dataframe()
    xgb_df_hat.columns = ['xgb_afrr_up_cap_price']
    
    return xgb_df_hat


def run_xgb_pipeline(
    data_path,
    hyper_params_path,
    train_start,
    test_start,
    test_end,
    output_chunk_length,
    forecast_horizon,
    stride
):
    """
    Run the full XGBoost pipeline
    
    Args:
        data_path: Path to the data file
        hyper_params_path: Path to hyperparameters file
        train_start: Start date for training data
        test_start: Start date for test data
        test_end: End date for test data
        output_chunk_length: Length of forecast output chunks
        forecast_horizon: Horizon for forecasting
        stride: Stride for historical forecasts
        
    Returns:
        Tuple containing the trained model and forecast results
    """
    # Load data with specified date ranges
    (
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        afrr_pr_ts_orig_train, 
        afrr_pr_ts_orig_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test,
        afrr_pr_scaler
    ) = preprocess_afrr_data(data_path, train_start, test_start, test_end)
    
    # Load hyperparameters from specified path
    xgb_opt_params = load_hyperparameters(file_path=hyper_params_path)
    
    # Train XGBoost model
    xgb_model = train_xgb_model(
        afrr_pr_ts_scl_train, 
        exog_ts_scl_train, 
        xgb_opt_params, 
        output_chunk_length
    )
    
    # Generate forecasts
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
    
    print("XGBoost forecasting completed. Results shape:", xgb_results.shape)