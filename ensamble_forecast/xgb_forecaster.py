import pandas as pd
from darts import TimeSeries
from darts.timeseries import concatenate
from darts.models import XGBModel
from utils.forecast_utils import load_hyperparameters, get_forecast_params
from utils.afrr_preprocessing import preprocess_afrr_data


def train_xgb_model(afrr_pr_ts_scl_train, exog_ts_scl_train, xgb_params, output_chunk_length):
    """Train XGBoost model without quantile regression"""
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
        # Removed likelihood="quantile" and quantiles parameters
    )

    xgb_model.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
    
    return xgb_model


def unscale_forecast(forecast_ts, scaler):
    """Unscale forecast using the provided scaler"""
    return scaler.inverse_transform(forecast_ts)


def generate_xgb_forecasts(xgb_model, afrr_pr_ts_scl_train, afrr_pr_ts_scl_test, exog_ts_scl_train, exog_ts_scl_test, forecast_horizon, stride, afrr_pr_scaler):

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
    
    forecasts_unscaled = unscale_forecast(concatenate(xgb_backtest_forecasts), afrr_pr_scaler)
    
    final_df = forecasts_unscaled.with_columns_renamed(['aFRR_UpCapPriceEUR_cl'], col_names_new=['xgb_afrr_up_cap_price']).to_dataframe()

    return final_df


def run_xgb_pipeline(data_path, target_col, hyper_params_path, train_start, test_start, test_end, output_chunk_length, forecast_horizon, stride):
    """Run the complete XGBoost forecasting pipeline"""
    
    (afrr_pr_ts_scl_train, 
    afrr_pr_ts_scl_test, 
    afrr_pr_ts_orig_train, 
    afrr_pr_ts_orig_test, 
    exog_ts_scl_train, 
    exog_ts_scl_test,
    afrr_pr_scaler) = preprocess_afrr_data(data_path, train_start, test_start, test_end, val_start=None, use_validation=False, target_col=target_col)

    
    xgb_opt_params = load_hyperparameters(file_path=hyper_params_path)
    
    xgb_model = train_xgb_model(
        afrr_pr_ts_scl_train, 
        exog_ts_scl_train, 
        xgb_opt_params, 
        output_chunk_length)
    
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

"""
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
    print("Example forecast window (unscaled):")
    print(xgb_results.head())

"""
