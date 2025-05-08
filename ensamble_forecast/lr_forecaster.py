import pandas as pd
from sklearn.linear_model import LinearRegression
from darts import TimeSeries
from darts.timeseries import concatenate
from darts.models import RegressionModel
from utils.forecast_utils import load_hyperparameters, get_forecast_params
from utils.afrr_preprocessing import preprocess_afrr_data

def train_lr_model(afrr_pr_ts_scl_train, exog_ts_scl_train, lr_params, output_chunk_length):

    linear_regressor = LinearRegression()
    
    lr_model = RegressionModel(
        output_chunk_length=output_chunk_length,
        lags=list(range(-1, -lr_params["lags_max"], -1)),
        lags_past_covariates=list(range(-1, -lr_params["lags_past_covariates_max"], -1)), 
        model=linear_regressor
    )
    
    lr_model.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
    
    return lr_model

def generate_lr_forecast(lr_model, target_col, afrr_pr_ts_scl_train, afrr_pr_ts_scl_test, exog_ts_scl_train, exog_ts_scl_test, forecast_horizon, stride, afrr_pr_scaler):

    lr_backtest_forecast = lr_model.historical_forecasts(
        series=concatenate([afrr_pr_ts_scl_train, afrr_pr_ts_scl_test], axis=0),
        past_covariates=concatenate([exog_ts_scl_train, exog_ts_scl_test], axis=0),
        start=1,  
        forecast_horizon=forecast_horizon,
        enable_optimization=True,
        stride=stride,
        retrain=False,
        last_points_only=False,
        verbose=True)
    
    if target_col == 'aFRR_UpCapPriceEUR':
        lr_df_hat = afrr_pr_scaler.inverse_transform(
            concatenate(lr_backtest_forecast).with_columns_renamed(
                ['aFRR_UpCapPriceEUR_cl'], 
                col_names_new=[f'lr_afrr_up_cap_price'])).to_dataframe()
    elif target_col == 'aFRR_DownCapPriceEUR':
        lr_df_hat = afrr_pr_scaler.inverse_transform(
            concatenate(lr_backtest_forecast).with_columns_renamed(
                ['aFRR_DownCapPriceEUR_cl'], 
                col_names_new=[f'lr_afrr_down_cap_price'])).to_dataframe()
    else:
        raise ValueError(f"Unknown target column: {target_col}")

    return lr_df_hat

def run_lr_pipeline(data_path, target_col, hyper_params_path, train_start, test_start, test_end, output_chunk_length, forecast_horizon, stride):


    (afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        afrr_pr_ts_orig_train, 
        afrr_pr_ts_orig_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test,
        afrr_pr_scaler) = preprocess_afrr_data(data_path, train_start, test_start, test_end, val_start=None, use_validation=False, target_col=target_col)

    lr_opt_params = load_hyperparameters(file_path=hyper_params_path)
    lr_model = train_lr_model(afrr_pr_ts_scl_train, exog_ts_scl_train, lr_opt_params, output_chunk_length)
    lr_results = generate_lr_forecast(lr_model, target_col, afrr_pr_ts_scl_train, afrr_pr_ts_scl_test, exog_ts_scl_train, exog_ts_scl_test,forecast_horizon,stride,afrr_pr_scaler)
    
    return lr_model, lr_results



"""
if __name__ == "__main__":
    data_path = "./data/afrr_price.parquet"
    hyper_params_path = "./data/results/lr_hp_results.json"
    train_start = "2024-10-01 22:00:00"
    test_start = "2025-01-09 22:00:00"
    test_end = "2025-03-20 22:00:00"
    
    forecast_params = get_forecast_params()
    output_chunk_length = forecast_params['output_chunk_length']
    forecast_horizon = forecast_params['forecast_horizon']
    stride = forecast_params['stride']
    
    lr_model, lr_results = run_lr_pipeline(
        data_path=data_path,
        target_col='aFRR_DownCapPriceEUR',
        hyper_params_path=hyper_params_path,
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
        output_chunk_length=output_chunk_length,
        forecast_horizon=forecast_horizon,
        stride=stride
    )
    
    print("LR forecasting completed. Results shape:", lr_results.shape)
"""
