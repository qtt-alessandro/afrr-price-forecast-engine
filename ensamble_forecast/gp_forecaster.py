import pandas as pd
from scipy import stats
from darts import TimeSeries
from darts.timeseries import concatenate
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from models.gp_regressor import GPRegressor
from utils.afrr_preprocessing import preprocess_afrr_data
from utils.forecast_utils import load_hyperparameters, get_forecast_params


def train_gp_model(afrr_pr_ts_scl_train, exog_ts_scl_train, gp_params, output_chunk_length):
    """
    Train Gaussian Process model
    
    Args:
        afrr_pr_ts_scl_train: Scaled training data for AFRR price
        exog_ts_scl_train: Scaled training data for exogenous variables
        gp_params: Hyperparameters for GP model
        output_chunk_length: Length of forecast output chunks
        
    Returns:
        Trained GP model
    """
    kernel = DotProduct() + WhiteKernel()
    
    gp_model = GPRegressor(
        lags=gp_params["lags"],
        lags_past_covariates=gp_params["lags_past_covariates"],
        output_chunk_length=output_chunk_length,
        kernel=kernel
    )

    gp_model.fit(afrr_pr_ts_scl_train, past_covariates=exog_ts_scl_train)
    
    return gp_model


def generate_gp_forecasts(
    gp_model, 
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
    Generate forecasts using the GP model and calculate quantiles
    
    Args:
        gp_model: Trained GP model
        afrr_pr_ts_scl_train, afrr_pr_ts_scl_test: Scaled training and test data for AFRR price
        exog_ts_scl_train, exog_ts_scl_test: Scaled training and test data for exogenous variables
        forecast_horizon: Horizon for forecasting
        stride: Stride for historical forecasts
        quantiles: List of quantiles to calculate
        afrr_pr_scaler: Scaler for inverse transformation
        
    Returns:
        DataFrame with forecast results
    """
    # Generate backtests
    gp_backtest_forecasts = gp_model.historical_forecasts(
        series=concatenate([afrr_pr_ts_scl_train, afrr_pr_ts_scl_test], axis=0),
        past_covariates=concatenate([exog_ts_scl_train, exog_ts_scl_test], axis=0),
        start=1,  
        forecast_horizon=forecast_horizon,
        enable_optimization=True,
        num_samples=1,
        predict_likelihood_parameters=True,    
        stride=stride,
        retrain=False,
        last_points_only=False,
        verbose=True
    )

    # Rename columns in the forecasts
    gp_backtest_forecasts = concatenate(gp_backtest_forecasts).with_columns_renamed(
        ['aFRR_UpCapPriceEUR_cl_mu', 'aFRR_UpCapPriceEUR_cl_sigma'],
        col_names_new=['gp_afrr_up_cap_price_mu', 'gp_afrr_up_cap_price_sigma']
    ).pd_dataframe()

    # Calculate quantiles
    gp_df_hat = pd.DataFrame(index=gp_backtest_forecasts.index)

    for q in quantiles:
        gp_df_hat[f'gp_afrr_up_cap_price_{q}'] = afrr_pr_scaler.inverse_transform(
            TimeSeries.from_series(
                gp_backtest_forecasts['gp_afrr_up_cap_price_mu'] + 
                stats.norm.ppf(q) * gp_backtest_forecasts['gp_afrr_up_cap_price_sigma']
            )
        ).pd_series()
        
    return gp_df_hat


def run_gp_pipeline(
    data_path,
    hyper_params_path,
    train_start,
    test_start,
    test_end,
    output_chunk_length,
    forecast_horizon,
    stride,
    quantiles
):
    """
    Run the full Gaussian Process pipeline
    
    Args:
        data_path: Path to the data file
        hyper_params_path: Path to hyperparameters file
        train_start: Start date for training data
        test_start: Start date for test data
        test_end: End date for test data
        output_chunk_length: Length of forecast output chunks
        forecast_horizon: Horizon for forecasting
        stride: Stride for historical forecasts
        quantiles: List of quantiles to calculate
        
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
    gp_opt_params = load_hyperparameters(file_path=hyper_params_path)
    
    # Train GP model
    gp_model = train_gp_model(
        afrr_pr_ts_scl_train, 
        exog_ts_scl_train, 
        gp_opt_params, 
        output_chunk_length
    )
    
    # Generate forecasts
    gp_results = generate_gp_forecasts(
        gp_model,
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test,
        forecast_horizon,
        stride,
        quantiles,
        afrr_pr_scaler
    )
    
    return gp_model, gp_results


if __name__ == "__main__":
    # Define all parameters here at the end
    data_path = "./data/afrr_price.parquet"
    hyper_params_path = "./data/results/gp_hp_results.json"
    train_start = "2024-10-01 22:00:00"
    test_start = "2025-01-09 22:00:00"
    test_end = "2025-03-20 22:00:00"
    
    # Get forecast parameters
    forecast_params = get_forecast_params()
    output_chunk_length = forecast_params['output_chunk_length']
    forecast_horizon = forecast_params['forecast_horizon']
    stride = forecast_params['stride']
    quantiles = forecast_params['quantiles']
    
    # Run the pipeline with all parameters defined at the end
    gp_model, gp_results = run_gp_pipeline(
        data_path=data_path,
        hyper_params_path=hyper_params_path,
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
        output_chunk_length=output_chunk_length,
        forecast_horizon=forecast_horizon,
        stride=stride,
        quantiles=quantiles
    )
    
    print("GP forecasting completed. Results shape:", gp_results.shape)