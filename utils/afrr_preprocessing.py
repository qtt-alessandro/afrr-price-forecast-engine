#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aFRR Price Forecasting - Preprocessing Module

This module handles data loading, preparation, and preprocessing for aFRR price forecasting.
It can be used as a standalone module in Jupyter notebooks or imported by other modules.
"""

# Standard libraries
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

# Data manipulation libraries
import numpy as np
import pandas as pd

# Custom modules
from utils.ssa import mySSA
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler


def load_and_prepare_data(data_path=None):
    """
    Load and prepare the aFRR price data.
    
    Args:
        data_path (str): Path to the parquet file containing aFRR price data
        
    Returns:
        pd.DataFrame: Prepared dataframe with aFRR price data
    """
    data = pd.read_parquet(data_path)
    
    data = data[['wind_offshore_dayahead', 'wind_onshore_dayahead',
           'solar_dayahead','aFRR_DownCapPriceEUR', 'aFRR_UpCapPriceEUR', 'load_forecasts',
           'da_price', 'prod_unbalance']]
    
    data.index = data.index.tz_localize(None)
    
    return data


def apply_ssa_decomposition(data):
    """
    Apply Singular Spectrum Analysis (SSA) decomposition to the target variable.
    
    Args:
        data (pd.DataFrame): DataFrame containing aFRR price data
        
    Returns:
        pd.DataFrame: DataFrame with added cleaned aFRR price column
    """
    ts = data["aFRR_UpCapPriceEUR"]
    ssa_ts = mySSA(data["aFRR_UpCapPriceEUR"])
    ssa_ts.embed(embedding_dimension=128, suspected_frequency=24, verbose=True)
    ssa_ts.decompose(True)
    
    # Components to use for reconstruction (to tune for optimal decomposition)
    components = [i for i in range(15)]
    ts_clean = ssa_ts.view_reconstruction(*[ssa_ts.Xs[i] for i in components], names=components, return_df=True, plot=False)
    
    data["aFRR_UpCapPriceEUR_cl"] = ts_clean.values
    
    return data


def prepare_time_series(data):
    """
    Convert data to Darts TimeSeries objects and preprocess.
    
    Args:
        data (pd.DataFrame): DataFrame containing aFRR price data
        
    Returns:
        tuple: Tuple containing:
            - afrr_pr_ts_scl (TimeSeries): Scaled target series
            - afrr_pr_ts_orig (TimeSeries): Original target series
            - exog_ts_scl (TimeSeries): Scaled exogenous variables
            - afrr_pr_scaler (Pipeline): Scaler for target variable
    """
    afrr_pr = 'aFRR_UpCapPriceEUR_cl'
    afrr_pr_orig = 'aFRR_UpCapPriceEUR'
    
    exog_cols = ['wind_offshore_dayahead', 'wind_onshore_dayahead', 
                'solar_dayahead', 'load_forecasts', 'da_price', 'prod_unbalance']
    
    afrr_pr_ts = TimeSeries.from_series(data[afrr_pr], freq="1h")
    afrr_pr_ts_orig = TimeSeries.from_series(data[afrr_pr_orig], freq="1h")
    exog_ts = TimeSeries.from_dataframe(data[exog_cols], freq="1h")
    
    # Setup preprocessing pipelines
    scaler_target_ts = Scaler()
    scaler_exog_ts = Scaler()
    filler_target_ts = MissingValuesFiller()
    filler_exog_ts = MissingValuesFiller()
    
    afrr_pr_scaler = Pipeline([scaler_target_ts])
    exog_ts_preprocess = Pipeline([scaler_exog_ts])
    
    # Apply preprocessing
    afrr_pr_ts_filled = filler_target_ts.transform(afrr_pr_ts)
    exog_ts_filled = filler_exog_ts.transform(exog_ts)
    
    afrr_pr_ts_scl = afrr_pr_scaler.fit_transform(afrr_pr_ts_filled)
    exog_ts_scl = exog_ts_preprocess.fit_transform(exog_ts_filled)
    
    return afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl, afrr_pr_scaler


def split_data(afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl, 
               train_start="2024-10-01 22:00:00", 
               test_start="2025-01-09 22:00:00", 
               test_end="2025-02-20 22:00:00"):
    """
    Split data into training and test sets.
    
    Args:
        afrr_pr_ts_scl (TimeSeries): Scaled target series
        afrr_pr_ts_orig (TimeSeries): Original target series
        exog_ts_scl (TimeSeries): Scaled exogenous variables
        train_start (str): Start date for training data
        test_start (str): Start date for test data
        test_end (str): End date for test data
        
    Returns:
        tuple: Tuple containing split datasets:
            - afrr_pr_ts_scl_train (TimeSeries): Training target scaled
            - afrr_pr_ts_scl_test (TimeSeries): Test target scaled
            - afrr_pr_ts_orig_train (TimeSeries): Training target original
            - afrr_pr_ts_orig_test (TimeSeries): Test target original
            - exog_ts_scl_train (TimeSeries): Training exogenous scaled
            - exog_ts_scl_test (TimeSeries): Test exogenous scaled
    """
    # Convert string to timestamps if provided as strings
    if isinstance(train_start, str):
        train_start = pd.Timestamp(train_start)
    if isinstance(test_start, str):
        test_start = pd.Timestamp(test_start)
    if isinstance(test_end, str):
        test_end = pd.Timestamp(test_end)
    
    # Split target series
    afrr_pr_ts_scl_train = afrr_pr_ts_scl[train_start : test_start - afrr_pr_ts_scl.freq]
    afrr_pr_ts_scl_test = afrr_pr_ts_scl[test_start : test_end]
    
    afrr_pr_ts_orig_train = afrr_pr_ts_orig[train_start : test_start - afrr_pr_ts_scl.freq]
    afrr_pr_ts_orig_test = afrr_pr_ts_orig[test_start : test_end]
    
    # Split exogenous series
    exog_ts_scl_train = exog_ts_scl[train_start : test_start - exog_ts_scl.freq]
    exog_ts_scl_test = exog_ts_scl[test_start : test_end]
    
    return (
        afrr_pr_ts_scl_train, 
        afrr_pr_ts_scl_test, 
        afrr_pr_ts_orig_train, 
        afrr_pr_ts_orig_test, 
        exog_ts_scl_train, 
        exog_ts_scl_test
    )


def preprocess_afrr_data(data_path=None,
                         train_start="2024-10-01 22:00:00", 
                         test_start="2025-01-09 22:00:00", 
                         test_end="2025-03-20 22:00:00"):
    """
    Complete preprocessing pipeline for aFRR data.
    
    Args:
        data_path (str): Path to the parquet file containing aFRR price data
        train_start (str): Start date for training data
        test_start (str): Start date for test data
        test_end (str): End date for test data
        
    Returns:
        tuple: Tuple containing all necessary data for modeling:
            - afrr_pr_ts_scl_train (TimeSeries): Training target scaled
            - afrr_pr_ts_scl_test (TimeSeries): Test target scaled
            - afrr_pr_ts_orig_train (TimeSeries): Training target original
            - afrr_pr_ts_orig_test (TimeSeries): Test target original
            - exog_ts_scl_train (TimeSeries): Training exogenous scaled
            - exog_ts_scl_test (TimeSeries): Test exogenous scaled
            - afrr_pr_scaler (Pipeline): Scaler for target variable
    """
    # Load and prepare data
    data = load_and_prepare_data(data_path)
    
    # Apply SSA decomposition
    data = apply_ssa_decomposition(data)
    
    # Prepare time series
    afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl, afrr_pr_scaler = prepare_time_series(data)
    
    # Split data
    splits = split_data(
        afrr_pr_ts_scl, 
        afrr_pr_ts_orig, 
        exog_ts_scl,
        train_start,
        test_start,
        test_end
    )
    
    # Return all components needed for modeling
    return (*splits, afrr_pr_scaler)


if __name__ == "__main__":
    # Example of using the module for preprocessing
    print("Preprocessing aFRR data...")
    data = load_and_prepare_data(data_path='/home/alqua/git/boiler_OptiBid/data/afrr_price.parquet')
    data = apply_ssa_decomposition(data)
    afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl, afrr_pr_scaler = prepare_time_series(data)
    
    print("Data preprocessing completed.")
    print("Target time series shape:", afrr_pr_ts_scl.shape)
    print("Exogenous time series shape:", exog_ts_scl.shape)