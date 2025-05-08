#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
import logging
import numpy as np
import pandas as pd

from utils.ssa import mySSA
from darts import TimeSeries
from darts.dataprocessing import Pipeline
from darts.dataprocessing.transformers import MissingValuesFiller, Scaler

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


def load_and_prepare_data(data_path):

    data = pd.read_parquet(data_path)
    
    data = data[['wind_offshore_dayahead', 'wind_onshore_dayahead',
           'solar_dayahead','aFRR_DownCapPriceEUR', 'aFRR_UpCapPriceEUR', 'load_forecasts',
           'da_price', 'prod_unbalance']]
    
    data.index = data.index.tz_localize(None)
    
    return data


def apply_ssa_decomposition(data, target_col):

    ts = data[target_col]
    ssa_ts = mySSA(ts)
    ssa_ts.embed(embedding_dimension=128, suspected_frequency=24, verbose=True)
    ssa_ts.decompose(True)
    
    # Components to use for reconstruction (to tune for optimal decomposition)
    components = [i for i in range(15)]
    ts_clean = ssa_ts.view_reconstruction(*[ssa_ts.Xs[i] for i in components], names=components, return_df=True, plot=False)
    
    data[target_col + '_cl'] = ts_clean.values
    
    return data


def prepare_time_series(data, target_col):

    afrr_pr = target_col + '_cl'
    afrr_pr_orig = target_col
    
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


def split_data(afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl, train_start, test_start, test_end, val_start, use_validation=False):

    if isinstance(train_start, str):
        train_start = pd.Timestamp(train_start)
    if isinstance(test_start, str):
        test_start = pd.Timestamp(test_start)
    if isinstance(test_end, str):
        test_end = pd.Timestamp(test_end)
    
    if use_validation:
        if val_start is None:
            val_start = train_start + (test_start - train_start) / 2
        elif isinstance(val_start, str):
            val_start = pd.Timestamp(val_start)
        
        afrr_pr_ts_scl_train = afrr_pr_ts_scl[train_start : val_start - afrr_pr_ts_scl.freq]
        afrr_pr_ts_scl_val = afrr_pr_ts_scl[val_start : test_start - afrr_pr_ts_scl.freq]
        afrr_pr_ts_scl_test = afrr_pr_ts_scl[test_start : test_end]
        
        afrr_pr_ts_orig_train = afrr_pr_ts_orig[train_start : val_start - afrr_pr_ts_orig.freq]
        afrr_pr_ts_orig_val = afrr_pr_ts_orig[val_start : test_start - afrr_pr_ts_orig.freq]
        afrr_pr_ts_orig_test = afrr_pr_ts_orig[test_start : test_end]
        
        exog_ts_scl_train = exog_ts_scl[train_start : val_start - exog_ts_scl.freq]
        exog_ts_scl_val = exog_ts_scl[val_start : test_start - exog_ts_scl.freq]
        exog_ts_scl_test = exog_ts_scl[test_start : test_end]
        
        return (
            afrr_pr_ts_scl_train, 
            afrr_pr_ts_scl_val,
            afrr_pr_ts_scl_test, 
            afrr_pr_ts_orig_train,
            afrr_pr_ts_orig_val, 
            afrr_pr_ts_orig_test, 
            exog_ts_scl_train,
            exog_ts_scl_val, 
            exog_ts_scl_test
        )
    else:
        afrr_pr_ts_scl_train = afrr_pr_ts_scl[train_start : test_start - afrr_pr_ts_scl.freq]
        afrr_pr_ts_scl_test = afrr_pr_ts_scl[test_start : test_end]
        
        afrr_pr_ts_orig_train = afrr_pr_ts_orig[train_start : test_start - afrr_pr_ts_orig.freq]
        afrr_pr_ts_orig_test = afrr_pr_ts_orig[test_start : test_end]
        
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

def preprocess_afrr_data(data_path, train_start, test_start, test_end, val_start, use_validation, target_col):
    
    data = load_and_prepare_data(data_path)
    data = apply_ssa_decomposition(data, target_col)
    afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl, afrr_pr_scaler = prepare_time_series(data, target_col)
    splits = split_data(afrr_pr_ts_scl, afrr_pr_ts_orig, exog_ts_scl, train_start, test_start, test_end, val_start, use_validation)
    
    return (*splits, afrr_pr_scaler)




"""
if __name__ == "__main__":
    data_path = "./data/afrr_price.parquet"
    train_start = "2020-01-01"
    test_start = "2021-01-01"
    test_end = "2021-12-31"
    target_col = 'aFRR_DownCapPriceEUR'  # Specify your target column here
    
    afrr_pr_ts_scl_train, afrr_pr_ts_scl_test, afrr_pr_ts_orig_train, afrr_pr_ts_orig_test, exog_ts_scl_train, exog_ts_scl_test, afrr_pr_scaler = preprocess_afrr_data(
        data_path,
        train_start,
        test_start,
        test_end,
        target_col=target_col
    )
    
    print("Preprocessing complete.")

"""
