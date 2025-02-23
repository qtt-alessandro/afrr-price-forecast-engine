import numpy as np 
import pandas as pd 

def create_lag_features(y, X=None, input_chunk_length=1, future_covariates_length=0):
    if isinstance(y, pd.Series):
        y = y.to_frame(name='y')
    
    # Create empty result DataFrames with same index as y
    past_lags = pd.DataFrame(index=y.index)
    future_covariates = pd.DataFrame(index=y.index) if future_covariates_length > 0 else None
    
    # Create past lags of target variable
    for lag in range(1, input_chunk_length + 1):
        for col in y.columns:
            past_lags[f'{col}_lag_{lag}'] = y[col].shift(lag)
    
    # Add past lags of covariates if provided
    if X is not None:
        for lag in range(1, input_chunk_length + 1):
            for col in X.columns:
                past_lags[f'{col}_lag_{lag}'] = X[col].shift(lag)
        
        # Add future covariates if requested
        if future_covariates_length > 0:
            for lag in range(1, future_covariates_length + 1):
                for col in X.columns:
                    future_covariates[f'{col}_future_{lag}'] = X[col].shift(-lag)
    
    return past_lags, future_covariates