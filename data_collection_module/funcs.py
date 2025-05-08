import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from copulas.multivariate import GaussianMultivariate
from copulas.univariate import BetaUnivariate, GaussianKDE, GaussianUnivariate
import pandas as pd
import requests

def fit_ar_garch(residuals, ar_order=1, garch_p=1, garch_q=1):
    """Fit AR and GARCH models to make residuals IID"""
    # Fit AR model
    ar_model = ARIMA(residuals, order=(ar_order,0,0), enforce_stationarity=True, enforce_invertibility=True, trend=None)
    ar_results = ar_model.fit()
    ar_residuals = ar_results.resid
    ar_params = ar_results.arparams
    
    garch_model = arch_model(ar_residuals, mean='Zero', vol='GARCH', p=garch_p, q=garch_q)
    garch_results = garch_model.fit(disp='off')
    std_residuals = garch_results.resid / garch_results.conditional_volatility
    
    garch_params = {
        'omega': garch_results.params['omega'],
        'alpha': garch_results.params['alpha[1]'],
        'beta': garch_results.params['beta[1]']
    }
    
    return ar_params, garch_params, std_residuals, garch_results.conditional_volatility

def reconstruct_residuals(std_residuals, ar_params, garch_params, ar_constant=0, initial_values=None, initial_variance=None):
    n = len(std_residuals)
    
    # Step 1: Revert GARCH transformation
    garch_residuals = np.zeros(n)
    variances = np.zeros(n)
    
    if initial_variance is None:
        variances[0] = garch_params['omega'] / (1 - garch_params['alpha'] - garch_params['beta'])
    else:
        variances[0] = initial_variance

    garch_residuals[0] = std_residuals[0] * np.sqrt(variances[0])
    
    for t in range(1, n):
        variances[t] = garch_params['omega'] + \
                      garch_params['alpha'] * garch_residuals[t-1]**2 + \
                      garch_params['beta'] * variances[t-1]
        garch_residuals[t] = std_residuals[t] * np.sqrt(variances[t])
    
    # Step 2: Revert AR transformation
    p = len(ar_params)  # AR order
    reconstructed = np.zeros(n)
    
    # Initialize with provided values or garch residuals
    if initial_values is not None:
        reconstructed[:p] = initial_values[:p]
    else:
        reconstructed[:p] = garch_residuals[:p]
    
    # Apply AR model to reconstruct
    for t in range(p, n):
        ar_component = ar_constant
        for i in range(p):
            ar_component += ar_params[i] * reconstructed[t-i-1]
        
        reconstructed[t] = ar_component + garch_residuals[t]
    
    return reconstructed



def sample_from_quantile_range(std_residuals, lower_quantile, upper_quantile, n_samples, 
                              ar_params, garch_params, initial_variance):

    # Get values at the quantile boundaries
    sorted_residuals = np.sort(std_residuals)
    n = len(sorted_residuals)
    
    # Create indices for the quantile range
    lower_idx = int(lower_quantile * n)
    upper_idx = int(upper_quantile * n)
    
    # Sample with replacement from this range
    sampled_indices = np.random.randint(lower_idx, upper_idx, n_samples)
    std_samples = sorted_residuals[sampled_indices]
    
    # Convert back to original scale
    reconstructed_samples = reconstruct_residuals(
        std_samples, 
        ar_params, 
        garch_params, 
        initial_variance
    )
    
    return reconstructed_samples


def fit_sample_copula(residual_df:pd.DataFrame,columns:list, distribution:dict, return_df=False):
    
    dist = GaussianMultivariate(distribution=distribution)	

    dist.fit(residual_df[columns])
    synthetic = dist.sample(len(residual_df))
    if return_df:
        return synthetic
    else:
        for i in range(len(columns)):
            residual_df[str(columns[i])+"_cop"] = synthetic.values[:,i]
        return residual_df




def get_imbalance_price(start_date, end_date, price_area="DK1", sort_by="TimeUTC ASC", imbalance_type="afrr"):
    
    COLS_SEL = ['TimeUTC', 'BalancingDemand','SpotPriceEUR','DominatingDirection']
    
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime("%Y-%m-%dT%H:%M")
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime("%Y-%m-%dT%H:%M")
    
    base_url = "https://api.energidataservice.dk/dataset/ImbalancePrice"
    
    url = f"{base_url}?offset=0&start={start_date}&end={end_date}&filter={{\"PriceArea\":[\"{price_area}\"]}}&sort={sort_by}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data.get("records", []))
        
        if imbalance_type == "afrr":
            COLS_SEL.extend(['aFRRUpMW', 'aFRRVWAUpEUR', 'aFRRDownMW', 'aFRRVWADownEUR'])
            return df[COLS_SEL].set_index('TimeUTC')
        elif imbalance_type == "mfrr":
            COLS_SEL.append(['mFRRVWAUpEUR', 'mFRRVWADownEUR'])
            return df[[COLS_SEL]].set_index('TimeUTC')
            
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
    
