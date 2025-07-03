# Import Libraries
import pandas as pd
import numpy as np

# FRED
from fredapi import Fred

# Handle Files
import sys
import os

# Import Local Functions
sys.path.append(os.path.abspath("../source"))
from config import get_api_key


# Data Collection Function from FRED
def get_fred_data(
        symbol: str,
) -> pd.DataFrame:
    # Key to access the API
    key = get_api_key(provider='fred')

    # Access
    fred = Fred(api_key=key)

    # DataFrame
    df = fred.get_series(symbol)

    return df


# Create the Weights Function
def wexp(N, half_life):
    c = np.log(0.5) / half_life
    n = np.array(range(N))
    w = np.exp(c * n)
    return np.flip(w / np.sum(w))


# Helper: exclude tiny returns
def n_days_nonmiss(
        returns,
        tiny_ret=1e-6
):
    ix_ret_tiny = np.abs(returns) <= tiny_ret
    return np.sum(~ix_ret_tiny, axis=0)


# Relative Strength Calculation
def calc_rstr(
        returns,
        half_life=126,
        min_obs=100,
):
    rstr = np.log(1. + returns)
    if half_life == 0:
        weights = np.ones_like(rstr)
    else:
        weights = len(returns) * np.asmatrix(wexp(len(returns), half_life)).T
    rstr = (rstr * weights).sum()
    idx = n_days_nonmiss(returns) < min_obs
    rstr.where(~idx, other=np.nan, inplace=True)
    df = pd.Series(rstr)
    df.name = returns.index[-1]
    return df


# Rolling Relative Strength
def rolling_calc_rstr(
        returns,
        window_size=252,
        half_life=126,
        min_obs=100
):
    rolling_results = []
    range_to_iter = range(len(returns) - window_size + 1)
    for i in range_to_iter:
        window_returns = returns.iloc[i:i + window_size]
        rs_i = calc_rstr(
            returns=window_returns,
            half_life=half_life,
            min_obs=min_obs
        )

        rolling_results.append(rs_i)

    return pd.concat(rolling_results, axis=1)
