# Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Handle Files
import sys
import os

# Import Local Functions
sys.path.append(os.path.abspath("../source"))
from other_data_functions import wexp


# Helper: Add a constant
def add_constant(
        x_matrix: pd.DataFrame,
):
    # Create vector of ones
    ones = pd.Series(1, index=x_matrix.index, name="constant")

    x_matrix_with_constant = pd.concat([ones, x_matrix], axis=1)

    return x_matrix_with_constant


def wls_regression(
        y_matrix: pd.DataFrame,
        x_matrix: pd.DataFrame,
        weights,
):
    # Check if both arrays have the same rows
    if x_matrix.shape[0] != y_matrix.shape[0]:
        raise ValueError("The rows are not coincident.")

    # Set the components
    X = x_matrix
    Y = y_matrix
    W = np.diag(weights)

    # Weighted Arrays
    Weighted_X = W.dot(X)
    Weighted_Y = W.dot(Y)

    # Calculate the interaction arrays
    X_T = X.transpose()
    X_Weighted_Var = X_T.dot(Weighted_X)
    X_Y_Weighted_Covar = X_T.dot(Weighted_Y)
    X_Weighted_Var_Inv = np.linalg.inv(X_Weighted_Var)

    # Coefficients
    coef = X_Weighted_Var_Inv.dot(X_Y_Weighted_Covar)

    # Fitted values and residuals
    fitted = X.dot(coef)
    residuals = Y.to_numpy() - fitted

    # Sigmas
    stds = residuals.std(axis=0, ddof=1).to_numpy()

    # Output Series
    alphas = pd.Series(coef[0], index=Y.columns, name='alpha')
    betas = pd.Series(coef[1], index=Y.columns, name='beta')
    sigmas = pd.Series(stds, index=Y.columns, name='sigma')

    return alphas, betas, sigmas


# WLS Rolling Coefficients
def rolling_wls_regression(
        y_matrix: pd.DataFrame,
        x_matrix: pd.DataFrame,
        window: int = 252
):
    # Define lookback
    lookback = window

    # Trimmed Returns
    trimmed_y_matrix = y_matrix.iloc[lookback - 1:]

    # Define the dates
    dates = trimmed_y_matrix.index

    # Calculate weights
    weights = 252 * wexp(window, window / 2)

    # List to store data
    alphas_list = []
    betas_list = []
    sigmas_list = []

    # Loop
    for date in dates:

        # Set the windows
        x_window = x_matrix.loc[:date].iloc[-lookback:]
        y_window = y_matrix.loc[:date].iloc[-lookback:]

        # Select Valid Stocks (those with enough data)
        valid_stocks = y_window.count()[y_window.count() >= lookback].index
        if len(valid_stocks) < 2:
            continue

        # Calculate the components for the optimization
        valid_y_window = y_window[valid_stocks]

        # Optimization
        try:
            alphas, betas, sigmas = wls_regression(
                valid_y_window,
                x_window,
                weights
            )

            alphas.name = date
            betas.name = date
            sigmas.name = date

            alphas_list.append(alphas)
            betas_list.append(betas)
            sigmas_list.append(sigmas)

        except ValueError as e:
            print(f"Fail in {date}: {e}")
            continue

    # DF for alphas
    alphas_df = pd.DataFrame(alphas_list).reindex(columns=trimmed_y_matrix.columns)
    alphas_df = alphas_df.reindex(trimmed_y_matrix.index)

    # DF for betas
    betas_df = pd.DataFrame(betas_list).reindex(columns=trimmed_y_matrix.columns)
    betas_df = betas_df.reindex(trimmed_y_matrix.index)

    # DF for sigmas
    sigmas_df = pd.DataFrame(sigmas_list).reindex(columns=trimmed_y_matrix.columns)
    sigmas_df = sigmas_df.reindex(trimmed_y_matrix.index)

    coefficients = {
        'alphas': alphas_df,
        'betas': betas_df,
        'sigmas': sigmas_df,
    }

    return coefficients
