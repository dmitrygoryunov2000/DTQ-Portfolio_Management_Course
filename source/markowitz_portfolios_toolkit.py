import pandas as pd
import numpy as np


# Portfolio Variance Function
def portfolio_variance(
        weights,
        returns
):
    # Weights
    weights = np.array(weights)

    # Calculate Covariance Matrix
    cov_matrix = returns.cov()

    # Portfolio Variance
    port_var = weights.T @ cov_matrix @ weights

    return port_var


# Rolling Portfolio Variance Function
def rolling_portfolio_variance(
        returns_df,
        weights,
        window=252
):
    # Set weights as np.array
    weights = np.array(weights)

    # List to store data
    rolling_vars = []
    index = returns_df.index

    # Loop
    for i in range(window - 1, len(returns_df)):
        window_returns = returns_df.iloc[i - window + 1: i + 1]
        cov_matrix = np.cov(window_returns.T)
        var = weights.T @ cov_matrix @ weights
        rolling_vars.append(var)

    result = pd.Series([np.nan] * (window - 1) + rolling_vars, index=index)
    return result


# Helper: Preprocess inputs
def _preprocess_mkwz_inputs(
        expected_returns,
        covariance_matrix
):
    # Process Inputs
    mu = expected_returns.values.flatten().reshape(-1, 1)
    Sigma = covariance_matrix.values
    Sigma_inv = np.linalg.inv(Sigma)
    iota = np.ones_like(mu)

    return mu, Sigma, Sigma_inv, iota


# Helper: Compute Efficient Frontier components
def eff_components(
        expected_returns,
        covariance_matrix
):
    # Get Inputs
    mu, _, Sigma_inv, iota = _preprocess_mkwz_inputs(expected_returns, covariance_matrix)

    # Calculate components
    A = mu.T @ Sigma_inv @ mu
    B = iota.T @ Sigma_inv @ mu
    C = iota.T @ Sigma_inv @ iota
    D = (A * C) - (B * B)
    return A, B, C, D


# Helper: Get the Efficient Frontier Coefficients
def eff_coefficients(
        expected_returns,
        covariance_matrix
):
    # Get the components
    A, B, C, D = eff_components(expected_returns, covariance_matrix)

    # Calculate the coefficients
    pi_0 = A / D
    pi_1 = 2 * B / D
    pi_2 = C / D

    return pi_0.item(), pi_1.item(), pi_2.item()


# Function to get the efficient frontier
def eff_equation(
        coefficients,
        desired_returns
):
    # Get the coefficients
    pi_0, pi_1, pi_2 = coefficients

    # Set the desired returns
    mu_P = desired_returns

    return np.sqrt(pi_0 - pi_1 * mu_P + pi_2 * mu_P ** 2)


# Function to get the Markowitz Optimization Weights
def markowitz_weights(
        expected_returns,
        covariance_matrix,
        desired_returns
):
    # Inputs
    mu, _, Sigma_inv, iota = _preprocess_mkwz_inputs(expected_returns, covariance_matrix)

    # Components
    A, B, C, D = eff_components(expected_returns, covariance_matrix)

    # Calculate weights
    w_1 = ((desired_returns * C - B) / D) * (Sigma_inv @ mu)
    w_2 = ((A - desired_returns * B) / D) * (Sigma_inv @ iota)

    return (w_1 + w_2).flatten()


def rolling_markowitz_weights(
        returns,
        desired_returns,
        window=252,
        rebalance_freq=126
):
    # Lists to Store Things
    weights_list = []
    dates = []

    for i in range(window, len(returns), rebalance_freq):
        # Prepare Inputs
        past_returns = returns.iloc[i - window:i]  # Rolling Window
        past_excepted_returns = past_returns.mean()
        past_cov_matrix = past_returns.cov()

        # Calculate Weights
        w = markowitz_weights(past_excepted_returns, past_cov_matrix, desired_returns)

        # Save weights and dates
        weights_list.append(w)
        dates.append(returns.index[i])

    # Create the DataFrame
    weights_df = pd.DataFrame(weights_list, index=dates, columns=returns.columns)

    # Expand the DataFrame
    weights_df = weights_df.reindex(returns.index, method='ffill')

    return weights_df.dropna()


# Function to get the CAL Optimization Weights
def cal_weights(
        expected_returns,
        covariance_matrix,
        tangency_returns,
        desired_returns,
        risk_free_rate,
):
    # Calculate Tangents Weights
    tan_ws = markowitz_weights(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        desired_returns=tangency_returns
    )

    # Calculate discount factor
    discount_factor = (desired_returns - risk_free_rate) / (tangency_returns - risk_free_rate)

    # Calculate weights
    cal_ws = tan_ws * discount_factor

    return cal_ws


# Function to get the CAL volatility for a desired level of returns
def cal_volatility(
        risk_free_rate,
        sharpe_ratio,
        desired_returns
):
    # Calculate the volatility
    sigma = (desired_returns - risk_free_rate) / sharpe_ratio

    return sigma


# Function to get the CAL returns for a desired level of volatility
def cal_equation(
        risk_free_rate,
        sharpe_ratio,
        sigma_P
):
    return risk_free_rate + sharpe_ratio * sigma_P
