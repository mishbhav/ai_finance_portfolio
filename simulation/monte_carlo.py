import numpy as np
import pandas as pd

def simulate_price_paths(
    returns: pd.Series,
    num_simulations: int = 1000,
    num_days: int = 252,
    initial_value: float = 100.0,
) -> np.ndarray:
    """
    Simulates possible future value paths using Monte Carlo, sampling
    daily log returns from a normal distribution fitted to the historical
    mean/std of the portfolio's own log returns.

    Returns: array of shape (num_simulations, num_days) — simulated values.
    """
    log_returns = np.log(1 + returns.dropna())
    mu = log_returns.mean()
    sigma = log_returns.std()

    # 1. Draw random daily log returns: shape (num_simulations, num_days),
    #    using np.random.normal(loc=mu, scale=sigma, size=(num_simulations, num_days))
    daily_log_returns = np.random.normal(loc=mu, scale=sigma, size=(num_simulations, num_days))
    
    # 2. Cumulative-sum across the days axis (axis=1) to get cumulative log return
    cumulative_log_returns = np.cumsum(daily_log_returns, axis=1)
    # 3. Exponentiate (np.exp) to convert back to a growth factor, multiply by initial_value
    paths = initial_value * np.exp(cumulative_log_returns)

    return paths

def summarize_simulation(paths: np.ndarray) -> dict:
    """
    Given simulated paths, compute:
    - "expected_value": mean of final day's values across all simulations
    - "p5", "p50", "p95": 5th/50th/95th percentile of final values (np.percentile)
    - "probability_of_loss": fraction of simulations ending below the initial value
    """
    final_values = paths[:, -1]
    return {
        "expected_value": np.mean(final_values),
        "p5": np.percentile(final_values, 5),
        "p50": np.percentile(final_values, 50),
        "p95": np.percentile(final_values, 95),
        "probability_of_loss": np.mean(final_values < paths[0, 0])
    }