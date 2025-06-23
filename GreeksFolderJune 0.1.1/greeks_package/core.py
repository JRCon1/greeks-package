from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings

# Suppress all warnings (optional, but not recommended for debugging)
warnings.simplefilter(action='ignore', category=FutureWarning)

def download_options(
    ticker_symbol,
    opt_type='c',
    max_days=60,
    lower_moneyness=0.95,
    upper_moneyness=1.05,
    price=False  # New optional parameter
):
    """
    Downloads and filters option chains for a given ticker according to:
      1. Option type (calls or puts)
      2. Maximum days to expiration
      3. Moneyness bounds
      4. Optionally includes the stock price in each row (useful for ITM/OTM visualization).

    Parameters:
        ticker_symbol (str): The stock ticker.
        opt_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        max_days (int, optional): Max days until expiration (default: 60).
        lower_moneyness (float, optional): Lower bound for moneyness (default: 0.95).
        upper_moneyness (float, optional): Upper bound for moneyness (default: 1.05).
        price (bool, optional): If True, adds a 'Stock Price' column with the current stock price.

    Returns:
        pd.DataFrame: Filtered options chain.
    """

    # Retrieve the ticker data from yfinance
    ticker = yf.Ticker(ticker_symbol)

    # Grab the current underlying price
    underlying_price = ticker.history(period="1d")['Close'].iloc[-1]

    # Calculate the strike range using the specified moneyness
    lower_strike = underlying_price * lower_moneyness
    upper_strike = underlying_price * upper_moneyness

    # Prepare a DataFrame to hold all filtered data
    relevant_columns = [
        'contractSymbol',
        'inTheMoney',
        'strike',
        'lastPrice',
        'bid',
        'ask',
        'volume',
        'openInterest',
        'impliedVolatility'
    ]
    filtered_options = pd.DataFrame(columns=relevant_columns + ['expiry'])

    # Loop through each available expiration date, filtering by max_days
    for expiry_date_str in ticker.options:
        expiry_date = pd.to_datetime(expiry_date_str)
        days_to_expiry = (expiry_date - datetime.now()).days

        if days_to_expiry <= max_days:
            # Retrieve calls or puts for the given expiration
            option_chain = ticker.option_chain(expiry_date_str)
            if opt_type.lower() == 'c':
                data = option_chain.calls
            elif opt_type.lower() == 'p':
                data = option_chain.puts
            else:
                continue

            # Filter by strike based on moneyness
            data = data[(data['strike'] >= lower_strike) & (data['strike'] <= upper_strike)].copy()

            # Attach an expiry column
            data['expiry'] = expiry_date

            # Concatenate only if data is non-empty
            if not data.empty:
                data = data[relevant_columns + ['expiry']]
                filtered_options = pd.concat([filtered_options, data], ignore_index=True)

    # Calculate Days to Expiry for each row
    filtered_options['Days to Expiry'] = (
        pd.to_datetime(filtered_options['expiry']) - datetime.now()
    ).dt.days

    # Calculate a Mid-Point price from bid and ask
    filtered_options['Mid-Point Price'] = round((filtered_options['bid'] + filtered_options['ask']) / 2, 4)

    filtered_options['impliedVolatility'] = filtered_options['impliedVolatility'].round(2)

    # If include_stock_price is True, add a 'Stock Price' column
    if price:
        filtered_options['Stock Price'] = round(underlying_price, 4)

    return filtered_options

#D1 & D2
def compute_d1(S, K, t, r, sigma, epsilon=1e-9):
    t = max(t, epsilon)  # Ensure positive time to avoid division by zero
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

def compute_d2(S, K, t, r, sigma, epsilon=1e-9):
    return compute_d1(S, K, t, r, sigma, epsilon) - sigma * np.sqrt(t)

def compute_d1_d2(S, K, t, r, sigma, epsilon=1e-9):
    t = max(t, epsilon)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    return d1, d1 - sigma * np.sqrt(t)

# -------------------- SECOND ORDER --------------------

def vanna(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> float:
    """
    Computes Vanna (sensitivity of Vega to volatility) for a single row of an options DataFrame.
    """
    # Fetch latest stock price
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]

    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    if pd.isna(S) or pd.isna(K) or pd.isna(sigma) or S <= 0 or sigma <= 0:
        return np.nan

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    N_prime_d1 = norm.pdf(d1)
    vanna_val = np.exp(-r * T) * N_prime_d1 * (d2 / sigma)
    return round(vanna_val, 4)

def volga(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> float:
    """Computes Volga (sensitivity of Vega to volatility)"""
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    vega_val = vega(option_type, S, K, T, r, sigma)
    volga_val = vega_val * (d1 * d2) / sigma
    return round(volga_val, 4)

def charm(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> float:
    """Computes Charm (Delta decay)"""
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    N_prime_d1 = norm.pdf(d1)
    charm_value = -N_prime_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T)
    return round(charm_value, 4)

def veta(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> float:
    """Computes Veta (sensitivity of Vega to time decay)"""
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    vega_val = vega(option_type, S, K, T, r, sigma)
    interest_rate_term = (r - d1 / (2 * T)) / sigma
    veta_val = vega_val * np.exp(-r * T) * norm.pdf(d1) * np.sqrt(T) * interest_rate_term
    return round(veta_val, 4)

# -------------------- THIRD ORDER --------------------

def color(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> float:
    """Computes Color (Gamma decay)"""
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    N_prime_d1 = norm.pdf(d1)
    color_value = (N_prime_d1 / (2 * S * T * sigma * np.sqrt(T))) * (2 * r * T + 1 - d1 * d2)
    return round(color_value, 4)

def speed(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> float:
    """Computes Speed (rate of change of Gamma)"""
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    N_prime_d1 = norm.pdf(d1)
    speed_value = (N_prime_d1 / (S ** 2 * sigma * np.sqrt(T))) * ((d1 / (sigma * np.sqrt(T))) - 1)
    return round(speed_value, 4)

def ultima(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> float:
    """Computes Ultima (sensitivity of Vanna to volatility)"""
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    vega_val = vega(option_type, S, K, T, r, sigma)
    ultima_value = (vega_val * (d1 * d2 - 1) * d1 * d2) / sigma
    return round(ultima_value, 4)

def zomma(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> float:
    """Computes Zomma (sensitivity of Gamma to volatility)"""
    S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    gamma_val = gamma(option_type, S, K, T, r, sigma)
    zomma_value = (gamma_val * (d1 * d2 - 1)) / sigma
    return round(zomma_value, 4)

# -------------------- WRAPPERS --------------------

def first_order(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> pd.Series:
    """Computes first-order Greeks (Delta, Vega, Theta, Rho)"""
    try:
        S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except Exception:
        return pd.Series({'Error': 'Stock price retrieval failed'})

    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    if pd.isna(S) or pd.isna(K) or pd.isna(sigma) or S <= 0 or sigma <= 0:
        return pd.Series({'Delta': np.nan, 'Vega': np.nan, 'Theta': np.nan, 'Rho': np.nan})

    greek_values = {
        'Delta': delta(option_type, S, K, T, r, sigma),
        'Vega': vega(option_type, S, K, T, r, sigma),
        'Theta': theta(option_type, S, K, T, r, sigma),
        'Rho': rho(option_type, S, K, T, r, sigma)
    }
    return pd.Series({k: round(v, 4) for k, v in greek_values.items()})

def second_order(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> pd.Series:
    """Computes second-order Greeks (Vanna, Volga, Veta, Charm, Gamma)"""
    try:
        S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except Exception:
        return pd.Series({'Error': 'Stock price retrieval failed'})

    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    if pd.isna(S) or pd.isna(K) or pd.isna(sigma) or S <= 0 or sigma <= 0:
        return pd.Series({
            'Vanna': np.nan, 'Volga': np.nan, 'Veta': np.nan,
            'Charm': np.nan, 'Gamma': np.nan
        })

    greek_values = {
        'Vanna': vanna(row, ticker, r, option_type),
        'Volga': volga(row, ticker, r, option_type),
        'Veta': veta(row, ticker, r, option_type),
        'Charm': charm(row, ticker, r, option_type),
        'Gamma': gamma(option_type, S, K, T, r, sigma)
    }
    return pd.Series({k: round(v, 4) for k, v in greek_values.items()})

def third_order(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> pd.Series:
    """Computes third-order Greeks (Color, Speed, Ultima, Zomma)"""
    try:
        S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except Exception:
        return pd.Series({'Error': 'Stock price retrieval failed'})

    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    if pd.isna(S) or pd.isna(K) or pd.isna(sigma) or S <= 0 or sigma <= 0:
        return pd.Series({
            'Color': np.nan, 'Speed': np.nan, 'Ultima': np.nan, 'Zomma': np.nan
        })

    greek_values = {
        'Color': color(row, ticker, r, option_type),
        'Speed': speed(row, ticker, r, option_type),
        'Ultima': ultima(row, ticker, r, option_type),
        'Zomma': zomma(row, ticker, r, option_type)
    }
    return pd.Series({k: round(v, 4) for k, v in greek_values.items()})

def greeks(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', epsilon: float = 1e-9) -> pd.Series:
    """Computes all Greeks (first, second, third order) for a single option row"""
    try:
        S = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
    except Exception:
        return pd.Series({'Error': 'Stock price retrieval failed'})

    K = row['strike']
    T = max(row['Days to Expiry'] / 365, epsilon)
    sigma = max(row['impliedVolatility'], 0.01)

    if pd.isna(S) or pd.isna(K) or pd.isna(sigma) or S <= 0 or sigma <= 0:
        return pd.Series({
            'Delta': np.nan, 'Vega': np.nan, 'Theta': np.nan, 'Rho': np.nan,
            'Gamma': np.nan, 'Vanna': np.nan, 'Volga': np.nan, 'Veta': np.nan, 'Charm': np.nan,
            'Color': np.nan, 'Speed': np.nan, 'Ultima': np.nan, 'Zomma': np.nan
        })

    # First-order
    first = {
        'Delta': delta(option_type, S, K, T, r, sigma),
        'Vega': vega(option_type, S, K, T, r, sigma),
        'Theta': theta(option_type, S, K, T, r, sigma),
        'Rho': rho(option_type, S, K, T, r, sigma)
    }

    # Second-order
    second = {
        'Gamma': gamma(option_type, S, K, T, r, sigma),
        'Vanna': vanna(row, ticker, r, option_type),
        'Volga': volga(row, ticker, r, option_type),
        'Veta': veta(row, ticker, r, option_type),
        'Charm': charm(row, ticker, r, option_type),
    }

    # Third-order
    third = {
        'Color': color(row, ticker, r, option_type),
        'Speed': speed(row, ticker, r, option_type),
        'Ultima': ultima(row, ticker, r, option_type),
        'Zomma': zomma(row, ticker, r, option_type),
    }

    all_greeks = {**first, **second, **third}
    return pd.Series({k: round(v, 4) for k, v in all_greeks.items()}) 