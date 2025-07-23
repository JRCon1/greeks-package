import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Union

# Suppress all warnings (optional, but not recommended for debugging)
warnings.simplefilter(action='ignore', category=FutureWarning)

# -------------------- OPTION CHAIN UTILS --------------------

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
      1. Option type (calls or puts, or both with 'all')
      2. Maximum days to expiration
      3. Moneyness bounds
      4. Optionally includes the stock price in each row (useful for ITM/OTM visualization).

    Parameters:
        ticker_symbol (str): The stock ticker.
        opt_type (str, optional): 'c' for calls, 'p' for puts, 'all' for both (default: 'c').
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
        days_to_expiry = round((expiry_date - datetime.now()).total_seconds() / 86_400, 2)  # fractional days, 2-dp

        if days_to_expiry <= max_days:
            # Retrieve both calls and puts for the given expiration
            option_chain = ticker.option_chain(expiry_date_str)
            calls_data = option_chain.calls
            puts_data = option_chain.puts

            # Filter by strike based on moneyness
            calls_data = calls_data[(calls_data['strike'] >= lower_strike) & (calls_data['strike'] <= upper_strike)].copy()
            puts_data = puts_data[(puts_data['strike'] >= lower_strike) & (puts_data['strike'] <= upper_strike)].copy()

            # Attach an expiry column
            calls_data['expiry'] = expiry_date
            puts_data['expiry'] = expiry_date

            # Concatenate calls and puts based on opt_type
            if opt_type.lower() == 'all':
                data = pd.concat([calls_data, puts_data], ignore_index=True)
            elif opt_type.lower() == 'c':
                data = calls_data
            elif opt_type.lower() == 'p':
                data = puts_data
            else:
                continue

            # Concatenate only if data is non-empty
            if not data.empty:
                data = data[relevant_columns + ['expiry']]
                filtered_options = pd.concat([filtered_options, data], ignore_index=True)

    # Calculate Days to Expiry for each row
    filtered_options['Days to Expiry'] = (
        (pd.to_datetime(filtered_options['expiry']) - datetime.now()).dt.total_seconds() / 86_400
    ).round(2) + 1  # fractional days to 2-dp

    # Calculate a Mid-Point price from bid and ask
    filtered_options['Mid-Point Price'] = round((filtered_options['bid'] + filtered_options['ask']) / 2, 4)

    filtered_options['impliedVolatility'] = filtered_options['impliedVolatility'].round(2)

    # If include_stock_price is True, add a 'Stock Price' column
    if price:
        filtered_options['Stock Price'] = round(underlying_price, 4)

    return filtered_options

def multi_download(
    ticker_symbols: List[str],
    opt_type: str = 'c',
    max_days: int = 60,
    lower_moneyness: float = 0.95,
    upper_moneyness: float = 1.05,
    price: bool = False
) -> pd.DataFrame:
    """
    Downloads and filters option chains for multiple tickers using download_options.

    Parameters:
        ticker_symbols (List[str]): List of stock tickers (e.g., ['AAPL', 'TSLA']).
        opt_type (str, optional): 'c' for calls, 'p' for puts, 'all' for both (default: 'c').
        max_days (int, optional): Max days until expiration (default: 60).
        lower_moneyness (float, optional): Lower bound for moneyness (default: 0.95).
        upper_moneyness (float, optional): Upper bound for moneyness (default: 1.05).
        price (bool, optional): If True, adds a 'Stock Price' column for each ticker (default: False).

    Returns:
        pd.DataFrame: Combined options chain with a 'Ticker' column identifying the source ticker.

    Raises:
        ValueError: If ticker_symbols is empty or contains invalid tickers.
    """
    if not ticker_symbols:
        raise ValueError("At least one ticker symbol must be provided.")

    all_options = []
    for ticker in ticker_symbols:
        try:
            options_df = download_options(
                ticker_symbol=ticker,
                opt_type=opt_type,
                max_days=max_days,
                lower_moneyness=lower_moneyness,
                upper_moneyness=upper_moneyness,
                price=price
            )
            if not options_df.empty:
                options_df['Ticker'] = ticker
                all_options.append(options_df)
        except Exception as e:
            print(f"Warning: Failed to download options for {ticker}: {str(e)}")
            continue

    if not all_options:
        raise ValueError("No valid option chains retrieved for any ticker.")

    combined_df = pd.concat(all_options, ignore_index=True)
    # Reorder columns to place 'Ticker' first
    cols = ['Ticker'] + [col for col in combined_df.columns if col != 'Ticker']
    return combined_df[cols]

# -------------------- BS-HELPERS --------------------

def compute_d1(S, K, t, r, sigma, eps=1e-9):
    """
    Calculate d1 parameter for Black-Scholes formula.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        t (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        eps (float, optional): Minimum time value to prevent division by zero.
    
    Returns:
        float: The d1 value used in Black-Scholes calculations.
    """
    t = max(t, eps)
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

def compute_d2(S, K, t, r, sigma, eps=1e-9):
    """
    Calculate d2 parameter for Black-Scholes formula.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        t (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        eps (float, optional): Minimum time value to prevent division by zero.
    
    Returns:
        float: The d2 value used in Black-Scholes calculations (d1 - sigma*sqrt(t)).
    """
    return compute_d1(S, K, t, r, sigma, eps) - sigma * np.sqrt(t)

def compute_d1_d2(S, K, t, r, sigma, eps=1e-9):
    """
    Calculate both d1 and d2 parameters for Black-Scholes formula efficiently.
    
    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        t (float): Time to expiration in years.
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        eps (float, optional): Minimum time value to prevent division by zero.
    
    Returns:
        tuple: (d1, d2) values used in Black-Scholes calculations.
    """
    d1 = compute_d1(S, K, t, r, sigma, eps)
    return d1, d1 - sigma * np.sqrt(t)

# -------------------- PRICING --------------------

def bsm_price(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05) -> float:
    """
    Calculate the Black-Scholes theoretical price for an option.
    
    This function computes the theoretical option price using the Black-Scholes formula
    for European options, using the current underlying price and option parameters.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        r (float, optional): Risk-free interest rate (default: 0.05).
    
    Returns:
        float: The Black-Scholes theoretical price, rounded to 4 decimal places.
               Returns np.nan if time to expiry or volatility is invalid.
    """
    S = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['impliedVolatility']
    if T <= 0 or sigma <= 0:
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    if option_type.lower() == 'c':
        val = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        val = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return round(val, 4)

# -------------------- MONTE CARLO --------------------

def monte_carlo_price(
    row: pd.Series,
    ticker: str,
    option_type: str = 'c',
    n: int = 10_000,
    r: float = 0.05,
    q: float = 0.0,
    return_paths: bool = False,
):
    """Estimate option price via Monte-Carlo under GBM."""
    S0 = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
    K, T, sigma = row['strike'], row['Days to Expiry'] / 365, row['impliedVolatility']
    if T <= 0 or sigma <= 0:
        return (np.nan, np.array([])) if return_paths else np.nan
    Z = np.random.normal(size=n)
    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    payoffs = np.maximum(ST - K, 0) if option_type.lower() == 'c' else np.maximum(K - ST, 0)
    price = np.exp(-r * T) * payoffs.mean()
    price = round(price, 4)
    return (price, ST) if return_paths else price

# -------------------- GREEKS (ROW-LEVEL) --------------------

def _get_spot(ticker: str):
    """
    Get the current stock price for a ticker symbol.
    
    Parameters:
        ticker (str): Stock ticker symbol.
    
    Returns:
        float: Most recent closing price.
    """
    return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]

def delta(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    """
    Calculate the delta (Δ) of an option - the rate of change of option price with respect to underlying price.
    
    Delta represents the sensitivity of the option's price to changes in the underlying asset's price.
    For calls: Delta ranges from 0 to 1. For puts: Delta ranges from -1 to 0.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        r (float, optional): Risk-free interest rate (default: 0.05).
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The delta value, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1 = compute_d1(S, K, T, r, sigma)
    return round(norm.cdf(d1) if option_type.lower() == 'c' else norm.cdf(d1) - 1, 4)

def theta(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    """
    Calculate the theta (Θ) of an option - the rate of change of option price with respect to time decay.
    
    Theta measures the time decay of an option's value. It represents how much the option price
    decreases as one day passes, holding all other factors constant. Theta is typically negative.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        r (float, optional): Risk-free interest rate (default: 0.05).
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The theta value per day, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    if T <= 0:
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    v = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    r_term = r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type.lower() == 'c' else -norm.cdf(-d2))
    return round((v + r_term) / 365, 4)

def vega(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    """
    Calculate the vega (ν) of an option - the rate of change of option price with respect to volatility.
    
    Vega measures the sensitivity of the option's price to changes in the underlying asset's volatility.
    Vega is always positive for both calls and puts, and is highest for at-the-money options.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        r (float, optional): Risk-free interest rate (default: 0.05).
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The vega value per 1% change in volatility, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1 = compute_d1(S, K, T, r, sigma)
    return round(S * np.sqrt(T) * norm.pdf(d1) * 0.01, 4)  # per 1% vol

def rho(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    """
    Calculate the rho (ρ) of an option - the rate of change of option price with respect to interest rates.
    
    Rho measures the sensitivity of the option's price to changes in the risk-free interest rate.
    Call options have positive rho, put options have negative rho.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        r (float, optional): Risk-free interest rate (default: 0.05).
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The rho value per 1% change in interest rate, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d2 = compute_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    val = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type.lower() == 'c' else -norm.cdf(-d2))
    return round(val * 0.01, 4)  # per 1%

# 2nd-order

def gamma(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    """
    Calculate the gamma (Γ) of an option - the rate of change of delta with respect to underlying price.
    
    Gamma is the second derivative of option price with respect to the underlying asset price.
    It measures the convexity of the option's price curve and is highest for at-the-money options.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        r (float, optional): Risk-free interest rate (default: 0.05).
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The gamma value, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1 = compute_d1(S, K, T, r, sigma)
    return round(norm.pdf(d1) / (S * sigma * np.sqrt(T)), 4)

def vanna(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    """
    Calculate the vanna of an option - the rate of change of delta with respect to volatility.
    
    Vanna is a second-order Greek that measures how delta changes as volatility changes.
    It's also the rate of change of vega with respect to the underlying price.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The vanna value, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round(np.exp(-r*T) * norm.pdf(d1) * (d2 / sigma), 4)

def volga(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    """
    Calculate the volga of an option - the rate of change of vega with respect to volatility.
    
    Volga is a second-order Greek that measures how vega changes as volatility changes.
    It's the second derivative of option price with respect to volatility.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The volga value, rounded to 4 decimal places.
    """
    v = vega(row, ticker, option_type, r, eps)
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round(v * (d1 * d2) / sigma, 4)

def charm(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    """
    Calculate the charm of an option - the rate of change of delta with respect to time.
    
    Charm is a second-order Greek that measures how delta changes as time passes.
    It represents the time decay of delta and is sometimes called delta decay.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The charm value, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round(-norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T), 4)

def veta(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    """
    Calculate the veta of an option - the rate of change of vega with respect to time.
    
    Veta is a second-order Greek that measures how vega changes as time passes.
    It represents the time decay of vega and shows how volatility sensitivity changes over time.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The veta value, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    term1 = (r * d1) / (sigma * np.sqrt(T))
    term2 = (1 + d1 * d2) / (2 * T)
    return round(-S * norm.pdf(d1) * np.sqrt(T) * (term1 - term2), 4)

# 3rd-order

def color(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    """
    Calculate the color of an option - the rate of change of gamma with respect to time.
    
    Color is a third-order Greek that measures how gamma changes as time passes.
    It represents the time decay of gamma and is useful for understanding convexity changes.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The color value, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round((norm.pdf(d1) / (2*S*T*sigma*np.sqrt(T))) * (2*r*T + 1 - d1*d2), 4)

def speed(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    """
    Calculate the speed of an option - the rate of change of gamma with respect to underlying price.
    
    Speed is a third-order Greek that measures how gamma changes as the underlying price changes.
    It's the third derivative of option price with respect to the underlying asset price.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The speed value, rounded to 4 decimal places.
    """
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, _ = compute_d1_d2(S, K, T, r, sigma)
    return round((norm.pdf(d1) / (S**2 * sigma * np.sqrt(T))) * ((d1/(sigma*np.sqrt(T))) - 1), 4)

def ultima(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    """
    Calculate the ultima of an option - the rate of change of volga with respect to volatility.
    
    Ultima is a third-order Greek that measures how volga changes as volatility changes.
    It's the third derivative of option price with respect to volatility.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The ultima value, rounded to 4 decimal places.
    """
    v = vega(row, ticker, option_type, r, eps)
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round(-v / sigma**2 * (d1*d2*(1 - d1*d2) + d1**2 + d2**2), 4)

def zomma(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    """
    Calculate the zomma of an option - the rate of change of gamma with respect to volatility.
    
    Zomma is a third-order Greek that measures how gamma changes as volatility changes.
    It's also known as DgammaDvol and provides insight into convexity changes due to volatility.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        float: The zomma value, rounded to 4 decimal places.
    """
    g = gamma(row, ticker, option_type, r, eps)
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round((g * (d1*d2 - 1)) / sigma, 4)

# -------------------- WRAPPERS --------------------

def first_order(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9) -> pd.Series:
    """
    Calculate all first-order Greeks (Delta, Vega, Theta, Rho) for an option.
    
    This is a convenience function that computes the four main first-order sensitivities
    of an option's price to changes in underlying factors.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        pd.Series: Series containing Delta, Vega, Theta, and Rho values.
    """
    vals = {
        'Delta': delta(row, ticker, option_type, r, eps),
        'Vega': vega(row, ticker, option_type, r, eps),
        'Theta': theta(row, ticker, option_type, r, eps),
        'Rho': rho(row, ticker, option_type, r, eps),
    }
    return pd.Series(vals)

def second_order(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9) -> pd.Series:
    """
    Calculate all second-order Greeks (Gamma, Vanna, Volga, Veta, Charm) for an option.
    
    This is a convenience function that computes the second-order sensitivities
    which measure how the first-order Greeks change with respect to various factors.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        pd.Series: Series containing Gamma, Vanna, Volga, Veta, and Charm values.
    """
    vals = {
        'Gamma': gamma(row, ticker, option_type, r, eps),
        'Vanna': vanna(row, ticker, r, option_type, eps),
        'Volga': volga(row, ticker, r, option_type, eps),
        'Veta': veta(row, ticker, r, option_type, eps),
        'Charm': charm(row, ticker, r, option_type, eps),
    }
    return pd.Series(vals)

def third_order(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9) -> pd.Series:
    """
    Calculate all third-order Greeks (Color, Speed, Ultima, Zomma) for an option.
    
    This is a convenience function that computes the third-order sensitivities
    which measure how the second-order Greeks change with respect to various factors.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        pd.Series: Series containing Color, Speed, Ultima, and Zomma values.
    """
    vals = {
        'Color': color(row, ticker, r, option_type, eps),
        'Speed': speed(row, ticker, r, option_type, eps),
        'Ultima': ultima(row, ticker, r, option_type, eps),
        'Zomma': zomma(row, ticker, r, option_type, eps),
    }
    return pd.Series(vals)

def greeks(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9) -> pd.Series:
    """
    Calculate all Greeks (first-, second-, and third-order) for an option in one call.
    
    This is the main convenience function that computes all available Greeks
    by combining first_order, second_order, and third_order functions.
    
    Parameters:
        row (pd.Series): Option data containing 'strike', 'Days to Expiry', and 'impliedVolatility'.
        ticker (str): The stock ticker symbol.
        r (float, optional): Risk-free interest rate (default: 0.05).
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        eps (float, optional): Small value to prevent division by zero (default: 1e-9).
    
    Returns:
        pd.Series: Series containing all 13 Greeks: Delta, Vega, Theta, Rho, Gamma, 
                  Vanna, Volga, Veta, Charm, Color, Speed, Ultima, and Zomma.
    """
    return pd.concat([
        first_order(row, ticker, r, option_type, eps),
        second_order(row, ticker, r, option_type, eps),
        third_order(row, ticker, r, option_type, eps),
    ])

# -------------------- COMBINATION HELPER --------------------

def comb(*dfs: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple DataFrames by joining them horizontally with automatic column renaming.
    
    This utility function joins multiple DataFrames side-by-side, automatically handling
    duplicate column names by appending suffixes. Useful for combining options data
    with Greeks calculations or multiple analysis results.
    
    Parameters:
        *dfs (pd.DataFrame): Variable number of DataFrames to combine. At least one required.
    
    Returns:
        pd.DataFrame: Combined DataFrame with all columns from input DataFrames.
                     Duplicate column names are renamed with suffixes (_1, _2, etc.).
    
    Raises:
        ValueError: If no DataFrames are provided.
        TypeError: If any argument is not a pandas DataFrame.
    
    Examples:
        >>> opts = gp.download_options("AAPL")
        >>> greeks_df = opts.apply(gp.greeks, axis=1, ticker="AAPL")
        >>> pricing_df = opts.apply(gp.bsm_price, axis=1, ticker="AAPL")
        >>> combined = gp.comb(opts, greeks_df, pricing_df)
    """
    if not dfs:
        raise ValueError("Provide at least one DataFrame to combine")
    base = dfs[0].copy()
    for other in dfs[1:]:
        if not isinstance(other, pd.DataFrame):
            raise TypeError("All arguments must be pandas DataFrames")
        dupes = set(base.columns) & set(other.columns)
        if dupes:
            other = other.rename(columns={c: f"{c}_{i+1}" for i, c in enumerate(dupes)})
        base = base.join(other, how="left")
    return base

# -------------------- VISUALISATION --------------------

def _maybe_compute_z(df: pd.DataFrame, z: str, ticker: str, option_type: str, r: float):
    """
    Internal helper to ensure a Greek column exists in DataFrame for plotting.
    
    Checks if the specified Greek column exists in the DataFrame. If not, computes it
    using the appropriate Greek function. Used by plotting functions to auto-calculate
    Greeks if they don't already exist in the data.
    
    Parameters:
        df (pd.DataFrame): Options DataFrame.
        z (str): Name of the Greek to ensure exists.
        ticker (str): Stock ticker symbol.
        option_type (str): 'c' for calls, 'p' for puts.
        r (float): Risk-free interest rate.
    
    Returns:
        pd.DataFrame: DataFrame with the specified Greek column added if it was missing.
    
    Raises:
        ValueError: If the Greek name is not recognized and can't be computed.
    """
    z_l = z.lower()
    funcs = {
        'delta': delta, 'theta': theta, 'vega': vega, 'rho': rho, 'gamma': gamma,
        'vanna': vanna, 'volga': volga, 'veta': veta, 'charm': charm,
        'color': color, 'speed': speed, 'ultima': ultima, 'zomma': zomma,
    }
    if z_l in df.columns:
        return df
    col_map = {c.lower(): c for c in df.columns}
    if z_l in col_map:
        df[z_l] = df[col_map[z_l]]
        return df
    if z_l in funcs:
        df[z_l] = df.apply(lambda r_: funcs[z_l](r_, ticker, option_type=option_type, r=r), axis=1)
        return df
    raise ValueError(f"Column '{z}' not found and no computation rule available.")

def surf_scatter(df: pd.DataFrame, ticker: str, z: str = 'delta', option_type: str = 'c', r: float = 0.05, **kwargs):
    """
    Create an interactive 3D scatter plot of a Greek against strike and days to expiry.
    
    Generates a 3D scatter plot using Plotly showing how a specified Greek (or other metric)
    varies across different strikes and time to expiration. Points are colored by moneyness
    (ITM vs OTM) and include hover information for detailed inspection.
    
    Parameters:
        df (pd.DataFrame): Options DataFrame containing required columns.
        ticker (str): Stock ticker symbol for current price lookup.
        z (str, optional): Name of the Greek or metric to plot on z-axis (default: 'delta').
                          If not present in DataFrame, will be computed automatically.
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        r (float, optional): Risk-free interest rate for Greek calculations (default: 0.05).
        **kwargs: Additional arguments passed to plotly.express.scatter_3d().
    
    Required DataFrame Columns:
        - 'strike': Option strike prices
        - 'Days to Expiry': Time until expiration in days
        - 'impliedVolatility': Implied volatility values
        - 'contractSymbol': Option contract identifiers (for hover data)
        - 'lastPrice': Option market prices (for hover data)
    
    Raises:
        ValueError: If required columns are missing from the DataFrame.
    
    Examples:
        >>> opts = gp.download_options("AAPL")
        >>> gp.surf_scatter(opts, "AAPL", z="delta")  # Plot delta surface
        >>> gp.surf_scatter(opts, "AAPL", z="gamma")  # Plot gamma surface
        >>> gp.surf_scatter(opts, "AAPL", z="impliedVolatility")  # Plot IV surface
    """
    req = {'strike', 'Days to Expiry', 'impliedVolatility'}
    if req - set(df.columns):
        raise ValueError(f"DataFrame missing required columns: {req - set(df.columns)}")
    df = _maybe_compute_z(df, z, ticker, option_type, r)
    if 'moneyness_tag' not in df.columns:
        S0 = _get_spot(ticker)
        df['moneyness_tag'] = np.where(df['strike'] < S0, 'ITM', 'OTM')
    fig = px.scatter_3d(
        df, x='Days to Expiry', y='strike', z=z.lower(), color='moneyness_tag',
        color_discrete_map={'ITM': 'green', 'OTM': 'red'},
        hover_data=['contractSymbol', 'lastPrice', 'impliedVolatility'],
        height=700, width=900,
        title=f"{z.upper()} vs Days to Expiry / Strike", **kwargs,
    )
    fig.update_layout(scene=dict(xaxis=dict(title='Days to Expiry', autorange='reversed'),
                                 yaxis=dict(title='Strike'),
                                 zaxis=dict(title=z.upper())))
    fig.update_coloraxes(showscale=False)
    fig.show()

def surface_plot(df: pd.DataFrame, ticker: str, z: str = 'impliedVolatility', option_type: str = 'c', r: float = 0.05, **kwargs):
    """
    Create an interactive 3D surface plot of a Greek or metric across strike and time dimensions.
    
    Generates a smooth 3D surface using Plotly showing how a specified Greek (or other metric)
    varies continuously across different strikes and times to expiration. The surface is
    interpolated from available data points and provides a comprehensive view of the
    Greek's behavior across the option chain.
    
    Parameters:
        df (pd.DataFrame): Options DataFrame containing required columns.
        ticker (str): Stock ticker symbol for current price lookup.
        z (str, optional): Name of the Greek or metric to plot as surface height 
                          (default: 'impliedVolatility'). If not present in DataFrame, 
                          will be computed automatically.
        option_type (str, optional): 'c' for calls, 'p' for puts (default: 'c').
        r (float, optional): Risk-free interest rate for Greek calculations (default: 0.05).
        **kwargs: Additional arguments passed to plotly graph objects (currently unused).
    
    Required DataFrame Columns:
        - 'strike': Option strike prices
        - 'Days to Expiry': Time until expiration in days
        - Additional columns needed depend on the chosen metric z
    
    Note:
        This function creates a pivot table from the data and interpolates missing values
        to create a smooth surface. Data points with identical strike/expiry combinations
        are averaged using the mean aggregation function.
    
    Examples:
        >>> opts = gp.download_options("AAPL")
        >>> gp.surface_plot(opts, "AAPL")  # Default: implied volatility surface
        >>> gp.surface_plot(opts, "AAPL", z="delta")  # Delta surface
        >>> gp.surface_plot(opts, "AAPL", z="gamma")  # Gamma surface
    """
    df = _maybe_compute_z(df, z, ticker, option_type, r)
    x = np.sort(df['Days to Expiry'].unique())[::-1]
    y = np.sort(df['strike'].unique())
    z_mat = np.full((len(y), len(x)), np.nan)
    piv = df.pivot_table(index='strike', columns='Days to Expiry', values=z.lower(), aggfunc='mean')
    for i, yv in enumerate(y):
        if yv not in piv.index:
            continue
        row_vals = piv.loc[yv]
        for j, xv in enumerate(x):
            z_mat[i, j] = row_vals.get(xv, np.nan)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z_mat, colorscale='Viridis')])
    fig.update_layout(title=f"{z.upper()} Surface",
                      scene=dict(xaxis=dict(title='Days to Expiry', autorange='reversed'),
                                 yaxis=dict(title='Strike'),
                                 zaxis=dict(title=z.upper())),
                      height=700, width=900)
    fig.show()

def greek_plot(df: pd.DataFrame, greek_col: str, x_axis: str = 'Days to Expiry', return_fig: bool = False, **kwargs):
    """
    Create an interactive Plotly graph to visualize a Greek value over time with strike selection.

    Generates a line plot using Plotly where users can select different strikes via a dropdown
    to view the specified Greek's behavior over the x-axis (default: Days to Expiry).

    Parameters:
        df (pd.DataFrame): Options DataFrame containing the Greek column and required axes columns.
        greek_col (str): Name of the column containing the Greek to plot (e.g., 'Delta', 'Theta').
        x_axis (str, optional): Column name for the x-axis (default: 'Days to Expiry').
        return_fig (bool, optional): If True, returns the Plotly figure object instead of displaying it (default: False).
        **kwargs: Additional arguments passed to plotly.graph_objects.Figure().

    Required DataFrame Columns:
        - The specified greek_col (e.g., 'Delta', 'Theta', etc.).
        - The specified x_axis (default: 'Days to Expiry').
        - 'strike': For dropdown selection.

    Returns:
        None: Displays the plot interactively if return_fig is False.
        plotly.graph_objects.Figure: Returns the figure object if return_fig is True, allowing further customization or saving.

    Examples:
        >>> opts = gp.download_options("AAPL")
        >>> opts_with_greeks = opts.apply(lambda row: gp.greeks(row, ticker="AAPL"), axis=1, result_type='expand')
        >>> combined_df = pd.concat([opts, opts_with_greeks], axis=1)
        >>> gp.greek_plot(combined_df, greek_col="Delta")  # Plot Delta with strike dropdown
        >>> fig = gp.greek_plot(combined_df, greek_col="Delta", return_fig=True)  # Return figure
        >>> fig.show()  # Display in notebook
        >>> fig.write_html("delta_plot.html")  # Save to HTML
    """
    # Validate required columns
    required_cols = {greek_col, x_axis, 'strike'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Get unique strikes for dropdown
    unique_strikes = df['strike'].unique()
    if len(unique_strikes) < 1:
        raise ValueError("No unique strikes found in DataFrame.")

    # Create figure with dropdown
    fig = make_subplots(rows=1, cols=1)

    # Initial trace for the first strike
    initial_strike = unique_strikes[0]
    trace = go.Scatter(x=df[df['strike'] == initial_strike][x_axis],
                       y=df[df['strike'] == initial_strike][greek_col],
                       mode='lines+markers',
                       name=f'Strike {initial_strike}')
    fig.add_trace(trace)

    # Define dropdown buttons for each strike
    buttons = []
    for strike in unique_strikes:
        buttons.append({
            'label': f'Strike {strike}',
            'method': 'update',
            'args': [{'x': [df[df['strike'] == strike][x_axis]],
                      'y': [df[df['strike'] == strike][greek_col]],
                      'name': [f'Strike {strike}']}]
        })

    # Update layout with dropdown and reversed x-axis
    fig.update_layout(
        title=f"{greek_col.upper()} vs {x_axis}",
        xaxis_title=x_axis,
        yaxis_title=greek_col.upper(),
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }],
        paper_bgcolor='rgba(30, 30, 30, 1)',  # Darker background
        plot_bgcolor='rgba(20, 20, 20, 1)',  # Lighter plot area
        font=dict(color='white'),  # White text for contrast
        template='plotly_dark',
        annotations=[
            dict(
                text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EDT')}",
                xref="paper", yref="paper",
                x=0.01, y=0.98, showarrow=False,
                font=dict(size=10, color="white")
            )
        ],
        xaxis=dict(autorange='reversed'),  # Reverse the x-axis
        **kwargs
    )

    if return_fig:
        return fig
    fig.show()

def iv_plot(df: pd.DataFrame, ticker: str, return_fig: bool = False, **kwargs):
    """
    Create an interactive Plotly line plot of implied volatility vs Days to Expiry with strike selection.

    Generates a line plot with markers showing implied volatility for a selected strike price across
    different expiration dates, with a dropdown menu to switch between available strikes. The x-axis is
    reversed to show time progressing from right (nearer expiry) to left (further expiry).

    Parameters:
        df (pd.DataFrame): Options DataFrame containing required columns (e.g., from download_options).
        ticker (str): Stock ticker symbol (used for plot title and validation).
        return_fig (bool, optional): If True, returns the Plotly figure object instead of displaying it (default: False).
        **kwargs: Additional arguments passed to plotly.graph_objects.Figure().

    Required DataFrame Columns:
        - 'strike': Option strike prices.
        - 'Days to Expiry': Time until expiration in days.
        - 'impliedVolatility': Implied volatility values.
        - 'contractSymbol': Option contract identifiers (for hover data).
        - 'lastPrice': Option market prices (for hover data).
        - 'expiry': Expiration dates (for hover data).

    Returns:
        None: Displays the plot interactively if return_fig is False.
        plotly.graph_objects.Figure: Returns the figure object if return_fig is True, allowing further customization.

    Raises:
        ValueError: If required columns are missing or no valid strike prices are found.

    Examples:
        >>> v_calls = download_options("V", opt_type="c", max_days=30, price=True)
        >>> iv_plot(v_calls, "V")  # Display plot directly
        >>> fig = iv_plot(v_calls, "V", return_fig=True)  # Return figure for customization
        >>> fig.show()  # Display in notebook
        >>> fig.write_html("iv_plot.html")  # Save to HTML
    """
    # Validate required columns
    required_cols = {'strike', 'Days to Expiry', 'impliedVolatility', 'contractSymbol', 'lastPrice', 'expiry'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Get unique strikes for dropdown
    unique_strikes = sorted(df['strike'].unique())
    if len(unique_strikes) < 1:
        raise ValueError("No unique strikes found in DataFrame.")

    # Create figure with subplots
    fig = make_subplots(rows=1, cols=1)

    # Initial trace for the first strike
    initial_strike = unique_strikes[0]
    df_strike = df[df['strike'] == initial_strike]
    trace = go.Scatter(
        x=df_strike['Days to Expiry'],
        y=df_strike['impliedVolatility'],
        mode='lines+markers',
        name=f'Strike {initial_strike}',
        text=df_strike['impliedVolatility'].round(4),
        hovertemplate=(
            'Contract: %{customdata[0]}<br>'
            'Expiry: %{customdata[1]}<br>'
            'Days to Expiry: %{x:.2f}<br>'
            'Implied Volatility: %{y:.4f}<br>'
            'Last Price: $%{customdata[2]:.2f}<extra></extra>'
        ),
        customdata=df_strike[['contractSymbol', 'expiry', 'lastPrice']].values
    )
    fig.add_trace(trace)

    # Define dropdown buttons for each strike
    buttons = []
    for strike in unique_strikes:
        df_strike = df[df['strike'] == strike]
        buttons.append({
            'label': f'Strike {strike}',
            'method': 'update',
            'args': [{
                'x': [df_strike['Days to Expiry']],
                'y': [df_strike['impliedVolatility']],
                'name': [f'Strike {strike}'],
                'text': [df_strike['impliedVolatility'].round(4)],
                'customdata': [df_strike[['contractSymbol', 'expiry', 'lastPrice']].values]
            }]
        })

    # Update layout with dropdown, reversed x-axis, and dark theme
    fig.update_layout(
        title=f"Implied Volatility vs Days to Expiry for {ticker} Options",
        xaxis_title="Days to Expiry",
        yaxis_title="Implied Volatility",
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }],
        paper_bgcolor='rgba(30, 30, 30, 1)',  # Darker background
        plot_bgcolor='rgba(20, 20, 20, 1)',  # Lighter plot area
        font=dict(color='white'),  # White text for contrast
        template='plotly_dark',
        annotations=[
            dict(
                text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EDT')}",
                xref="paper", yref="paper",
                x=0.01, y=0.98, showarrow=False,
                font=dict(size=10, color="white")
            )
        ],
        xaxis=dict(autorange='reversed'),  # Reverse the x-axis
        yaxis=dict(tickformat='.2f'),  # Format y-axis as decimal
        **kwargs
    )

    if return_fig:
        return fig
    fig.show()

def oi_plot(df: pd.DataFrame, ticker: str, return_fig: bool = False, **kwargs):
    """
    Create an interactive Plotly line plot of open interest vs Days to Expiry with strike selection.

    Generates a line plot with markers showing open interest for a selected strike price across
    different expiration dates, with a dropdown menu to switch between available strikes. The x-axis is
    reversed to show time progressing from right (nearer expiry) to left (further expiry).

    Parameters:
        df (pd.DataFrame): Options DataFrame containing required columns (e.g., from download_options).
        ticker (str): Stock ticker symbol (used for plot title and validation).
        return_fig (bool, optional): If True, returns the Plotly figure object instead of displaying it (default: False).
        **kwargs: Additional arguments passed to plotly.graph_objects.Figure().

    Required DataFrame Columns:
        - strike: Option strike prices.
        - Days to Expiry: Time until expiration in days.
        - openInterest: Open interest values.
        - contractSymbol: Option contract identifiers (for hover data).
        - lastPrice: Option market prices (for hover data).
        - expiry: Expiration dates (for hover data).

    Returns:
        None: Displays the plot interactively if return_fig is False.
        plotly.graph_objects.Figure: Returns the figure object if return_fig is True, allowing further customization.

    Raises:
        ValueError: If required columns are missing or no valid strike prices are found.

    Examples:
        >>> v_calls = download_options('V', opt_type='c', max_days=30, price=True)
        >>> oi_plot(v_calls, 'V')  # Display plot directly
        >>> fig = oi_plot(v_calls, 'V', return_fig=True)  # Return figure for customization
        >>> fig.show()  # Display in notebook
        >>> fig.write_html('oi_plot.html')  # Save to HTML
    """
    # Validate required columns
    required_cols = {'strike', 'Days to Expiry', 'openInterest', 'contractSymbol', 'lastPrice', 'expiry'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Get unique strikes for dropdown
    unique_strikes = sorted(df['strike'].unique())
    if len(unique_strikes) < 1:
        raise ValueError("No unique strikes found in DataFrame.")

    # Create figure with subplots
    fig = make_subplots(rows=1, cols=1)

    # Initial trace for the first strike
    initial_strike = unique_strikes[0]
    df_strike = df[df['strike'] == initial_strike]
    trace = go.Scatter(
        x=df_strike['Days to Expiry'],
        y=df_strike['openInterest'],
        mode='lines+markers',
        name=f'Strike {initial_strike}',
        text=df_strike['openInterest'].round(0).astype(int).astype(str),
        hovertemplate=(
            'Contract: %{customdata[0]}<br>'
            'Expiry: %{customdata[1]}<br>'
            'Days to Expiry: %{x:.2f}<br>'
            'Open Interest: %{y:.0f}<br>'
            'Last Price: $%{customdata[2]:.2f}<extra></extra>'
        ),
        customdata=df_strike[['contractSymbol', 'expiry', 'lastPrice']].values
    )
    fig.add_trace(trace)

    # Define dropdown buttons for each strike
    buttons = []
    for strike in unique_strikes:
        df_strike = df[df['strike'] == strike]
        buttons.append({
            'label': f'Strike {strike}',
            'method': 'update',
            'args': [{
                'x': [df_strike['Days to Expiry']],
                'y': [df_strike['openInterest']],
                'name': [f'Strike {strike}'],
                'text': [df_strike['openInterest'].round(0).astype(int).astype(str)],
                'customdata': [df_strike[['contractSymbol', 'expiry', 'lastPrice']].values]
            }]
        })

    # Update layout with dropdown, reversed x-axis, and dark theme
    fig.update_layout(
        title=f"Open Interest vs Days to Expiry for {ticker} Options",
        xaxis_title="Days to Expiry",
        yaxis_title="Open Interest",
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }],
        paper_bgcolor='rgba(30, 30, 30, 1)',  # Darker background
        plot_bgcolor='rgba(20, 20, 20, 1)',  # Lighter plot area
        font=dict(color='white'),  # White text for contrast
        template='plotly_dark',
        annotations=[
            dict(
                text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EDT')}",
                xref="paper", yref="paper",
                x=0.01, y=0.98, showarrow=False,
                font=dict(size=10, color="white")
            )
        ],
        xaxis=dict(autorange='reversed'),  # Reverse the x-axis
        yaxis=dict(tickformat='.0f'),  # Format y-axis as integer
        **kwargs
    )

    if return_fig:
        return fig
    fig.show()

def vol_curve(df: pd.DataFrame, ticker: str, return_fig: bool = False, **kwargs):
    """
    Create an interactive Plotly line plot of implied volatility vs strike price with expiration date selection.

    Generates a line plot with markers showing implied volatility across strike prices for a selected
    expiration date, with a dropdown menu to switch between available expiration dates. Includes a
    vertical line indicating the current stock price to show ITM/OTM regions.

    Parameters:
        df (pd.DataFrame): Options DataFrame containing required columns (e.g., from download_options).
        ticker (str): Stock ticker symbol (used for plot title and validation).
        return_fig (bool, optional): If True, returns the Plotly figure object instead of displaying it (default: False).
        **kwargs: Additional arguments passed to plotly.graph_objects.Figure().

    Required DataFrame Columns:
        - strike: Option strike prices.
        - impliedVolatility: Implied volatility values.
        - expiry: Expiration dates.
        - contractSymbol: Option contract identifiers (for hover data).
        - lastPrice: Option market prices (for hover data).
        - openInterest: Open interest values (for hover data).
        - Stock Price: Current stock price (required for ITM/OTM indicator).

    Returns:
        None: Displays the plot interactively if return_fig is False.
        plotly.graph_objects.Figure: Returns the figure object if return_fig is True, allowing further customization.

    Raises:
        ValueError: If required columns are missing, no valid expiration dates are found, or Stock Price is missing/inconsistent.

    Examples:
        >>> v_calls = download_options('V', opt_type='c', max_days=30, price=True)
        >>> vol_curve(v_calls, 'V')  # Display plot with stock price line
        >>> fig = vol_curve(v_calls, 'V', return_fig=True)  # Return figure for customization
        >>> fig.show()  # Display in notebook
        >>> fig.write_html('vol_curve.html')  # Save to HTML
    """
    # Validate required columns
    required_cols = {'strike', 'impliedVolatility', 'expiry', 'contractSymbol', 'lastPrice', 'openInterest', 'Stock Price'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    # Validate stock price consistency
    stock_prices = df['Stock Price'].unique()
    if len(stock_prices) != 1:
        raise ValueError(f"Inconsistent or missing Stock Price values: {stock_prices}. Ensure price=True in download_options.")
    stock_price = stock_prices[0]

    # Get unique expiration dates for dropdown
    unique_expiries = sorted(df['expiry'].dt.strftime('%Y-%m-%d').unique())
    if len(unique_expiries) < 1:
        raise ValueError("No valid expiration dates found in DataFrame.")

    # Create figure with subplots
    fig = make_subplots(rows=1, cols=1)

    # Initial trace for the first expiry
    initial_expiry = unique_expiries[0]
    df_expiry = df[df['expiry'] == pd.to_datetime(initial_expiry)]
    trace = go.Scatter(
        x=df_expiry['strike'],
        y=df_expiry['impliedVolatility'],
        mode='lines+markers',
        name=f'Expiry {initial_expiry}',
        text=df_expiry['impliedVolatility'].round(4),
        hovertemplate=(
            'Contract: %{customdata[0]}<br>'
            'Expiry: %{customdata[1]}<br>'
            'Strike: %{x:.2f}<br>'
            'Implied Volatility: %{y:.4f}<br>'
            'Last Price: $%{customdata[2]:.2f}<br>'
            'Open Interest: %{customdata[3]:.0f}<extra></extra>'
        ),
        customdata=df_expiry[['contractSymbol', 'expiry', 'lastPrice', 'openInterest']].values
    )
    fig.add_trace(trace)

    # Define dropdown buttons for each expiry
    buttons = []
    for expiry in unique_expiries:
        df_expiry = df[df['expiry'] == pd.to_datetime(expiry)]
        buttons.append({
            'label': f'Expiry {expiry}',
            'method': 'update',
            'args': [{
                'x': [df_expiry['strike']],
                'y': [df_expiry['impliedVolatility']],
                'name': [f'Expiry {expiry}'],
                'text': [df_expiry['impliedVolatility'].round(4)],
                'customdata': [df_expiry[['contractSymbol', 'expiry', 'lastPrice', 'openInterest']].values]
            }]
        })

    # Add vertical line for stock price
    fig.add_shape(
        type="line",
        x0=stock_price,
        x1=stock_price,
        y0=0,
        y1=1,
        yref="paper",  # Span entire y-axis
        line=dict(
            color="red",
            width=1,
            dash="dash"
        ),
        name="Stock Price"
    )

    # Add annotation for stock price
    fig.add_annotation(
        x=stock_price,
        y=1,
        yref="paper",
        text=f"Stock Price: ${stock_price:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-30,
        font=dict(size=10, color="red")
    )

    # Update layout with dropdown and dark theme
    fig.update_layout(
        title=f"Implied Volatility Curve for {ticker} Options",
        xaxis_title="Strike Price",
        yaxis_title="Implied Volatility",
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }],
        paper_bgcolor='rgba(30, 30, 30, 1)',  # Darker background
        plot_bgcolor='rgba(20, 20, 20, 1)',  # Lighter plot area
        font=dict(color='white'),  # White text for contrast
        template='plotly_dark',
        annotations=[
            dict(
                text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EDT')}",
                xref="paper", yref="paper",
                x=0.01, y=0.98, showarrow=False,
                font=dict(size=10, color="white")
            )
        ],
        xaxis=dict(tickformat='.2f'),  # Format x-axis as decimal
        yaxis=dict(tickformat='.2f'),  # Format y-axis as decimal
        showlegend=True,
        **kwargs
    )

    if return_fig:
        return fig
    fig.show()

# -------------------- PUBLIC EXPORTS --------------------
__all__ = [
    'download_options', 'multi_download', 'bsm_price', 'monte_carlo_price',
    # Greeks (row-level)
    'delta', 'theta', 'vega', 'rho', 'gamma',
    'vanna', 'volga', 'veta', 'charm',
    'color', 'speed', 'ultima', 'zomma',
    # Wrappers
    'first_order', 'second_order', 'third_order', 'greeks',
    # Utils
    'comb', 'surf_scatter', 'surface_plot', 'greek_plot', 'iv_plot', 'oi_plot', 'vol_curve',
    # Strategy
    'strategy_builder', 'identify_strategy', 'validate_strategy',
]


def strategy_builder(
    df: pd.DataFrame,
    legs: List[Dict[str, Union[float, str, int]]],
    plot: bool = False,
    greeks: bool = False
) -> Dict[str, Union[float, pd.DataFrame]]:
    """
    Calculate the net cost of a multi-leg options strategy and optionally plot the payoff diagram.

    Parameters:
    - df (pd.DataFrame): Options DataFrame from download_options (columns: strike, expiry, bid, ask, Mid-Point Price, etc.).
    - legs (List[Dict]): List of leg dictionaries, each with 'strike', 'option_type' ('call'/'put'), 'action' ('buy'/'sell'), 'contracts' (int), and 'expiry' (str or pd.Timestamp).
    - plot (bool): If True, generates and displays a payoff diagram (default: False).
    - greeks (bool): If True, calculates and displays net Greek exposure (default: False). Requires at least one Greek column (e.g., Delta, Theta) from index 13 onward.

    Returns:
    - Dict with:
        - 'net_cost': Net cash flow (positive = inflow, negative = outflow).
        - 'details': DataFrame summarizing each leg's cost and details.
        - 'strategy': Identified strategy based on leg configuration.
        - 'net_greeks' (if greeks=True): Dict of net Greek exposures for all available Greek columns from index 13 onward.
    """
    try:
        # Input validation
        if not legs:
            raise ValueError("At least one leg must be specified.")
        for leg in legs:
            if not {'strike', 'option_type', 'action', 'contracts', 'expiry'}.issubset(leg.keys()):
                raise ValueError("Each leg must specify 'strike', 'option_type', 'action', 'contracts', and 'expiry'.")
            if leg['action'] not in ['buy', 'sell']:
                raise ValueError("Action must be 'buy' or 'sell'.")
            if leg['option_type'].lower() not in ['call', 'put']:
                raise ValueError("option_type must be 'call' or 'put'.")
            if not isinstance(leg['contracts'], int) or leg['contracts'] <= 0:
                raise ValueError("Contracts must be a positive integer.")

        # Add option_type column if not present (infer from contractSymbol)
        if 'option_type' not in df.columns:
            df = df.copy()
            df['option_type'] = df['contractSymbol'].str[-9].map({'C': 'call', 'P': 'put'})

        # Normalize expiry format in DataFrame and legs
        df = df.copy()
        df['expiry'] = pd.to_datetime(df['expiry']).dt.strftime('%Y-%m-%d')
        for leg in legs:
            leg['expiry'] = pd.to_datetime(leg['expiry']).strftime('%Y-%m-%d')

        # Identify strategy
        strategy = identify_strategy(legs, net_cashflow=0.0)  # Initial call with default net_cashflow

        # Calculate cash flow for each leg
        leg_details = []
        net_cashflow = 0.0
        for leg in legs:
            # Filter DataFrame for the specific contract
            query = (
                (df['strike'] == leg['strike']) &
                (df['option_type'].str.lower() == leg['option_type'].lower()) &
                (df['expiry'] == leg['expiry'])
            )

            contract = df[query]
            if contract.empty:
                raise ValueError(f"No contract found for strike {leg['strike']}, type {leg['option_type']}, expiry {leg['expiry']}.")
            if len(contract) > 1:
                raise ValueError(f"Multiple contracts found for strike {leg['strike']}, type {leg['option_type']}, expiry {leg['expiry']}.")

            # Use Mid-Point Price for cost calculation
            price = contract['Mid-Point Price'].iloc[0]
            if pd.isna(price):
                raise ValueError(f"Mid-Point Price is missing for strike {leg['strike']}, type {leg['option_type']}, expiry {leg['expiry']}.")

            cashflow = -price if leg['action'] == 'buy' else price
            total_cost = cashflow * leg['contracts'] * 100  # Per contract = 100 shares

            net_cashflow += total_cost
            leg_details.append({
                'strike': leg['strike'],
                'option_type': leg['option_type'],
                'action': leg['action'],
                'price': price,
                'total_cost': total_cost,
                'contracts': leg['contracts'],
                'expiry': contract['expiry'].iloc[0]
            })

        # Re-identify strategy with actual net_cashflow
        strategy = identify_strategy(legs, net_cashflow)

        # Validate specific strategy constraints
        if strategy:
            validate_strategy(legs, strategy)

        # Create details DataFrame
        details_df = pd.DataFrame(leg_details)

        # Calculate net Greek exposure if requested
        net_greeks = {}
        if greeks:
            # Define set of valid Greek names
            valid_greeks = {"Delta", "Theta", "Vega", "Rho", "Gamma", "Vanna", "Volga", "Veta", "Charm", "Color", "Speed", "Ultima", "Zomma"}
            # Get all columns from index 12 onward
            potential_greek_columns = df.columns[12:]
            # Check if at least one Greek column exists
            greek_columns = [col for col in potential_greek_columns if col in valid_greeks]
            if not greek_columns:
                raise ValueError("No Greek columns found in DataFrame from index 13 onward. At least one Greek (e.g., Delta, Theta) is required.")
            # Calculate net exposure for all available Greek columns
            for greek in greek_columns:
                net_greek = 0.0
                for leg in legs:
                    query = (
                        (df['strike'] == leg['strike']) &
                        (df['option_type'].str.lower() == leg['option_type'].lower()) &
                        (df['expiry'] == leg['expiry'])
                    )
                    contract = df[query]
                    value = contract[greek].iloc[0]
                    if pd.isna(value):
                        raise ValueError(f"Missing {greek} value for strike {leg['strike']}, type {leg['option_type']}, expiry {leg['expiry']}.")
                    # Adjust for action: + for buy, - for sell
                    adjustment = 1 if leg['action'] == 'buy' else -1
                    net_greek += value * adjustment * leg['contracts']
                net_greeks[greek] = net_greek

        # Plot payoff diagram if requested
        if plot:
            earliest_expiry = min(pd.to_datetime(details_df['expiry']))

            # Infer strategy if not provided for plotting (already set by identify_strategy)
            if strategy == 'custom' and len(details_df['expiry'].unique()) > 1:
                strategy = 'calendar_spread'
            elif strategy == 'custom':
                print("Warning: Strategy not specified or inferred as 'custom'. Using generic payoff calculation.")

            # Determine stock price internally
            if 'Stock Price' in df.columns:
                stock_price = df['Stock Price'].iloc[0]
            else:
                stock_price = details_df['strike'].iloc[0]  # Fallback to strike price

            # Set stock price range
            range_percent = 0.075  # ±7.5% to achieve a total 15% range
            min_price = stock_price * (1 - range_percent)
            max_price = stock_price * (1 + range_percent)
            stock_price_range = (min_price, max_price)
            prices = np.linspace(stock_price_range[0], stock_price_range[1], 100)
            payoff = np.zeros_like(prices)

            # Calculate payoff for each leg and sum them up
            for _, leg in details_df.iterrows():
                if leg['option_type'].lower() == 'call':
                    leg_payoff = np.maximum(0, prices - leg['strike']) * leg['contracts'] * 100
                else:  # put
                    leg_payoff = np.maximum(0, leg['strike'] - prices) * leg['contracts'] * 100

                # Adjust for action (buy or sell)
                if leg['action'] == 'sell':
                    leg_payoff *= -1

                payoff += leg_payoff

            # Add the net cost to the payoff
            payoff += net_cashflow

            # Calculate breakeven(s) - adjusted for net_cashflow direction
            breakeven = None
            if strategy in ['bull_call_spread', 'bear_put_spread', 'bear_call_spread', 'bull_put_spread']:
                if len(details_df) == 2 and len(details_df['expiry'].unique()) == 1:
                    leg1 = details_df.iloc[0]
                    leg2 = details_df.iloc[1]
                    strikes = sorted([leg1['strike'], leg2['strike']])
                    if net_cashflow < 0:  # Debit spread
                        breakeven = strikes[0] + (-net_cashflow / 100)
                    elif net_cashflow > 0:  # Credit spread
                        breakeven = strikes[1] - (net_cashflow / 100)

            # Determine y-axis range
            y_min = min(0, payoff.min())
            y_max = max(0, payoff.max())
            y_range = y_max - y_min
            y_buffer = y_range * 0.2  # Add some buffer to the y-axis

            # Create Plotly figure
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices, y=payoff, mode='lines', name=strategy))

            # Add current stock price and breakeven lines
            fig.add_vline(x=stock_price, line_dash="dash", line_color="gray", annotation_text="Current Price")
            if breakeven is not None:
                fig.add_vline(x=breakeven, line_dash="dot", line_color="green", annotation_text=f"Breakeven (${breakeven:.2f})")

            # Update layout
            fig.update_layout(
                title=f'{strategy} Payoff at Earliest Expiry ({earliest_expiry.strftime("%Y-%m-%d")})',
                xaxis_title='Stock Price at Expiry',
                yaxis_title='Profit/Loss ($)',
                showlegend=True,
                template='plotly_dark',
                yaxis_range=[y_min - y_buffer, y_max + y_buffer]
            )
            fig.show()

        # Automatically print strategy details
        net_cost_label = "Net Cost (Debit)" if net_cashflow < 0 else "Net Cost (Credit)"
        print(f"Strategy: {strategy}")
        print(f"{net_cost_label}: ${abs(net_cashflow):.2f}")
        print("\nLeg Details:")
        print(details_df.to_string(index=False))
        if greeks and net_greeks:
            print("\nNet Greek Exposure:")
            for greek, value in net_greeks.items():
                print(f"{greek}: {value:.4f}")

        return {
            'net_cost': net_cashflow,
            'details': details_df,
            'strategy': strategy,
            'net_greeks': net_greeks if greeks else None
        }
    except ValueError as e:
        print(f"Error: {e}")
        return {
            'net_cost': np.nan,
            'details': pd.DataFrame(),
            'strategy': None,
            'net_greeks': None
        }

def identify_strategy(legs: List[Dict], net_cashflow: float = 0.0) -> str:
    """
    Identify the options strategy based on leg configuration.
    """
    if len(legs) == 1:
        leg = legs[0]
        if leg['option_type'].lower() == 'call' and leg['action'] == 'buy':
            return 'long_call'
        elif leg['option_type'].lower() == 'put' and leg['action'] == 'buy':
            return 'long_put'
        elif leg['option_type'].lower() == 'call' and leg['action'] == 'sell':
            return 'short_call'
        elif leg['option_type'].lower() == 'put' and leg['action'] == 'sell':
            return 'short_put'
    elif len(legs) == 2:
        leg1, leg2 = legs
        if leg1['option_type'].lower() == 'call' and leg2['option_type'].lower() == 'call':
            if leg1['action'] == 'buy' and leg2['action'] == 'sell':
                if leg1['strike'] < leg2['strike'] and leg1['expiry'] == leg2['expiry']:
                    return 'bull_call_spread'
                elif leg1['expiry'] != leg2['expiry']:
                    return 'calendar_spread'
            elif leg1['action'] == 'sell' and leg2['action'] == 'buy' and leg1['strike'] < leg2['strike'] and leg1['expiry'] == leg2['expiry']:
                return 'bear_call_spread'
        elif leg1['option_type'].lower() == 'put' and leg2['option_type'].lower() == 'put':
            if leg1['action'] == 'buy' and leg2['action'] == 'sell' and leg1['strike'] > leg2['strike'] and leg1['expiry'] == leg2['expiry']:
                return 'bear_put_spread'
            elif leg1['action'] == 'sell' and leg2['action'] == 'buy' and leg1['strike'] > leg2['strike'] and leg1['expiry'] == leg2['expiry']:
                return 'bull_put_spread'
            elif leg1['action'] == 'buy' and leg2['action'] == 'sell' and leg1['expiry'] != leg2['expiry']:
                return 'calendar_spread'
        elif leg1['option_type'].lower() != leg2['option_type'].lower() and leg1['strike'] == leg2['strike'] and leg1['expiry'] == leg2['expiry']:
            if leg1['action'] == 'buy' and leg2['action'] == 'buy':
                return 'long_straddle'
            elif leg1['action'] == 'sell' and leg2['action'] == 'sell':
                return 'short_strangle'
        elif leg1['option_type'].lower() != leg2['option_type'].lower() and leg1['expiry'] == leg2['expiry']:
            if leg1['action'] == 'buy' and leg2['action'] == 'buy' and leg1['strike'] < leg2['strike']:
                return 'long_strangle'
            elif leg1['action'] == 'sell' and leg2['action'] == 'sell' and leg1['strike'] < leg2['strike']:
                return 'short_straddle'
    elif len(legs) == 3:
        if all(leg['option_type'].lower() == 'call' for leg in legs) or all(leg['option_type'].lower() == 'put' for leg in legs):
            strikes = sorted(leg['strike'] for leg in legs)
            if len(set(leg['expiry'] for leg in legs)) == 1:  # All expiries must match
                actions = [leg['action'] for leg in legs]
                if strikes[0] < strikes[1] == strikes[2] < strikes[3] and actions == ['buy', 'sell', 'sell', 'buy']:
                    return 'long_butterfly_spread'
                elif strikes[0] < strikes[1] == strikes[2] < strikes[3] and actions == ['sell', 'buy', 'buy', 'sell']:
                    return 'short_butterfly_spread'
    elif len(legs) == 4:
        calls = [leg for leg in legs if leg['option_type'].lower() == 'call']
        puts = [leg for leg in legs if leg['option_type'].lower() == 'put']
        if len(calls) == 2 and len(puts) == 2 and len(set(leg['expiry'] for leg in legs)) == 1:
            call_strikes = sorted(leg['strike'] for leg in calls)
            put_strikes = sorted(leg['strike'] for leg in puts)
            call_actions = {leg['strike']: leg['action'] for leg in calls}
            put_actions = {leg['strike']: leg['action'] for leg in puts}
            # Check for sell inner, buy outer pattern
            if ((call_actions[call_strikes[0]] == 'sell' and call_actions[call_strikes[1]] == 'buy') or
                (call_actions[call_strikes[1]] == 'sell' and call_actions[call_strikes[0]] == 'buy')) and \
               ((put_actions[put_strikes[0]] == 'sell' and put_actions[put_strikes[1]] == 'buy') or
                (put_actions[put_strikes[1]] == 'sell' and put_actions[put_strikes[0]] == 'buy')) and \
               put_strikes[1] < call_strikes[0]:
                return 'short iron_condor' if net_cashflow > 0 else 'iron_condor'
            elif (puts[0]['action'] == 'buy' and puts[1]['action'] == 'sell' and
                  calls[0]['action'] == 'sell' and calls[1]['action'] == 'buy' and
                  put_strikes[0] < put_strikes[1] == call_strikes[0] < call_strikes[1] and
                  len(set(leg['strike'] for leg in legs)) == 3):
                return 'iron_butterfly'
    return 'custom'

def validate_strategy(legs: List[Dict], strategy: str):
    """
    Validate that the leg configuration matches the specified strategy.
    """
    if strategy == 'bull_call_spread':
        if len(legs) != 2 or any(leg['option_type'].lower() != 'call' for leg in legs) or legs[0]['expiry'] != legs[1]['expiry']:
            raise ValueError("Bull call spread requires two calls with the same expiry.")
        strikes = [leg['strike'] for leg in legs]
        actions = [leg['action'] for leg in legs]
        if not ((actions[0] == 'buy' and actions[1] == 'sell' and strikes[0] < strikes[1]) or (actions[1] == 'buy' and actions[0] == 'sell' and strikes[1] < strikes[0])):
            raise ValueError("Bull call spread: Buy lower strike, sell higher strike.")
    elif strategy == 'bear_put_spread':
        if len(legs) != 2 or any(leg['option_type'].lower() != 'put' for leg in legs) or legs[0]['expiry'] != legs[1]['expiry']:
            raise ValueError("Bear put spread requires two puts with the same expiry.")
        strikes = [leg['strike'] for leg in legs]
        actions = [leg['action'] for leg in legs]
        if not ((actions[0] == 'buy' and actions[1] == 'sell' and strikes[0] > strikes[1]) or (actions[1] == 'buy' and actions[0] == 'sell' and strikes[1] > strikes[0])):
            raise ValueError("Bear put spread: Buy higher strike, sell lower strike.")
    elif strategy == 'bull_put_spread':
        if len(legs) != 2 or any(leg['option_type'].lower() != 'put' for leg in legs) or legs[0]['expiry'] != legs[1]['expiry']:
            raise ValueError("Bull put spread requires two puts with the same expiry.")
        strikes = [leg['strike'] for leg in legs]
        actions = [leg['action'] for leg in legs]
        if not ((actions[0] == 'sell' and actions[1] == 'buy' and strikes[0] > strikes[1]) or (actions[1] == 'sell' and actions[0] == 'buy' and strikes[1] > strikes[0])):
            raise ValueError("Bull put spread: Sell higher strike, buy lower strike.")
    elif strategy == 'bear_call_spread':
        if len(legs) != 2 or any(leg['option_type'].lower() != 'call' for leg in legs) or legs[0]['expiry'] != legs[1]['expiry']:
            raise ValueError("Bear call spread requires two calls with the same expiry.")
        strikes = [leg['strike'] for leg in legs]
        actions = [leg['action'] for leg in legs]
        if not ((actions[0] == 'sell' and actions[1] == 'buy' and strikes[0] < strikes[1]) or (actions[1] == 'sell' and actions[0] == 'buy' and strikes[1] < strikes[0])):
            raise ValueError("Bear call spread: Sell lower strike, buy higher strike.")
    elif strategy == 'long_straddle':
        if len(legs) != 2 or any(leg['option_type'].lower() not in ['call', 'put'] for leg in legs) or legs[0]['expiry'] != legs[1]['expiry']:
            raise ValueError("Long straddle requires one call and one put with the same expiry.")
        strikes = [leg['strike'] for leg in legs]
        actions = [leg['action'] for leg in legs]
        if not (strikes[0] == strikes[1] and actions == ['buy', 'buy']):
            raise ValueError("Long straddle: Buy call and put at the same strike.")
    elif strategy == 'short_strangle':
        if len(legs) != 2 or any(leg['option_type'].lower() not in ['call', 'put'] for leg in legs) or legs[0]['expiry'] != legs[1]['expiry']:
            raise ValueError("Short strangle requires one call and one put with the same expiry.")
        strikes = sorted(leg['strike'] for leg in legs)
        actions = [leg['action'] for leg in legs]
        if not (strikes[0] < strikes[1] and actions == ['sell', 'sell']):
            raise ValueError("Short strangle: Sell call and put at different strikes.")
    elif strategy in ['iron_condor', 'short iron_condor']:
        if len(legs) != 4 or len(set(leg['expiry'] for leg in legs)) != 1:
            raise ValueError("Iron condor requires four legs with the same expiry.")
        calls = [leg for leg in legs if leg['option_type'].lower() == 'call']
        puts = [leg for leg in legs if leg['option_type'].lower() == 'put']
        if len(calls) != 2 or len(puts) != 2:
            raise ValueError("Iron condor requires two calls and two puts.")
        call_strikes = sorted(leg['strike'] for leg in calls)
        put_strikes = sorted(leg['strike'] for leg in puts)
        call_actions = {leg['strike']: leg['action'] for leg in calls}
        put_actions = {leg['strike']: leg['action'] for leg in puts}
        if not (call_strikes[0] < call_strikes[1] and put_strikes[0] < put_strikes[1] and
                ((call_actions[call_strikes[0]] == 'sell' and call_actions[call_strikes[1]] == 'buy') or
                 (call_actions[call_strikes[1]] == 'sell' and call_actions[call_strikes[0]] == 'buy')) and
                ((put_actions[put_strikes[0]] == 'sell' and put_actions[put_strikes[1]] == 'buy') or
                 (put_actions[put_strikes[1]] == 'sell' and put_actions[put_strikes[0]] == 'buy')) and
                put_strikes[1] < call_strikes[0]):
            raise ValueError("Iron condor: Sell inner put, buy outer put, sell inner call, buy outer call.")
    elif strategy == 'long_call':
        if len(legs) != 1 or legs[0]['option_type'].lower() != 'call' or legs[0]['action'] != 'buy':
            raise ValueError("Long call requires one call with action 'buy'.")
    elif strategy == 'long_put':
        if len(legs) != 1 or legs[0]['option_type'].lower() != 'put' or legs[0]['action'] != 'buy':
            raise ValueError("Long put requires one put with action 'buy'.")
    elif strategy == 'short_call':
        if len(legs) != 1 or legs[0]['option_type'].lower() != 'call' or legs[0]['action'] != 'sell':
            raise ValueError("Short call requires one call with action 'sell'.")
    elif strategy == 'short_put':
        if len(legs) != 1 or legs[0]['option_type'].lower() != 'put' or legs[0]['action'] != 'sell':
            raise ValueError("Short put requires one put with action 'sell'.")
    elif strategy == 'short_straddle':
        if len(legs) != 2 or any(leg['option_type'].lower() not in ['call', 'put'] for leg in legs) or legs[0]['expiry'] != legs[1]['expiry']:
            raise ValueError("Short straddle requires one call and one put with the same expiry.")
        strikes = [leg['strike'] for leg in legs]
        actions = [leg['action'] for leg in legs]
        if not (strikes[0] == strikes[1] and actions == ['sell', 'sell']):
            raise ValueError("Short straddle: Sell call and put at the same strike.")
    elif strategy == 'long_strangle':
        if len(legs) != 2 or any(leg['option_type'].lower() not in ['call', 'put'] for leg in legs) or legs[0]['expiry'] != legs[1]['expiry']:
            raise ValueError("Long strangle requires one call and one put with the same expiry.")
        strikes = sorted(leg['strike'] for leg in legs)
        actions = [leg['action'] for leg in legs]
        if not (strikes[0] < strikes[1] and actions == ['buy', 'buy']):
            raise ValueError("Long strangle: Buy call and put at different strikes with call strike > put strike.")
    elif strategy == 'iron_butterfly':
        if len(legs) != 4 or len(set(leg['expiry'] for leg in legs)) != 1:
            raise ValueError("Iron butterfly requires four legs with the same expiry.")
        calls = [leg for leg in legs if leg['option_type'].lower() == 'call']
        puts = [leg for leg in legs if leg['option_type'].lower() == 'put']
        if len(calls) != 2 or len(puts) != 2:
            raise ValueError("Iron butterfly requires two calls and two puts.")
        call_strikes = sorted(leg['strike'] for leg in calls)
        put_strikes = sorted(leg['strike'] for leg in puts)
        if not (puts[0]['action'] == 'buy' and puts[1]['action'] == 'sell' and
                calls[0]['action'] == 'sell' and calls[1]['action'] == 'buy' and
                put_strikes[0] < put_strikes[1] == call_strikes[0] < call_strikes[1] and
                len(set(leg['strike'] for leg in legs)) == 3):
            raise ValueError("Iron butterfly: Buy lower put, sell middle put/call, buy higher call.")
    elif strategy == 'long_butterfly_spread':
        if len(legs) != 3 or not (all(leg['option_type'].lower() == 'call' for leg in legs) or all(leg['option_type'].lower() == 'put' for leg in legs)) or len(set(leg['expiry'] for leg in legs)) != 1:
            raise ValueError("Long butterfly spread requires three calls or three puts with the same expiry.")
        strikes = sorted(leg['strike'] for leg in legs)
        actions = [leg['action'] for leg in legs]
        if not (strikes[0] < strikes[1] == strikes[2] < strikes[3] and actions == ['buy', 'sell', 'sell', 'buy']):
            raise ValueError("Long butterfly spread: Buy lower strike, sell two middle strikes, buy higher strike.")
    elif strategy == 'short_butterfly_spread':
        if len(legs) != 3 or not (all(leg['option_type'].lower() == 'call' for leg in legs) or all(leg['option_type'].lower() == 'put' for leg in legs)) or len(set(leg['expiry'] for leg in legs)) != 1:
            raise ValueError("Short butterfly spread requires three calls or three puts with the same expiry.")
        strikes = sorted(leg['strike'] for leg in legs)
        actions = [leg['action'] for leg in legs]
        if not (strikes[0] < strikes[1] == strikes[2] < strikes[3] and actions == ['sell', 'buy', 'buy', 'sell']):
            raise ValueError("Short butterfly spread: Sell lower strike, buy two middle strikes, sell higher strike.")
    elif strategy == 'calendar_spread':
        if len(legs) != 2 or legs[0]['option_type'].lower() != legs[1]['option_type'].lower() or legs[0]['strike'] != legs[1]['strike'] or legs[0]['expiry'] == legs[1]['expiry']:
            raise ValueError("Calendar spread requires two legs of the same option type and strike with different expiries.")
        actions = [leg['action'] for leg in legs]
        if not ((actions[0] == 'buy' and actions[1] == 'sell') or (actions[1] == 'buy' and actions[0] == 'sell')):
            raise ValueError("Calendar spread: Buy one leg, sell the other.")
    elif strategy == 'diagonal_spread':
        if len(legs) != 2 or legs[0]['option_type'].lower() != legs[1]['option_type'].lower() or legs[0]['expiry'] == legs[1]['expiry']:
            raise ValueError("Diagonal spread requires two legs of the same option type with different strikes and expiries.")
        strikes = [leg['strike'] for leg in legs]
        actions = [leg['action'] for leg in legs]
        if not ((actions[0] == 'buy' and actions[1] == 'sell') or (actions[1] == 'buy' and actions[0] == 'sell')):
            raise ValueError("Diagonal spread: Buy one leg, sell the other.")