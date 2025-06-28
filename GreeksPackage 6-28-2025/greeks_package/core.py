import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import warnings
import plotly.express as px
import plotly.graph_objects as go

# Suppress all warnings (optional, but not recommended for debugging)
warnings.simplefilter(action='ignore', category=FutureWarning)

# -------------------- OPTION CHAIN UTILS --------------------

def download_options(
    ticker_symbol,
    opt_type='c',
    max_days=60,
    lower_moneyness=0.95,
    upper_moneyness=1.05,
    price=False  # Optional: include underlying price column
):
    """
    Download and filter an option chain from Yahoo! Finance.

    Parameters
    ----------
    ticker_symbol : str
        Underlying ticker symbol.
    opt_type : {'c', 'p'}, default 'c'
        Retrieve calls ('c') or puts ('p').
    max_days : int, default 60
        Maximum days-to-expiry to include.
    lower_moneyness, upper_moneyness : float, default 0.95, 1.05
        Keep strikes within ``lower_moneyness * S0`` and ``upper_moneyness * S0``.
    price : bool, default False
        If *True*, add a column with the current underlying price.

    Returns
    -------
    pandas.DataFrame
        One row per option with standard Yahoo! columns plus::

            • expiry           – Expiration datetime.
            • Days to Expiry   – Fractional days-to-expiry.
            • Mid-Point Price  – (bid + ask)/2.
            • Stock Price      – Current underlying (if *price*).

    Notes
    -----
    The function is intentionally lightweight (no multithreading) as Yahoo!
    already throttles aggressively.  Expect ~0.5–1 s per call depending on the
    number of expiries.
    """

    tk = yf.Ticker(ticker_symbol)

    # Current underlying price
    S0 = tk.history(period="1d")['Close'].iloc[-1]
    lo_strike, hi_strike = S0 * lower_moneyness, S0 * upper_moneyness

    cols = [
        'contractSymbol', 'inTheMoney', 'strike', 'lastPrice', 'bid', 'ask',
        'volume', 'openInterest', 'impliedVolatility'
    ]
    out = pd.DataFrame(columns=cols + ['expiry'])

    for expiry_str in tk.options:
        exp = pd.to_datetime(expiry_str)
        dt_exp = round((exp - datetime.now()).total_seconds() / 86_400, 2)
        if dt_exp > max_days:
            continue

        chain = tk.option_chain(expiry_str)
        data = chain.calls if opt_type.lower() == 'c' else chain.puts
        data = data[(data['strike'] >= lo_strike) & (data['strike'] <= hi_strike)].copy()
        if data.empty:
            continue
        data['expiry'] = exp
        out = pd.concat([out, data[cols + ['expiry']]], ignore_index=True)

    if out.empty:
        return out

    out['Days to Expiry'] = (
        (pd.to_datetime(out['expiry']) - datetime.now()).dt.total_seconds() / 86_400
    ).round(2)
    out['Mid-Point Price'] = ((out['bid'] + out['ask']) / 2).round(4)
    out['impliedVolatility'] = out['impliedVolatility'].round(2)
    if price:
        out['Stock Price'] = round(S0, 4)
    return out

# -------------------- BS-HELPERS --------------------

def compute_d1(S, K, t, r, sigma, eps=1e-9):
    t = max(t, eps)
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

def compute_d2(S, K, t, r, sigma, eps=1e-9):
    return compute_d1(S, K, t, r, sigma, eps) - sigma * np.sqrt(t)

def compute_d1_d2(S, K, t, r, sigma, eps=1e-9):
    d1 = compute_d1(S, K, t, r, sigma, eps)
    return d1, d1 - sigma * np.sqrt(t)

# -------------------- PRICING --------------------

def bsm_price(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05) -> float:
    """Black-Scholes price for *row* of an option chain."""
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
    return yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]

def delta(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1 = compute_d1(S, K, T, r, sigma)
    return round(norm.cdf(d1) if option_type.lower() == 'c' else norm.cdf(d1) - 1, 4)

def theta(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    if T <= 0:
        return np.nan
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    v = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    r_term = r * K * np.exp(-r * T) * (norm.cdf(d2) if option_type.lower() == 'c' else -norm.cdf(-d2))
    return round((v + r_term) / 365, 4)

def vega(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1 = compute_d1(S, K, T, r, sigma)
    return round(S * np.sqrt(T) * norm.pdf(d1) * 0.01, 4)  # per 1% vol

def rho(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d2 = compute_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    val = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type.lower() == 'c' else -norm.cdf(-d2))
    return round(val * 0.01, 4)  # per 1%

# 2nd-order

def gamma(row: pd.Series, ticker: str, option_type: str = 'c', r: float = 0.05, eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1 = compute_d1(S, K, T, r, sigma)
    return round(norm.pdf(d1) / (S * sigma * np.sqrt(T)), 4)

def vanna(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round(np.exp(-r*T) * norm.pdf(d1) * (d2 / sigma), 4)

def volga(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    v = vega(row, ticker, option_type, r, eps)
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round(v * (d1 * d2) / sigma, 4)

def charm(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round(-norm.pdf(d1) * (2*r*T - d2*sigma*np.sqrt(T)) / (2*T), 4)

def veta(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    term1 = (r * d1) / (sigma * np.sqrt(T))
    term2 = (1 + d1 * d2) / (2 * T)
    return round(-S * norm.pdf(d1) * np.sqrt(T) * (term1 - term2), 4)

# 3rd-order

def color(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round((norm.pdf(d1) / (2*S*T*sigma*np.sqrt(T))) * (2*r*T + 1 - d1*d2), 4)

def speed(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, _ = compute_d1_d2(S, K, T, r, sigma)
    return round((norm.pdf(d1) / (S**2 * sigma * np.sqrt(T))) * ((d1/(sigma*np.sqrt(T))) - 1), 4)

def ultima(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    v = vega(row, ticker, option_type, r, eps)
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round(-v / sigma**2 * (d1*d2*(1 - d1*d2) + d1**2 + d2**2), 4)

def zomma(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9):
    g = gamma(row, ticker, option_type, r, eps)
    S, K, T, sigma = _get_spot(ticker), row['strike'], max(row['Days to Expiry']/365, eps), max(row['impliedVolatility'], 0.01)
    d1, d2 = compute_d1_d2(S, K, T, r, sigma)
    return round((g * (d1*d2 - 1)) / sigma, 4)

# -------------------- WRAPPERS --------------------

def first_order(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9) -> pd.Series:
    vals = {
        'Delta': delta(row, ticker, option_type, r, eps),
        'Vega': vega(row, ticker, option_type, r, eps),
        'Theta': theta(row, ticker, option_type, r, eps),
        'Rho': rho(row, ticker, option_type, r, eps),
    }
    return pd.Series(vals)

def second_order(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9) -> pd.Series:
    vals = {
        'Gamma': gamma(row, ticker, option_type, r, eps),
        'Vanna': vanna(row, ticker, r, option_type, eps),
        'Volga': volga(row, ticker, r, option_type, eps),
        'Veta': veta(row, ticker, r, option_type, eps),
        'Charm': charm(row, ticker, r, option_type, eps),
    }
    return pd.Series(vals)

def third_order(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9) -> pd.Series:
    vals = {
        'Color': color(row, ticker, r, option_type, eps),
        'Speed': speed(row, ticker, r, option_type, eps),
        'Ultima': ultima(row, ticker, r, option_type, eps),
        'Zomma': zomma(row, ticker, r, option_type, eps),
    }
    return pd.Series(vals)

def greeks(row: pd.Series, ticker: str, r: float = 0.05, option_type: str = 'c', eps: float = 1e-9) -> pd.Series:
    return pd.concat([
        first_order(row, ticker, r, option_type, eps),
        second_order(row, ticker, r, option_type, eps),
        third_order(row, ticker, r, option_type, eps),
    ])

# -------------------- COMBINATION HELPER --------------------

def comb(*dfs: pd.DataFrame) -> pd.DataFrame:
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

# -------------------- PUBLIC EXPORTS --------------------
__all__ = [
    'download_options', 'bsm_price', 'monte_carlo_price',
    # Greeks (row-level)
    'delta', 'theta', 'vega', 'rho', 'gamma',
    'vanna', 'volga', 'veta', 'charm',
    'color', 'speed', 'ultima', 'zomma',
    # Wrappers
    'first_order', 'second_order', 'third_order', 'greeks',
    # Utils
    'comb', 'surf_scatter', 'surface_plot',
] 