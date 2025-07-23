# greeks_package – Complete Usage Guide

> **Version:** 1.1.0  |  **Python ≥** 3.9  |  **Author:** JR Concepcion -- Lehigh MFE 

This comprehensive guide covers every aspect of **greeks_package** - from basic usage to advanced risk management applications. Whether you're a quantitative trader, risk manager, or options researcher, this guide will help you leverage the full power of the package.

---

## 📋 Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Quick Start](#2-quick-start)
3. [Function Reference](#3-function-reference)
4. [Advanced Usage Patterns](#4-advanced-usage-patterns)
5. [Performance & Best Practices](#5-performance--best-practices)
6. [Error Handling](#6-error-handling)
7. [Examples & Use Cases](#7-examples--use-cases)

---

## 1. Installation & Setup

### 1.1 Installation Options

```bash
# From PyPI (recommended)
pip install greeks-package

# From source (development)
git clone https://github.com/JRCon1/greeks-package.git
cd greeks-package
pip install -e .

# With specific dependencies
pip install greeks-package[dev]  # includes development tools
```

### 1.2 Dependencies

**Required:**
- `numpy>=1.23` - Numerical computations
- `pandas>=1.5` - Data manipulation
- `scipy>=1.10` - Statistical functions
- `yfinance>=0.2` - Market data
- `plotly>=5.19` - Interactive visualization

**Python Compatibility:** 3.9+

### 1.3 Basic Import Patterns

```python
# Standard import (recommended)
import greeks_package as gp

# Selective imports
from greeks_package import download_options, greeks, first_order

# Core module for advanced users
from greeks_package.core import delta, gamma, vanna, surf_scatter, greek_plot, iv_plot
```

---

## 2. Quick Start

### 2.1 The 3-Step Workflow

```python
import greeks_package as gp

# Step 1: Download option chain
opts = gp.download_options("AAPL", opt_type="c", max_days=30)

# Step 2: Calculate Greeks
all_greeks = opts.apply(gp.greeks, axis=1, ticker="AAPL")

# Step 3: Combine and analyze
full_data = opts.join(all_greeks)
print(full_data[['strike', 'lastPrice', 'Delta', 'Gamma', 'Vega']].head())
```

### 2.2 Key Concepts

- **Row-wise Processing**: All Greek functions work on individual option rows
- **Ticker Required**: Current stock price is fetched automatically for calculations
- **Option Type**: Specify 'c' for calls, 'p' for puts
- **Flexible Output**: Functions return pandas Series for easy DataFrame integration

---

## 3. Function Reference

### 3.1 Data Acquisition

#### `download_options(ticker_symbol, opt_type='c', max_days=60, lower_moneyness=0.95, upper_moneyness=1.05, price=False)`

Downloads and filters option chains from Yahoo Finance.

**Parameters:**
- `ticker_symbol` (str): Stock ticker (e.g., "AAPL", "TSLA")
- `opt_type` (str): Option type - 'c' (calls), 'p' (puts), 'all' (both calls and puts)
- `max_days` (int): Maximum days to expiration filter
- `lower_moneyness` (float): Lower strike/spot ratio bound
- `upper_moneyness` (float): Upper strike/spot ratio bound
- `price` (bool): Include current stock price column

**Returns:** `pd.DataFrame` with filtered options data

**Example:**
```python
# Conservative filter: calls only, 30 days, tight moneyness
opts = gp.download_options("AAPL", opt_type="c", max_days=30, 
                          lower_moneyness=0.95, upper_moneyness=1.05)

# Download both calls and puts together
opts_all = gp.download_options("AAPL", opt_type="all", max_days=90,
                              lower_moneyness=0.8, upper_moneyness=1.2, price=True)
```

#### `multi_download(ticker_symbols, opt_type='c', max_days=60, lower_moneyness=0.95, upper_moneyness=1.05, price=False)` **NEW!**

Downloads and filters option chains for multiple tickers simultaneously.

**Parameters:**
- `ticker_symbols` (List[str]): List of stock tickers (e.g., ['AAPL', 'MSFT', 'GOOGL'])
- `opt_type` (str): Option type - 'c' (calls), 'p' (puts), 'all' (both calls and puts)
- `max_days` (int): Maximum days to expiration filter
- `lower_moneyness` (float): Lower strike/spot ratio bound
- `upper_moneyness` (float): Upper strike/spot ratio bound
- `price` (bool): Include current stock price column for each ticker

**Returns:** `pd.DataFrame` with filtered options data and 'Ticker' column identifying the source

**Key Features:**
- **Automatic Error Handling**: Continues processing other tickers if one fails
- **Ticker Identification**: Adds 'Ticker' column to identify source of each option
- **Consistent Interface**: Same parameters as `download_options()` but for multiple tickers
- **Efficient Processing**: Handles multiple downloads with built-in error recovery

**Examples:**

```python
# Download calls for tech portfolio
tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
tech_opts = gp.multi_download(
    ticker_symbols=tech_tickers,
    opt_type="c",
    max_days=45,
    price=True
)

print(f"Downloaded {len(tech_opts)} options across {len(tech_tickers)} tickers")

# Analyze by ticker
for ticker in tech_tickers:
    ticker_data = tech_opts[tech_opts['Ticker'] == ticker]
    print(f"{ticker}: {len(ticker_data)} options")
```

```python
# Download both calls and puts for multiple tickers
portfolio_tickers = ['SPY', 'QQQ', 'IWM']
portfolio_opts = gp.multi_download(
    ticker_symbols=portfolio_tickers,
    opt_type="all",  # Both calls and puts
    max_days=60,
    lower_moneyness=0.9,
    upper_moneyness=1.1,
    price=True
)

# Separate calls and puts
calls = portfolio_opts[~portfolio_opts['contractSymbol'].str.contains('P')]
puts = portfolio_opts[portfolio_opts['contractSymbol'].str.contains('P')]

print(f"Portfolio analysis: {len(calls)} calls, {len(puts)} puts")
```

```python
# Calculate Greeks for multiple tickers
def calc_greeks_multi_ticker(row):
    """Calculate Greeks using the ticker from the row"""
    return gp.greeks(row, ticker=row['Ticker'])

# Apply Greeks calculation using each row's ticker
multi_greeks = portfolio_opts.apply(calc_greeks_multi_ticker, axis=1)
full_portfolio = portfolio_opts.join(multi_greeks)

# Analyze average Greeks by ticker
for ticker in portfolio_tickers:
    ticker_data = full_portfolio[full_portfolio['Ticker'] == ticker]
    avg_delta = ticker_data['Delta'].mean()
    print(f"{ticker} Average Delta: {avg_delta:.3f}")
```

**Error Handling:**
The function automatically handles individual ticker failures and continues processing:

```python
# Some tickers may fail, function continues with valid ones
mixed_tickers = ['AAPL', 'INVALID_TICKER', 'MSFT', 'ANOTHER_BAD_TICKER']
try:
    opts = gp.multi_download(mixed_tickers, opt_type="c")
    print(f"Successfully downloaded data for valid tickers")
except ValueError as e:
    print(f"No valid data retrieved: {e}")
```

### 3.2 Greek Calculation Functions

#### Wrapper Functions

##### `greeks(row, ticker, r=0.05, option_type='c', eps=1e-9)`
Calculates all 13 Greeks in one call.

**Returns:** Series with all Greeks: Delta, Vega, Theta, Rho, Gamma, Vanna, Volga, Veta, Charm, Color, Speed, Ultima, Zomma

##### `first_order(row, ticker, r=0.05, option_type='c', eps=1e-9)`
First-order Greeks: Delta, Vega, Theta, Rho

##### `second_order(row, ticker, r=0.05, option_type='c', eps=1e-9)`
Second-order Greeks: Gamma, Vanna, Volga, Veta, Charm

##### `third_order(row, ticker, r=0.05, option_type='c', eps=1e-9)`
Third-order Greeks: Color, Speed, Ultima, Zomma

#### Individual Greek Functions

All individual Greeks follow the same signature:
```python
function_name(row: pd.Series, ticker: str, option_type: str = 'c', 
              r: float = 0.05, eps: float = 1e-9) -> float
```

**First-Order Greeks:**
- `delta()` - Price sensitivity to underlying (Δ)
- `vega()` - Price sensitivity to volatility (ν) 
- `theta()` - Time decay (Θ)
- `rho()` - Interest rate sensitivity (ρ)

**Second-Order Greeks:**
- `gamma()` - Delta sensitivity to underlying (Γ)
- `vanna()` - Delta sensitivity to volatility
- `volga()` - Vega sensitivity to volatility
- `charm()` - Delta time decay
- `veta()` - Vega time decay

**Third-Order Greeks:**
- `color()` - Gamma time decay
- `speed()` - Gamma sensitivity to underlying
- `ultima()` - Volga sensitivity to volatility
- `zomma()` - Gamma sensitivity to volatility

### 3.3 Utility Functions

#### `comb(*dfs)`
Combines multiple DataFrames with automatic duplicate column handling.

```python
# Combine options data with multiple Greek calculations
base_data = gp.download_options("AAPL")
first_greeks = base_data.apply(gp.first_order, axis=1, ticker="AAPL")
second_greeks = base_data.apply(gp.second_order, axis=1, ticker="AAPL")

# Combine all (handles duplicate columns automatically)
full_dataset = gp.comb(base_data, first_greeks, second_greeks)
```

#### `bsm_price(row, ticker, option_type='c', r=0.05)`
Black-Scholes theoretical option pricing.

```python
# Compare theoretical vs market prices
opts['Theoretical_Price'] = opts.apply(gp.bsm_price, axis=1, ticker="AAPL")
opts['Price_Difference'] = opts['Theoretical_Price'] - opts['lastPrice']
```

### 3.4 Visualization Functions

#### 3D Plotting (v1.0+)
- `surf_scatter(df, ticker, z='delta', option_type='c', r=0.05, **kwargs)` – Interactive 3D scatter plot of Greeks vs strike and time
- `surface_plot(df, ticker, z='impliedVolatility', option_type='c', r=0.05, **kwargs)` – Smooth 3D surface plot for comprehensive Greek visualization

#### 🆕 Interactive Analysis (v1.1.0)
- `greek_plot(df, greek_col, x_axis='Days to Expiry', return_fig=False, **kwargs)` – Greek values vs time with strike selection
- `iv_plot(df, ticker, return_fig=False, **kwargs)` – Implied volatility term structure  
- `oi_plot(df, ticker, return_fig=False, **kwargs)` – Open interest distribution analysis
- `vol_curve(df, ticker, return_fig=False, **kwargs)` – Volatility smile/skew curves

```python
# 3D visualization
gp.surf_scatter(opts, "AAPL", z="delta")        # Delta scatter points
gp.surface_plot(opts, "AAPL", z="vega")         # Vega surface

# NEW v1.1.0: Interactive line plots with dropdowns
gp.greek_plot(opts_with_greeks, greek_col="Delta")     # Delta vs time by strike
gp.iv_plot(opts, "AAPL")                               # IV term structure  
gp.oi_plot(opts, "AAPL")                               # Open interest patterns
gp.vol_curve(opts, "AAPL")                             # Volatility smiles

# Advanced: Return figures for customization or saving
fig = gp.iv_plot(opts, "AAPL", return_fig=True)
fig.write_html("iv_analysis.html")  # Save interactive plot
```

---

## 4. Advanced Usage Patterns

### 4.1 Multi-Asset Portfolio Analysis

```python
import greeks_package as gp
import pandas as pd

def analyze_multi_asset_portfolio(tickers, max_days=45):
    """Analyze Greeks across multiple underlyings"""
    all_data = []
    
    for ticker in tickers:
        try:
            # Download options for each ticker
            opts = gp.download_options(ticker, opt_type="all", 
                                     max_days=max_days, price=True)
            
            # Calculate all Greeks
            greeks_data = opts.apply(gp.greeks, axis=1, ticker=ticker)
            
            # Combine and add ticker column
            full_data = opts.join(greeks_data)
            full_data['Underlying'] = ticker
            all_data.append(full_data)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# Analyze tech portfolio
tech_portfolio = analyze_multi_asset_portfolio(['AAPL', 'MSFT', 'GOOGL', 'TSLA'])
```

### 4.2 Greeks-Based Option Screening

```python
def screen_options_by_greeks(ticker, delta_min=0.3, delta_max=0.7, 
                           gamma_min=0.01, vega_max=50):
    """Screen options based on Greek criteria"""
    
    # Download options
    opts = gp.download_options(ticker, opt_type="c", max_days=60)
    
    # Calculate Greeks
    greeks_data = opts.apply(gp.greeks, axis=1, ticker=ticker)
    full_data = opts.join(greeks_data)
    
    # Apply Greek-based filters
    filtered = full_data[
        (full_data['Delta'] >= delta_min) & 
        (full_data['Delta'] <= delta_max) &
        (full_data['Gamma'] >= gamma_min) &
        (full_data['Vega'] <= vega_max)
    ]
    
    return filtered.sort_values('Gamma', ascending=False)

# Find high-gamma, moderate-delta calls
high_gamma_calls = screen_options_by_greeks("AAPL", delta_min=0.4, 
                                          delta_max=0.6, gamma_min=0.02)
```

### 4.3 Time Series Analysis

```python
def greek_time_series_analysis(ticker, option_strike, days_range=range(1, 91)):
    """Analyze how Greeks change over time to expiration"""
    
    results = []
    base_opts = gp.download_options(ticker, opt_type="c", max_days=90)
    target_option = base_opts[base_opts['strike'] == option_strike].iloc[0]
    
    for days in days_range:
        # Simulate different time to expiry
        modified_option = target_option.copy()
        modified_option['Days to Expiry'] = days
        
        # Calculate Greeks for this time point
        greeks = gp.greeks(modified_option, ticker=ticker)
        greeks['Days_to_Expiry'] = days
        results.append(greeks)
    
    return pd.DataFrame(results)

# Analyze how Greeks decay over time
time_decay_analysis = greek_time_series_analysis("AAPL", 150)
```

### 4.4 Risk Management Dashboard

```python
def create_risk_dashboard(portfolio_positions):
    """Create a comprehensive risk dashboard using Greeks"""
    
    dashboard_data = []
    
    for position in portfolio_positions:
        ticker = position['ticker']
        contracts = position['contracts']
        
        # Get option data
        opts = gp.download_options(ticker, opt_type="all", max_days=60)
        greeks_data = opts.apply(gp.greeks, axis=1, ticker=ticker)
        
        # Calculate position-level Greeks
        position_greeks = greeks_data * contracts
        
        # Aggregate to portfolio level
        portfolio_greeks = {
            'Ticker': ticker,
            'Contracts': contracts,
            'Portfolio_Delta': position_greeks['Delta'].sum(),
            'Portfolio_Gamma': position_greeks['Gamma'].sum(),
            'Portfolio_Vega': position_greeks['Vega'].sum(),
            'Portfolio_Theta': position_greeks['Theta'].sum(),
        }
        
        dashboard_data.append(portfolio_greeks)
    
    return pd.DataFrame(dashboard_data)

# Example portfolio
portfolio = [
    {'ticker': 'AAPL', 'contracts': 10},
    {'ticker': 'TSLA', 'contracts': -5},
    {'ticker': 'MSFT', 'contracts': 15}
]

risk_summary = create_risk_dashboard(portfolio)
```

### 4.1 Multi-Asset Portfolio Analysis with Multi-Download

```python
import greeks_package as gp
import pandas as pd

def analyze_sector_portfolio(sector_tickers, max_days=45):
    """Enhanced multi-asset analysis using multi_download"""
    
    # Download options for entire sector at once
    print(f"Downloading options for {len(sector_tickers)} tickers...")
    sector_opts = gp.multi_download(
        ticker_symbols=sector_tickers,
        opt_type="all",  # Both calls and puts
        max_days=max_days,
        price=True
    )
    
    # Calculate Greeks for all options using their respective tickers
    def calc_greeks_by_ticker(row):
        option_type = 'c' if 'C' in row['contractSymbol'] else 'p'
        return gp.greeks(row, ticker=row['Ticker'], option_type=option_type)
    
    print("🧮 Calculating Greeks for all options...")
    all_greeks = sector_opts.apply(calc_greeks_by_ticker, axis=1)
    
    # Combine data
    full_sector_data = sector_opts.join(all_greeks)
    
    # Sector-level analysis
    sector_summary = {}
    for ticker in sector_tickers:
        ticker_data = full_sector_data[full_sector_data['Ticker'] == ticker]
        calls = ticker_data[~ticker_data['contractSymbol'].str.contains('P')]
        puts = ticker_data[ticker_data['contractSymbol'].str.contains('P')]
        
        sector_summary[ticker] = {
            'total_options': len(ticker_data),
            'calls': len(calls),
            'puts': len(puts),
            'avg_delta': ticker_data['Delta'].mean(),
            'avg_gamma': ticker_data['Gamma'].mean(),
            'avg_vega': ticker_data['Vega'].mean()
        }
    
    return full_sector_data, sector_summary

# Analyze tech sector
tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'META']
tech_data, tech_summary = analyze_sector_portfolio(tech_tickers)

# Display results
for ticker, stats in tech_summary.items():
    print(f"{ticker}: {stats['calls']} calls, {stats['puts']} puts, "
          f"Δ={stats['avg_delta']:.3f}")
```

### 4.2 Cross-Asset Greeks Comparison

```python
def compare_assets_greeks(tickers_list, focus_greek='Delta'):
    """Compare specific Greek across multiple assets"""
    
    # Download all assets at once
    multi_data = gp.multi_download(
        ticker_symbols=tickers_list,
        opt_type="c",  # Focus on calls for comparison
        max_days=30,
        lower_moneyness=0.95,
        upper_moneyness=1.05
    )
    
    # Calculate focused Greek for all assets
    greek_func = getattr(gp, focus_greek.lower())
    multi_data[focus_greek] = multi_data.apply(
        lambda row: greek_func(row, ticker=row['Ticker']), axis=1
    )
    
    # Create comparison DataFrame
    comparison = multi_data.groupby('Ticker')[focus_greek].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(4)
    
    return comparison

# Compare Delta across different asset classes
asset_mix = ['SPY', 'QQQ', 'GLD', 'TLT', 'VIX']  # Stocks, Tech, Gold, Bonds, Volatility
delta_comparison = compare_assets_greeks(asset_mix, 'Delta')
print("Delta Comparison Across Asset Classes:")
print(delta_comparison)
```

### 4.3 Enhanced Calls vs Puts Analysis

```python
def deep_calls_puts_analysis(ticker, max_days=60):
    """Comprehensive calls vs puts analysis using opt_type='all'"""
    
    # Download both calls and puts together
    all_opts = gp.download_options(
        ticker, 
        opt_type="all",  # Key feature: both types at once
        max_days=max_days,
        lower_moneyness=0.8,
        upper_moneyness=1.2,
        price=True
    )
    
    # Separate calls and puts
    calls = all_opts[~all_opts['contractSymbol'].str.contains('P')].copy()
    puts = all_opts[all_opts['contractSymbol'].str.contains('P')].copy()
    
    # Calculate Greeks for each type
    call_greeks = calls.apply(gp.greeks, axis=1, ticker=ticker, option_type='c')
    put_greeks = puts.apply(gp.greeks, axis=1, ticker=ticker, option_type='p')
    
    # Combine data
    calls_full = calls.join(call_greeks)
    puts_full = puts.join(put_greeks)
    
    # Comparative analysis
    analysis = {
        'calls_count': len(calls_full),
        'puts_count': len(puts_full),
        'call_put_ratio': len(calls_full) / len(puts_full),
        
        # Delta analysis
        'calls_avg_delta': calls_full['Delta'].mean(),
        'puts_avg_delta': puts_full['Delta'].mean(),
        'delta_spread': calls_full['Delta'].mean() - puts_full['Delta'].mean(),
        
        # Gamma analysis
        'calls_avg_gamma': calls_full['Gamma'].mean(),
        'puts_avg_gamma': puts_full['Gamma'].mean(),
        
        # Time decay analysis
        'calls_avg_theta': calls_full['Theta'].mean(),
        'puts_avg_theta': puts_full['Theta'].mean(),
        
        # Volatility sensitivity
        'calls_avg_vega': calls_full['Vega'].mean(),
        'puts_avg_vega': puts_full['Vega'].mean(),
    }
    
    return calls_full, puts_full, analysis

# Analyze AAPL calls vs puts
calls_data, puts_data, analysis = deep_calls_puts_analysis("AAPL")

print(f"AAPL Options Analysis:")
print(f"Call/Put Ratio: {analysis['call_put_ratio']:.2f}")
print(f"Delta Spread (Calls - Puts): {analysis['delta_spread']:.3f}")
print(f"Average Theta - Calls: {analysis['calls_avg_theta']:.4f}, Puts: {analysis['puts_avg_theta']:.4f}")
```

### 4.4 Multi-Ticker Strategy Screening

```python
def screen_multi_ticker_strategies(tickers, strategy_type='high_gamma'):
    """Screen for strategy opportunities across multiple tickers"""
    
    # Download comprehensive data for all tickers
    all_data = gp.multi_download(
        ticker_symbols=tickers,
        opt_type="all",
        max_days=45,
        price=True
    )
    
    # Calculate Greeks for all options
    def calc_greeks_with_correct_type(row):
        opt_type = 'c' if 'C' in row['contractSymbol'] else 'p'
        return gp.greeks(row, ticker=row['Ticker'], option_type=opt_type)
    
    all_greeks = all_data.apply(calc_greeks_with_correct_type, axis=1)
    full_data = all_data.join(all_greeks)
    
    # Apply strategy-specific screening
    if strategy_type == 'high_gamma':
        # Screen for high gamma opportunities
        screened = full_data[
            (full_data['Gamma'] > 0.02) &
            (full_data['Delta'].abs() > 0.3) &
            (full_data['Days to Expiry'] <= 30)
        ]
    elif strategy_type == 'low_vega':
        # Screen for low volatility sensitivity
        screened = full_data[
            (full_data['Vega'] < 10) &
            (full_data['Delta'].abs() > 0.5)
        ]
    else:
        screened = full_data
    
    # Rank by ticker
    results = {}
    for ticker in tickers:
        ticker_data = screened[screened['Ticker'] == ticker]
        if len(ticker_data) > 0:
            results[ticker] = ticker_data.nlargest(5, 'Gamma')
    
    return results

# Screen for high-gamma opportunities across tech stocks
tech_tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
opportunities = screen_multi_ticker_strategies(tech_tickers, 'high_gamma')

for ticker, options in opportunities.items():
    print(f"\n{ticker} High-Gamma Opportunities:")
    print(options[['contractSymbol', 'strike', 'Delta', 'Gamma', 'Vega']].head(3))
```

---

## 5. Performance & Best Practices

### 5.1 Performance Optimization

```python
# ✅ Good: Vectorized operations
all_greeks = opts.apply(gp.greeks, axis=1, ticker="AAPL")

# ❌ Avoid: Row-by-row loops
# for idx, row in opts.iterrows():  # Slow!
#     greeks = gp.greeks(row, ticker="AAPL")

# ✅ Good: Batch similar calculations
first_order_batch = opts.apply(gp.first_order, axis=1, ticker="AAPL")
second_order_batch = opts.apply(gp.second_order, axis=1, ticker="AAPL")

# ✅ Good: Reuse downloaded data
opts = gp.download_options("AAPL")  # Download once
call_greeks = opts.apply(gp.greeks, axis=1, ticker="AAPL", option_type="c")
put_greeks = opts.apply(gp.greeks, axis=1, ticker="AAPL", option_type="p")
```

### 5.2 Memory Management

```python
# For large datasets, process in chunks
def process_large_option_chain(ticker, chunk_size=1000):
    opts = gp.download_options(ticker, opt_type="all", max_days=90)
    
    results = []
    for i in range(0, len(opts), chunk_size):
        chunk = opts.iloc[i:i+chunk_size]
        chunk_greeks = chunk.apply(gp.greeks, axis=1, ticker=ticker)
        results.append(chunk.join(chunk_greeks))
    
    return pd.concat(results, ignore_index=True)
```

### 5.3 Error Handling Best Practices

```python
def robust_greek_calculation(ticker, retries=3, delay=1):
    """Robust Greek calculation with error handling"""
    import time
    
    for attempt in range(retries):
        try:
            opts = gp.download_options(ticker)
            greeks_data = opts.apply(gp.greeks, axis=1, ticker=ticker)
            return opts.join(greeks_data)
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {ticker}: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise e
```

---

## 6. Error Handling

### 6.1 Common Errors and Solutions

#### Market Data Issues
```python
try:
    opts = gp.download_options("INVALID_TICKER")
except Exception as e:
    print(f"Data download failed: {e}")
    # Handle gracefully - use cached data, different ticker, etc.
```

#### Invalid Option Parameters
```python
# Check for required columns before Greek calculation
required_cols = ['strike', 'Days to Expiry', 'impliedVolatility']
if all(col in opts.columns for col in required_cols):
    greeks_data = opts.apply(gp.greeks, axis=1, ticker="AAPL")
else:
    print(f"Missing required columns: {set(required_cols) - set(opts.columns)}")
```

#### Numerical Issues
```python
# Handle edge cases in Greek calculations
def safe_greek_calculation(row, ticker):
    try:
        # Check for valid inputs
        if row['Days to Expiry'] <= 0:
            return pd.Series({'Delta': np.nan, 'Gamma': np.nan})
        
        if row['impliedVolatility'] <= 0:
            return pd.Series({'Delta': np.nan, 'Gamma': np.nan})
        
        return gp.greeks(row, ticker=ticker)
    
    except Exception as e:
        print(f"Greek calculation failed for row: {e}")
        return pd.Series({'Delta': np.nan, 'Gamma': np.nan})

# Apply safe calculation
opts['Greeks'] = opts.apply(safe_greek_calculation, ticker="AAPL", axis=1)
```

---

## 7. Examples & Use Cases

### 7.1 Complete Examples

For comprehensive, runnable examples covering all major features, see:

- **[examples.py](examples.py)** - Seven complete examples showing real-world usage patterns

Run specific examples:
```bash
python examples.py 1    # Basic Greeks calculation
python examples.py 2    # Individual Greeks
python examples.py 3    # Puts and calls analysis
python examples.py 4    # Greek orders
python examples.py 5    # 3D visualization
python examples.py 6    # Pricing comparison
python examples.py 7    # Risk management
```

### 7.2 Interactive Help

```python
import greeks_package as gp

# Package overview
gp.help()

# Function-specific help
gp.help(gp.download_options)
gp.help(gp.greeks)
gp.help(gp.surf_scatter)
gp.help(gp.greek_plot)  # NEW in v1.1.0
gp.help(gp.iv_plot)     # NEW in v1.1.0

# Python's built-in help
help(gp.delta)
help(gp.second_order)
```

---
---

© 2025 JR Concepcion. Licensed under the MIT License.

For questions, suggestions, or contributions, please visit the project repository or contact the author directly. 