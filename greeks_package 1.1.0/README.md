# greeks-package

**Black-Scholes option Greeks made easy**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python package for calculating **first-, second-, and third-order Greeks** for European options using pure NumPy/SciPy implementations. No external Greeks library required – just clean, fast calculations with integrated option chain downloading from Yahoo Finance.

## Features

- **Complete Greeks Suite**: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Charm, Veta, Color, Speed, Ultima, Zomma
- **Multi-Ticker Download**: Download options for multiple stocks simultaneously with `multi_download()`
- **Enhanced Data Integration**: Download calls, puts, or both together from Yahoo Finance
- **Flexible Usage**: Calculate individual Greeks or all at once with convenient wrapper functions
- **Interactive Visualization**: 3D plotting of Greeks surfaces using Plotly
- **Strategy Analysis**: Multi-leg options strategy builder and analyzer
- **Production Ready**: Comprehensive error handling, type hints, and full documentation

## Quick Start

```python
import greeks_package as gp

# Download Apple call options within 30 days, ±5% moneyness
opts = gp.download_options("AAPL", opt_type="c", max_days=30)

# Calculate all Greeks in one line
all_greeks = opts.apply(gp.greeks, axis=1, ticker="AAPL")

# Combine with original data
full_data = opts.join(all_greeks)
print(full_data[['strike', 'lastPrice', 'Delta', 'Gamma', 'Vega', 'Theta']].head())
```

## Installation

```bash
# From PyPI (when published)
pip install greeks-package

# From source (development)
git clone https://github.com/JRCon1/greeks-package.git
cd greeks-package
pip install -e .
```

**Requirements**: Python ≥ 3.9, NumPy, Pandas, SciPy, yfinance, Plotly

## Usage Examples

### 🆕 Multi-Ticker Download
```python
import greeks_package as gp

# Download options for multiple tickers at once
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
multi_opts = gp.multi_download(
    ticker_symbols=tickers,
    opt_type="c",
    max_days=30,
    price=True  # Include stock prices
)

print(f"Downloaded {len(multi_opts)} options across {len(tickers)} tickers")
```

### 🆕 Calls and Puts Together
```python
# Download both calls and puts simultaneously
opts = gp.download_options("TSLA", opt_type="all", max_days=60)

# Separate calls and puts
calls = opts[~opts['contractSymbol'].str.contains('P')]
puts = opts[opts['contractSymbol'].str.contains('P')]

print(f"Downloaded {len(calls)} calls and {len(puts)} puts")
```

### Individual Greeks Calculation
```python
opts = gp.download_options("MSFT", max_days=45)

# Calculate specific Greeks
opts['Delta'] = opts.apply(gp.delta, axis=1, ticker="MSFT")
opts['Gamma'] = opts.apply(gp.gamma, axis=1, ticker="MSFT")
opts['Vanna'] = opts.apply(gp.vanna, axis=1, ticker="MSFT")
```

### Interactive Visualization
```python
# 3D plots
gp.surf_scatter(opts, "AAPL", z="delta")      # Delta scatter plot
gp.surface_plot(opts, "AAPL", z="gamma")      # Gamma surface plot

# NEW in v1.1.0: Advanced plotting functions
gp.greek_plot(opts, greek_col="Delta")        # Greek vs time with strike selection
gp.iv_plot(opts, "AAPL")                      # Implied volatility term structure
gp.oi_plot(opts, "AAPL")                      # Open interest analysis
gp.vol_curve(opts, "AAPL")                    # Volatility smile/skew curves
```

### Greek Orders Analysis
```python
# Calculate Greeks by order
first_order = opts.apply(gp.first_order, axis=1, ticker="NVDA")    # Δ, Vega, Θ, Rho
second_order = opts.apply(gp.second_order, axis=1, ticker="NVDA")  # Γ, Vanna, Volga, Veta, Charm
third_order = opts.apply(gp.third_order, axis=1, ticker="NVDA")    # Color, Speed, Ultima, Zomma

# Combine all data
full_analysis = gp.comb(opts, first_order, second_order, third_order)
```

## API Reference

### Core Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `download_options()` | Fetch & filter option chain from Yahoo Finance | DataFrame |
| `multi_download()` | **NEW!** Download options for multiple tickers | DataFrame |
| `greeks()` | Calculate all 13 Greeks at once | Series |
| `first_order()` | Calculate Δ, Vega, Θ, Rho | Series |
| `second_order()` | Calculate Γ, Vanna, Volga, Veta, Charm | Series |
| `third_order()` | Calculate Color, Speed, Ultima, Zomma | Series |

### Individual Greeks

**First Order**: `delta`, `vega`, `theta`, `rho`  
**Second Order**: `gamma`, `vanna`, `volga`, `veta`, `charm`  
**Third Order**: `color`, `speed`, `ultima`, `zomma`

### Utilities

| Function | Description |
|----------|-------------|
| `comb()` | Combine multiple DataFrames with automatic column handling |
| `surf_scatter()` | Interactive 3D scatter plots |
| `surface_plot()` | Smooth 3D surface plots |
| `greek_plot()` | **NEW!** Greek values vs time with strike selection |
| `iv_plot()` | **NEW!** Implied volatility term structure |
| `oi_plot()` | **NEW!** Open interest distribution analysis |
| `vol_curve()` | **NEW!** Volatility smile/skew curves |
| `bsm_price()` | Black-Scholes theoretical pricing |
| `strategy_builder()` | Multi-leg options strategy analysis |

### Function Signatures

All Greek functions follow the same pattern:
```python
function_name(row: pd.Series, ticker: str, option_type: str = 'c', 
              r: float = 0.05, eps: float = 1e-9) -> float
```

**Multi-download signature:**
```python
multi_download(ticker_symbols: List[str], opt_type: str = 'c', 
               max_days: int = 60, lower_moneyness: float = 0.95,
               upper_moneyness: float = 1.05, price: bool = False) -> pd.DataFrame
```

## 📊 Comprehensive Examples

See [`examples.py`](examples.py) for complete usage demonstrations including:

1. **Basic Options Greeks Calculation**
2. **🆕 Calls and Puts Together** - Using `opt_type="all"`
3. **🆕 Multi-Ticker Download** - Using `multi_download()`  
4. **🆕 Multi-Download with Calls & Puts**
5. **Individual Greeks Selection**
6. **3D Visualization**
7. **Strategy Analysis**

Run examples:
```bash
python examples.py           # Run all examples
python examples.py 3         # Run multi-download example
python examples.py 2         # Run calls/puts example
```

## 📚 Documentation

- **[USAGE.md](USAGE.md)**: Detailed function reference and advanced usage patterns
- **[examples.py](examples.py)**: Complete working examples for all major features
- **Interactive Help**: Use `gp.help()` for in-package documentation

## 🧮 Greek Formulas

This package implements standard Black-Scholes Greeks:

- **Delta (Δ)**: `∂V/∂S` - Price sensitivity to underlying
- **Gamma (Γ)**: `∂²V/∂S²` - Delta sensitivity to underlying  
- **Vega (ν)**: `∂V/∂σ` - Price sensitivity to volatility
- **Theta (Θ)**: `∂V/∂t` - Time decay
- **Rho (ρ)**: `∂V/∂r` - Interest rate sensitivity

Plus advanced second and third-order Greeks for sophisticated risk management.

## Performance

- **Vectorized Operations**: Efficient NumPy/SciPy implementations
- **Minimal Dependencies**: No external Greeks libraries required
- **Memory Efficient**: Designed for large option chains
- **Fast Execution**: Optimized for production use

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**JR Concepcion**

Built using NumPy, Pandas, SciPy, yfinance, and Plotly.

---

### Quick Reference

```python
import greeks_package as gp

# Basic workflow
opts = gp.download_options("AAPL", opt_type="c", max_days=30)
greeks_data = opts.apply(gp.greeks, axis=1, ticker="AAPL")
full_data = opts.join(greeks_data)

# Individual Greeks
opts['Delta'] = opts.apply(gp.delta, axis=1, ticker="AAPL")
opts['Vanna'] = opts.apply(gp.vanna, axis=1, ticker="AAPL")

# Visualization
gp.surf_scatter(opts, "AAPL", z="delta")
gp.surface_plot(opts, "AAPL", z="impliedVolatility")
``` 