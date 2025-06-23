# greeks-package

Utilities to download option-chain data from Yahoo Finance and compute first-, second-, and third-order Greeks (Delta, Gamma, Vanna, Volga, Ultima, etc.) using the Black-Scholes framework.

```python
import greeks_package as gp

# pull a filtered option chain
chain = gp.download_options("AAPL", opt_type="c")

# compute all greeks for each row
full = chain.join(chain.apply(gp.greeks, axis=1, ticker="AAPL"))
print(full.head())
```

Built with NumPy, Pandas, SciPy, yfinance, and py_vollib. 