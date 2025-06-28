# greeks-package  
Black-Scholes option Greeks made easy
====================================

Compute **first-, second-, and third-order Greeks** for European options in a
single line of code.  A tiny helper fetches an option chain from Yahoo! Finance,
and pure-Python utilities give you Δ, Γ, Vega, Vanna, Color, and more – **no
external Greeks library required**.

---

## Installation

```bash
# From PyPI (when published)
pip install greeks-package

# From local source (editable mode)
cd GreeksFolderJune
pip install -e .
```

Requires **Python ≥ 3.9** and pulls in NumPy, Pandas, SciPy, yfinance, and
Plotly automatically.

---

## Quick-start

```python
import greeks_package as gp

# 1️⃣  Pull a filtered option chain (calls within ±5 % moneyness, ≤30 d expiry)
chain = gp.download_options("AAPL", opt_type="c", max_days=30,
                            lower_moneyness=0.95, upper_moneyness=1.05)

# 2️⃣  Compute all Greeks for each row in one shot
all_greeks = chain.apply(gp.greeks, axis=1, ticker="AAPL")

# 3️⃣  Combine & inspect
full = chain.join(all_greeks)
print(full.head())
```

---

## Public API – top-level helpers

| Helper | Description |
|--------|-------------|
| `download_options` | Fetch & filter option chain from Yahoo! Finance |
| `first_order`      | Δ, Vega, Θ, Rho |
| `second_order`     | Γ, Vanna, Volga, Veta, Charm |
| `third_order`      | Color, Speed, Ultima, Zomma |
| `greeks`           | Convenience = first + second + third |
| `help`             | Interactive cheat-sheet (`gp.help()`)

All six names above are re-exported in the package root:

```python
from greeks_package import download_options, first_order, second_order,
                          third_order, greeks, help
```

---

## Low-level building blocks

Need finer control?  Import directly from `greeks_package.core`:

```python
from greeks_package.core import (
    compute_d1, compute_d2, compute_d1_d2,   # Black-Scholes internals
    vanna, volga, charm, veta,               # second-order Greeks
    color, speed, ultima, zomma,             # third-order Greeks
)
```
Each function follows the same signature `(row, ticker, r=0.05, option_type='c', epsilon=1e-9)` and returns a float.

---

## Interactive help

```python
import greeks_package as gp

gp.help()                       # prints cheat-sheet & quick-start

gp.help(gp.second_order)        # deep-dive on a specific helper
```

For an in-depth tutorial covering edge cases, recipes, and API stability, see
[`USAGE.md`](USAGE.md).

---

© 2025 JR Concepcion. Licensed under the MIT License.

```python
import greeks_package as gp

# pull a filtered option chain
chain = gp.download_options("AAPL", opt_type="c")

# compute all greeks for each row
full = chain.join(chain.apply(gp.greeks, axis=1, ticker="AAPL"))
print(full.head())
```

Built with NumPy, Pandas, SciPy, yfinance, and Plotly. 