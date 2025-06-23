# greeks_package – Detailed Usage Guide

> **Version:** 0.1.1  |  **Python ≥** 3.9

Created by: JR Concepcion

This guide walks through every public function shipped with **greeks_package**, how to import them, and practical patterns for integrating them into your workflow.

---

## 1  Installation

```bash
# From PyPI (when published)
pip install greeks-package

# From local source
cd GreeksFolderJune
pip install -e .
```

The package pulls in **NumPy**, **Pandas**, **SciPy**, **yfinance**, and **py_vollib** automatically.

---

## 2  Importing functions

### 2.1  Top-level helpers

```python
from greeks_package import (
    download_options,   # fetch & filter option chain
    first_order,        # Δ, Vega, Θ, Rho
    second_order,       # Γ, Vanna, Volga, Veta, Charm
    third_order,        # Color, Speed, Ultima, Zomma
    greeks,             # Convenience = first + second + third
    help,               # Interactive cheat-sheet
)
```
The above six names are re-exported in `__init__.py`, so you get them straight from the package.

### 2.2  Advanced / low-level utilities

Additional helpers live inside the *core* sub-module:

```python
from greeks_package.core import (
    # Black-Scholes building blocks
    compute_d1, compute_d2, compute_d1_d2,

    # Second-order Greeks
    vanna, volga, charm, veta,

    # Third-order Greeks
    color, speed, ultima, zomma,
)
```
These are **not** re-exported at the package root to avoid clutter, so you must import them explicitly as above.

### 2.3  Interactive help

Need a refresher on arguments or available helpers?

```python
import greeks_package as gp

gp.help()                       # prints cheat-sheet & quick-start

gp.help(gp.download_options)    # deep-dive on one helper
```

---

## 3  Function reference

Below each function you will find: signature, arguments, return type, and an example.  All Greeks assume **Black-Scholes** dynamics and a continuously-compounded risk-free rate (default `r = 0.05`).

### 3.1  `download_options()`
```python
download_options(
    ticker_symbol: str,
    opt_type: str = "c",           # 'c' calls | 'p' puts
    max_days: int = 60,             # days-to-expiry filter
    lower_moneyness: float = 0.95,  # strike / spot lower bound
    upper_moneyness: float = 1.05,  # strike / spot upper bound
    price: bool = False,            # attach underlying spot column
) -> pd.DataFrame
```
Returns a tidy **DataFrame** with one row per option contract plus:

* `expiry` – expiry date (datetime64)
* `Days to Expiry` – integer days until expiry
* Bid/Ask, open-interest, implied-volatility, etc.
* `Mid-Point Price` – (bid + ask)/2
* Optionally `Stock Price` if `price=True`

**Example**
```python
import greeks_package as gp
options = gp.download_options("AAPL", opt_type="p", max_days=30, price=True)
print(options.head())
```

> **Tip** – `yfinance` can throttle or fail on first call.  Wrap in `try/ except` with a short sleep for robustness.

---

### 3.2  First-order wrapper: `first_order()`
Calculates **Δ (Delta)**, **Vega**, **Θ (Theta)**, and **Rho**.

```python
first_order(row: pd.Series, ticker: str, r: float = 0.05,
            option_type: str = "c", epsilon: float = 1e-9) -> pd.Series
```
*Meant to be used with `DataFrame.apply`.*

```python
first = options.apply(gp.first_order, axis=1, ticker="AAPL")
```

Same pattern applies for the next two wrappers.

---

### 3.3  Second-order wrapper: `second_order()`
Adds **Γ (Gamma)**, **Vanna**, **Volga**, **Veta**, **Charm**.

```python
second_order(row: pd.Series, ticker: str, r: float = 0.05,
             option_type: str = "c", epsilon: float = 1e-9) -> pd.Series
```

---

### 3.4  Third-order wrapper: `third_order()`
Returns **Color**, **Speed**, **Ultima**, **Zomma**.

```python
third_order(row: pd.Series, ticker: str, r: float = 0.05,
            option_type: str = "c", epsilon: float = 1e-9) -> pd.Series
```

---

### 3.5  All-in-one: `greeks()`
Combines ⬆️ wrappers in one call.

```python
greeks(row: pd.Series, ticker: str, r: float = 0.05,
       option_type: str = "c", epsilon: float = 1e-9) -> pd.Series
```

---

### 3.6  Low-level building blocks

| Function | Description |
|----------|-------------|
| `compute_d1`, `compute_d2`, `compute_d1_d2` | ___d1 / d2___ of Black-Scholes; useful if you want to implement your own Greeks. |
| `vanna`, `volga`, `charm`, `veta` | Individual second-order Greeks. |
| `color`, `speed`, `ultima`, `zomma` | Individual third-order Greeks. |

Each of these functions follows the same signature `(row, ticker, r=0.05, option_type='c', epsilon=1e-9)` and returns a **float**.

```python
options['Vanna'] = options.apply(gp.core.vanna, axis=1, ticker="AAPL")
```

---

## 4  Merging Greeks back to the option chain

Because wrappers return `pd.Series` with the same index, **joining** is trivial:

```python
opts = gp.download_options("AAPL", max_days=30)

first  = opts.apply(gp.first_order,  axis=1, ticker="AAPL")
second = opts.apply(gp.second_order, axis=1, ticker="AAPL")
third  = opts.apply(gp.third_order, axis=1, ticker="AAPL")

full = opts.join([first, second, third])
```
Or in-place assignment for smaller sets:

```python
opts[['Color','Speed','Ultima','Zomma']] = third
```

---

## 5  Handling edge cases

* **Zero / negative time-to-expiry**: guarded by `epsilon` so you never divide by zero.
* **Missing or zero volatility**: floor at 1 % (`sigma >= 0.01`).
* **yfinance hiccups**: network issues raise exceptions; wrappers catch them and return `NaN` or an error **Series**.

---

## 6  Practical recipe – full workflow

```python
import greeks_package as gp
import pandas as pd

TICKER = "MSFT"

# 1. Pull option chain (calls within ±5 % moneyness, ≤60 d expiry)
chain = gp.download_options(TICKER, opt_type="c", lower_moneyness=0.95, upper_moneyness=1.05)

# 2. Compute every Greek available
all_greeks = chain.apply(gp.greeks, axis=1, ticker=TICKER)

# 3. Combine & inspect
full = pd.concat([chain, all_greeks], axis=1)
print(full.head())
```

---

## 7  Import style cheat-sheet

| Goal | Code |
|------|------|
| Quick use of wrappers | `import greeks_package as gp` |
| Pull **download_options** only | `from greeks_package import download_options` |
| Access *vanna* directly | `from greeks_package.core import vanna` |
| Keep namespace explicit | `import greeks_package.core as gpc` then `gpc.vanna(...)` |
| Print interactive help | `gp.help()` |

---

## 8  API stability

The top-level six helpers will remain stable through minor releases.  Lower-level helpers may evolve; pin a specific version in production environments:

```text
# requirements.txt
greeks-package==0.1.0
```

---

## 9  Citation

If you use this package in research, please cite the repository and the underlying **py_vollib** library.

---

© 2025 JR Concepcion. Licensed under the MIT License. 