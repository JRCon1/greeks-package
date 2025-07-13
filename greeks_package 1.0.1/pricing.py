"""greeks_package.pricing – Option pricing helpers

Convenience sub-module to import pricing functions directly::

    from greeks_package import pricing
    price_bs = pricing.bsm_price(row, ticker="AAPL")
    price_mc = pricing.monte_carlo_price(row, ticker="AAPL", n=50_000)
"""

from .core import bsm_price, monte_carlo_price

__all__ = [
    "bsm_price",
    "monte_carlo_price",
] 