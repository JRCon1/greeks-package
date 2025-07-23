"""\ngreeks_package – Black-Scholes option Greeks made easy\n=====================================================\n\nCompute first-, second-, and third-order Greeks (Δ, Γ, Vega, …) for European\noptions using pure-Python, NumPy/SciPy implementations – **no external Greeks\nlibrary required**.  A tiny wrapper around\n`yfinance` lets you pull an option chain and immediately enrich it with the\nGreeks you need.\n\nQuick-start::\n\n    import greeks_package as gp\n    chain = gp.download_options("AAPL", max_days=30)\n    full  = chain.join(chain.apply(gp.greeks, axis=1, ticker="AAPL"))\n    print(full.head())\n\nTop-level helpers\n-----------------\n• download_options – Fetch & filter option chain from Yahoo! Finance.\n• multi_download   – Download option chains for multiple tickers at once.\n• first_order      – Δ, Vega, Θ, Rho.\n• second_order     – Γ, Vanna, Volga, Veta, Charm.\n• third_order      – Color, Speed, Ultima, Zomma.\n• greeks           – Convenience wrapper = first + second + third.\n• strategy_builder – Build & analyse multi-leg options strategies.\n\nVisualization helpers\n--------------------\n• surf_scatter     – Interactive 3D scatter plots of Greeks.\n• surface_plot     – Smooth 3D surface plots.\n• greek_plot       – Greek values vs time with strike selection.\n• iv_plot          – Implied volatility term structure.\n• oi_plot          – Open interest distribution analysis.\n• vol_curve        – Volatility smile/skew curves.\n\nSee `USAGE.md` for a complete guide.\n"""
from .core import (
    download_options, multi_download, first_order, second_order, third_order, greeks, strategy_builder,
    # Individual Greek functions
    delta, theta, vega, rho, gamma, vanna, volga, veta, charm, 
    color, speed, ultima, zomma,
    # Plotting functions
    surf_scatter, surface_plot, greek_plot, iv_plot, oi_plot, vol_curve,
    # Utility functions
    comb,
    # Pricing functions
    bsm_price, monte_carlo_price
)

# Re-export submodules for convenience
from . import pricing as pricing
from . import plotting as plotting

# Created by: JR Concepcion

# Expose a friendly runtime help() so users can call gp.help() in addition to
# Python's built-in help(). ---- *if import greeks_package as gp is used*

def help(topic=None):
    """Interactive, package-specific help.

    • Calling ``greeks_package.help()`` prints a concise cheat-sheet and lists
      the major public helpers.
    • Pass any function, class, or module to delegate to Python's standard
      ``pydoc.help`` for detailed info, e.g.::

          gp.help(gp.download_options)
    """
    import pydoc
    if topic is None:
        # Print the package-level docstring plus one-liners for each helper.
        print(__doc__)
        print("Available helpers:")
        for name in __all__:
            obj = globals().get(name)
            if obj is None:
                continue
            one_line = (obj.__doc__ or "").strip().split("\n")[0]
            print(f"  • {name:<15} – {one_line}")
        print("\nFor detailed docs see USAGE.md or call greeks_package.help(<object>).")
    else:
        pydoc.help(topic)

__all__ = [
    "download_options",
    "multi_download",
    "first_order",
    "second_order", 
    "third_order",
    "greeks",
    "strategy_builder",
    "pricing",
    "plotting",
    "help",  # expose helper
    # Individual Greek functions
    "delta", "theta", "vega", "rho", "gamma", 
    "vanna", "volga", "veta", "charm",
    "color", "speed", "ultima", "zomma",
    # Plotting functions
    "surf_scatter", "surface_plot", "greek_plot", "iv_plot", "oi_plot", "vol_curve",
    # Utility functions
    "comb",
    # Pricing functions
    "bsm_price", "monte_carlo_price",
] 