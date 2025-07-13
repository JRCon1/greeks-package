# Changelog

All notable changes to the greeks_package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-01-16

### Added
- **Comb Function**: Added `comb()` function for combining multiple DataFrames (overlooked in previous version)

### Fixed
- **Strategy Builder**: Debugged strategy_builder greeks calculation

## [1.0.0] - 2025-07-10

### Added
- **Strategy Builder**: Multi-leg options strategy analysis and visualization
- **Enhanced Plotting**: Interactive 3D scatter plots and surface plots using Plotly
- **Individual Greek Access**: Direct access to all Greek functions (delta, gamma, vanna, etc.)
- **Performance Optimizations**: Refined processing speed for all calculation functions
- **Comprehensive Documentation**: 
  - Professional README with examples and badges
  - Complete USAGE.md guide with advanced patterns
  - Production-ready examples.py with 7 real-world scenarios
  - Comprehensive docstrings for all functions
- **Production Features**:
  - Full error handling and edge case management
  - Type hints throughout codebase
  - Memory-efficient processing for large datasets
  - Robust market data handling
- **Modern Packaging**: Both setup.py and pyproject.toml support
- **Complete Examples Suite**: Seven comprehensive usage examples

### Changed
- **Improved Documentation**: Complete rewrite of all documentation for production use
- **Enhanced Error Handling**: More robust handling of market data issues
- **Optimized Performance**: Faster execution for large option chains

### Technical Specifications
- **Python Support**: 3.9, 3.10, 3.11, 3.12
- **Dependencies**: NumPy, Pandas, SciPy, yfinance, Plotly
- **Package Structure**: Modular design with core, pricing, and plotting submodules
- **Packaging**: Modern setup with both setup.py and pyproject.toml

## [0.2.1] - 2025-06-27

### Added
- **Independent Greek Implementations**: Custom delta, gamma, vega, theta, rho calculations
- **Pricing Modules**: 
  - Black-Scholes theoretical pricing implementation
  - Monte Carlo simulation pricing
- **Enhanced Greek Suite**: All individual Greek functions now available

### Removed
- **pyvollib Dependency**: Eliminated external Greeks library dependency
- **Third-party Calculations**: Replaced with pure NumPy/SciPy implementations

### Changed
- **Core Architecture**: Rebuilt calculation engine with custom implementations
- **Performance**: Improved speed with optimized NumPy operations
- **Dependencies**: Reduced external dependencies for better reliability

## [0.1.0] - 2025-02-09

### Added
- **Core Greek Functions**: 
  - `greeks()`: All-in-one Greek calculation wrapper
  - `first_order()`: Delta, Vega, Theta, Rho calculations
  - `second_order()`: Gamma, Vanna, Volga, Veta, Charm calculations  
  - `third_order()`: Color, Speed, Ultima, Zomma calculations
- **Data Integration**: `download_options()` function for Yahoo Finance integration
- **Second-Order Greeks**: Vanna, Volga, Veta, Charm implementations
- **Third-Order Greeks**: Color, Speed, Ultima, Zomma implementations
- **Basic Documentation**: Initial README and function documentation

### Technical Foundation
- **Dependencies**: NumPy, Pandas, SciPy, yfinance, pyvollib
- **Python Support**: 3.8+
- **Architecture**: Wrapper-based approach with pyvollib backend

---

## Development Timeline

### Key Milestones
- **February 2025**: Initial release with core Greek calculations
- **June 2025**: Independence from external libraries with custom implementations
- **July 2025**: Production-ready release with strategy builder and advanced features

### Design Evolution
- **v0.1.0**: Foundation with external library dependency
- **v0.2.1**: Independence with custom implementations and pricing
- **v1.0.0**: Production-ready with strategy analysis and comprehensive tooling

---

© 2025 JR Concepcion. Licensed under the MIT License. 