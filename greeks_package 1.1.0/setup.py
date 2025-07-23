#!/usr/bin/env python3
"""
Setup script for greeks_package

A comprehensive Python package for calculating Black-Scholes option Greeks
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    """Read README.md file for long description"""
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt"""
    here = os.path.abspath(os.path.dirname(__file__))
    req_path = os.path.join(here, 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name="greeks-package",
    version="1.1.0",
    author="JR Concepcion",
    author_email="jr1concepcion@gmail.com",  
    description="Black-Scholes option Greeks made easy - comprehensive Greek calculations for European options",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/JRCon1/greeks-package", 
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "jupyter>=1.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
            "sphinx-autoapi>=2.0",
        ],
    },
    include_package_data=True,
    keywords=[
        "options", "greeks", "black-scholes", "finance", "derivatives", 
        "quantitative", "trading", "risk-management", "delta", "gamma", 
        "vega", "theta", "volatility", "options-pricing"
    ],
    project_urls={
        "Source": "https://github.com/JRCon1/greeks-package",
        "Documentation": "https://github.com/JRCon1/greeks-package/blob/main/greeks_package%201.0.1/README.md",
        "Tutorial (v0.1.0)": "https://youtu.be/geyCTGodXQk?si=zT3s4Gf2bMmGQk4I",
    },
) 