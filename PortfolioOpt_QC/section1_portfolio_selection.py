"""
Section 1: Portfolio Selection Module
=====================================

This module implements portfolio construction strategies based on established
financial theory. Two approaches are provided:
1. Markowitz Mean-Variance Optimization (classical)
2. Momentum Strategy (factor-based)

The selected portfolio will then be optimized using quantum computing.

References:
- Markowitz, H. (1952). Portfolio Selection. The Journal of Finance.
- Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers.

Author: Portfolio Optimization System
Version: 1.0
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from scipy.optimize import minimize


class PortfolioSelector:
    """
    Portfolio selection using established financial methodologies.
    """
    
    def __init__(self, universe: List[str], lookback_years: int = 3):
        """
        Initialize portfolio selector.
        
        Parameters
        ----------
        universe : List[str]
            List of ticker symbols to consider
        lookback_years : int
            Years of historical data for analysis
        """
        self.universe = universe
        self.lookback_years = lookback_years
        self.returns = None
        self.prices = None
        
    def download_data(self) -> pd.DataFrame:
        """
        Download historical price data for the universe.
        
        Returns
        -------
        pd.DataFrame
            Adjusted close prices for all tickers
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_years * 365)
        
        print(f"Downloading data for {len(self.universe)} tickers...")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Download with auto_adjust=True to get proper format
        data = yf.download(self.universe, start=start_date, end=end_date, 
                          progress=False, auto_adjust=True)
        
        # Handle different return formats from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-index format (multiple tickers)
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            elif 'Adj Close' in data.columns.get_level_values(0):
                data = data['Adj Close']
        else:
            # Single ticker or already simplified
            if len(self.universe) == 1:
                data = data[['Close']].copy()
                data.columns = self.universe
        
        # Remove tickers with insufficient data
        min_observations = 252 * 2
        data = data.dropna(thresh=min_observations, axis=1)
        
        # Forward fill small gaps, then drop remaining NaN
        data = data.fillna(method='ffill', limit=5).dropna()
        
        self.prices = data
        self.returns = data.pct_change().dropna()
        
        print(f"Successfully downloaded data for {len(data.columns)} tickers")
        print(f"Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        return data
    
    def markowitz_portfolio(self, n_assets: int = 15) -> Tuple[List[str], pd.DataFrame]:
        """
        Select portfolio using Markowitz mean-variance optimization.
        
        The approach:
        1. Calculate expected returns (mean) and covariance matrix
        2. Solve for maximum Sharpe ratio portfolio
        3. Select top N assets with highest weights
        
        Parameters
        ----------
        n_assets : int
            Number of assets to select
        
        Returns
        -------
        selected_tickers : List[str]
            Selected ticker symbols
        statistics : pd.DataFrame
            Statistics for selected assets
        """
        if self.returns is None:
            raise ValueError("Must call download_data() first")
        
        print("\nApplying Markowitz Mean-Variance Optimization...")
        
        # Annualize statistics
        annual_returns = self.returns.mean() * 252
        annual_cov = self.returns.cov() * 252
        
        # Define optimization problem: maximize Sharpe ratio
        n = len(annual_returns)
        
        def neg_sharpe(weights):
            """Negative Sharpe ratio (for minimization)."""
            ret = np.dot(weights, annual_returns)
            vol = np.sqrt(np.dot(weights, np.dot(annual_cov, weights)))
            return -ret / vol
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds: long-only (0 to 1)
        bounds = tuple((0, 1) for _ in range(n))
        
        # Initial guess: equal weight
        x0 = np.ones(n) / n
        
        # Optimize
        result = minimize(neg_sharpe, x0, method='SLSQP', 
                         bounds=bounds, constraints=constraints,
                         options={'ftol': 1e-9, 'maxiter': 1000})
        
        if not result.success:
            print("Warning: Optimization did not converge perfectly")
        
        optimal_weights = result.x
        
        # Select top N assets by weight
        weight_series = pd.Series(optimal_weights, index=annual_returns.index)
        top_assets = weight_series.nlargest(n_assets)
        selected_tickers = top_assets.index.tolist()
        
        # Prepare statistics
        stats = pd.DataFrame({
            'Ticker': selected_tickers,
            'Markowitz_Weight': top_assets.values,
            'Annual_Return': [annual_returns[t] for t in selected_tickers],
            'Annual_Volatility': [np.sqrt(annual_cov.loc[t, t]) for t in selected_tickers]
        })
        
        stats['Sharpe_Ratio'] = stats['Annual_Return'] / stats['Annual_Volatility']
        
        print(f"Selected {n_assets} assets using Markowitz optimization")
        print(f"Portfolio Sharpe Ratio: {-result.fun:.3f}")
        
        return selected_tickers, stats
    
    def momentum_portfolio(self, n_assets: int = 15, 
                          lookback_months: int = 12,
                          skip_month: int = 1) -> Tuple[List[str], pd.DataFrame]:
        """
        Select portfolio using momentum strategy.
        
        The approach:
        1. Calculate past returns over lookback period (skipping most recent month)
        2. Select top N assets by momentum
        3. This is a well-documented factor strategy (Jegadeesh & Titman, 1993)
        
        Parameters
        ----------
        n_assets : int
            Number of assets to select
        lookback_months : int
            Number of months for momentum calculation
        skip_month : int
            Number of most recent months to skip (avoid short-term reversal)
        
        Returns
        -------
        selected_tickers : List[str]
            Selected ticker symbols
        statistics : pd.DataFrame
            Statistics for selected assets
        """
        if self.prices is None:
            raise ValueError("Must call download_data() first")
        
        print(f"\nApplying Momentum Strategy ({lookback_months}-month lookback)...")
        
        # Calculate momentum: return from T-lookback to T-skip
        lookback_days = lookback_months * 21
        skip_days = skip_month * 21
        
        prices_current = self.prices.iloc[-skip_days]
        prices_past = self.prices.iloc[-(lookback_days + skip_days)]
        
        momentum = (prices_current / prices_past - 1).sort_values(ascending=False)
        
        # Select top N by momentum
        selected_tickers = momentum.head(n_assets).index.tolist()
        
        # Calculate additional statistics
        annual_returns = self.returns.mean() * 252
        annual_vol = self.returns.std() * np.sqrt(252)
        
        stats = pd.DataFrame({
            'Ticker': selected_tickers,
            'Momentum_Score': [momentum[t] for t in selected_tickers],
            'Annual_Return': [annual_returns[t] for t in selected_tickers],
            'Annual_Volatility': [annual_vol[t] for t in selected_tickers]
        })
        
        stats['Sharpe_Ratio'] = stats['Annual_Return'] / stats['Annual_Volatility']
        
        print(f"Selected {n_assets} assets using momentum strategy")
        print(f"Average momentum score: {stats['Momentum_Score'].mean():.2%}")
        
        return selected_tickers, stats
    
    def get_statistics(self, tickers: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Calculate expected returns and covariance matrix for selected tickers.
        
        Parameters
        ----------
        tickers : List[str]
            Selected ticker symbols
        
        Returns
        -------
        expected_returns : np.ndarray
            Annual expected returns
        cov_matrix : np.ndarray
            Annual covariance matrix
        tickers : List[str]
            Ticker symbols (potentially filtered)
        """
        returns_subset = self.returns[tickers]
        
        expected_returns = returns_subset.mean().values * 252
        cov_matrix = returns_subset.cov().values * 252
        
        return expected_returns, cov_matrix, tickers


def define_sp500_universe() -> List[str]:
    """
    Define a universe of high-quality S&P 500 stocks.
    
    Selection criteria:
    - Large market capitalization
    - High liquidity
    - Sector diversification
    - Stable historical data
    
    Returns
    -------
    List[str]
        Ticker symbols
    """
    universe = [
        # Technology
        'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'AMD', 'INTC',
        # Communication Services
        'GOOGL', 'META', 'NFLX', 'DIS',
        # Consumer Discretionary
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE',
        # Consumer Staples
        'WMT', 'PG', 'KO', 'PEP', 'COST',
        # Financials
        'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS',
        # Healthcare
        'JNJ', 'UNH', 'LLY', 'ABBV', 'MRK', 'PFE',
        # Industrials
        'BA', 'CAT', 'GE', 'HON', 'UPS',
        # Energy
        'XOM', 'CVX', 'COP',
        # Materials
        'LIN', 'APD',
        # Real Estate
        'PLD', 'AMT',
        # Utilities
        'NEE', 'DUK'
    ]
    
    return universe


if __name__ == "__main__":
    # Define universe
    universe = define_sp500_universe()
    print(f"Universe size: {len(universe)} stocks")
    
    # Initialize selector
    selector = PortfolioSelector(universe, lookback_years=3)
    
    # Download data
    selector.download_data()
    
    # Method 1: Markowitz
    print("\n" + "="*80)
    print("METHOD 1: MARKOWITZ MEAN-VARIANCE OPTIMIZATION")
    print("="*80)
    tickers_markowitz, stats_markowitz = selector.markowitz_portfolio(n_assets=15)
    print("\nSelected Portfolio:")
    print(stats_markowitz.to_string(index=False))
    
    # Method 2: Momentum
    print("\n" + "="*80)
    print("METHOD 2: MOMENTUM STRATEGY")
    print("="*80)
    tickers_momentum, stats_momentum = selector.momentum_portfolio(n_assets=15)
    print("\nSelected Portfolio:")
    print(stats_momentum.to_string(index=False))
    
    # Get final statistics for quantum optimization
    expected_returns, cov_matrix, tickers = selector.get_statistics(tickers_markowitz)
    
    print("\n" + "="*80)
    print("Portfolio statistics ready for quantum optimization")
    print(f"Number of assets: {len(tickers)}")
    print(f"Expected return range: {expected_returns.min():.2%} to {expected_returns.max():.2%}")
    print(f"Average volatility: {np.sqrt(np.diag(cov_matrix)).mean():.2%}")
    print("="*80)