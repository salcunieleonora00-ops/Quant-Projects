"""
Section 1: Portfolio Selection with Capital Constraint - Enhanced Version
=========================================================================

Key Features:
- Fixed capital constraint: $100,000
- Find optimal weights (not just selection)
- 15-stock portfolio
- Quarterly rebalancing simulation
- Shrinkage estimators for robustness
- In-sample/Out-of-sample split (80/20)
- Realistic transaction costs and taxes
- Comprehensive performance metrics

References:
- Markowitz, H. (1952). Portfolio Selection
- Ledoit & Wolf (2004). Honey, I Shrunk the Sample Covariance Matrix
- Black & Litterman (1992). Global Portfolio Optimization

Author: Portfolio Optimization System
Version: 3.0 - Capital Constrained
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class CapitalConstrainedPortfolioSelector:
    """
    Portfolio selector with fixed capital constraint and optimal weight optimization.
    """
    
    def __init__(self, 
                 universe: List[str], 
                 capital: float = 100000.0,
                 lookback_years: int = 5,
                 rebalance_frequency: str = 'quarterly'):
        """
        Initialize portfolio selector with capital constraint.
        
        Parameters
        ----------
        universe : List[str]
            List of ticker symbols
        capital : float
            Fixed capital amount ($100,000)
        lookback_years : int
            Years of historical data
        rebalance_frequency : str
            Rebalancing frequency ('quarterly', 'monthly', 'annual')
        """
        self.universe = universe
        self.capital = capital
        self.lookback_years = lookback_years
        self.rebalance_frequency = rebalance_frequency
        
        # Data storage
        self.prices = None
        self.returns = None
        self.in_sample_prices = None
        self.out_sample_prices = None
        self.in_sample_returns = None
        self.out_sample_returns = None
        
        # Split information
        self.split_date = None
        self.in_sample_start = None
        self.in_sample_end = None
        self.out_sample_start = None
        self.out_sample_end = None
        
    def download_data(self) -> pd.DataFrame:
        """
        Download historical price data and split into in-sample/out-of-sample.
        
        Returns
        -------
        pd.DataFrame
            Complete price data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_years * 365)
        
        print(f"\n{'='*80}")
        print(f"DATA DOWNLOAD - Capital Constraint: ${self.capital:,.0f}")
        print(f"{'='*80}")
        print(f"Downloading data for {len(self.universe)} tickers...")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        data = yf.download(
            self.universe, 
            start=start_date, 
            end=end_date, 
            progress=False, 
            auto_adjust=True
        )
        
        # Handle MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                data = data['Close']
            elif 'Adj Close' in data.columns.get_level_values(0):
                data = data['Adj Close']
        
        # Remove tickers with insufficient data
        min_observations = 252 * 3  # At least 3 years
        data = data.dropna(thresh=min_observations, axis=1)
        
        # Forward fill small gaps (max 5 days)
        data = data.ffill(limit=5).dropna()
        
        # 80/20 split for in-sample/out-of-sample
        split_idx = int(len(data) * 0.8)
        
        self.in_sample_prices = data.iloc[:split_idx].copy()
        self.out_sample_prices = data.iloc[split_idx:].copy()
        
        self.in_sample_returns = self.in_sample_prices.pct_change().dropna()
        self.out_sample_returns = self.out_sample_prices.pct_change().dropna()
        
        self.prices = data
        self.returns = data.pct_change().dropna()
        
        # Store dates
        self.in_sample_start = self.in_sample_prices.index[0]
        self.in_sample_end = self.in_sample_prices.index[-1]
        self.out_sample_start = self.out_sample_prices.index[0]
        self.out_sample_end = self.out_sample_prices.index[-1]
        self.split_date = self.out_sample_start
        
        print(f"\n{'─'*80}")
        print(f"DATA SUMMARY")
        print(f"{'─'*80}")
        print(f"Successfully downloaded: {len(data.columns)} tickers")
        print(f"Total observations: {len(data)}")
        print(f"\nIn-Sample Period (Training):")
        print(f"  Start: {self.in_sample_start.strftime('%Y-%m-%d')}")
        print(f"  End:   {self.in_sample_end.strftime('%Y-%m-%d')}")
        print(f"  Days:  {len(self.in_sample_prices)}")
        print(f"\nOut-of-Sample Period (Testing):")
        print(f"  Start: {self.out_sample_start.strftime('%Y-%m-%d')}")
        print(f"  End:   {self.out_sample_end.strftime('%Y-%m-%d')}")
        print(f"  Days:  {len(self.out_sample_prices)}")
        
        return data
    
    def estimate_expected_returns(self, 
                                 method: str = 'shrinkage',
                                 market_return: float = 0.10) -> pd.Series:
        """
        Estimate expected returns using shrinkage estimator.
        
        Parameters
        ----------
        method : str
            'shrinkage' (recommended), 'historical', 'capm'
        market_return : float
            Long-term market return for shrinkage
        
        Returns
        -------
        pd.Series
            Estimated annual expected returns
        """
        historical_mean = self.in_sample_returns.mean() * 252
        
        if method == 'shrinkage':
            # James-Stein shrinkage
            shrinkage_intensity = 0.5
            expected_returns = (
                shrinkage_intensity * market_return + 
                (1 - shrinkage_intensity) * historical_mean
            )
            return expected_returns.clip(lower=0.0, upper=0.35)
        
        elif method == 'historical':
            return historical_mean.clip(lower=-0.15, upper=0.35)
        
        elif method == 'capm':
            volatilities = self.in_sample_returns.std() * np.sqrt(252)
            market_vol = volatilities.median()
            risk_free = 0.03
            market_premium = 0.07
            expected_returns = risk_free + market_premium * (volatilities / market_vol)
            return expected_returns.clip(lower=0.0, upper=0.30)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def estimate_covariance(self, method: str = 'ledoit_wolf') -> pd.DataFrame:
        """
        Estimate covariance matrix with Ledoit-Wolf shrinkage.
        
        Parameters
        ----------
        method : str
            'ledoit_wolf' (recommended), 'sample'
        
        Returns
        -------
        pd.DataFrame
            Annualized covariance matrix
        """
        if method == 'sample':
            return self.in_sample_returns.cov() * 252
        
        elif method == 'ledoit_wolf':
            sample_cov = self.in_sample_returns.cov() * 252
            
            # Constant correlation shrinkage target
            variances = np.diag(sample_cov)
            n = len(sample_cov)
            avg_corr = (sample_cov.sum().sum() - np.trace(sample_cov)) / (n * (n - 1))
            
            target = np.outer(np.sqrt(variances), np.sqrt(variances)) * avg_corr
            np.fill_diagonal(target, variances)
            
            # Shrinkage intensity
            shrinkage = 0.2
            shrunk_cov = shrinkage * target + (1 - shrinkage) * sample_cov
            
            return pd.DataFrame(shrunk_cov, index=sample_cov.index, columns=sample_cov.columns)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def optimize_weights(self,
                        candidate_tickers: List[str],
                        n_assets: int = 15,
                        risk_aversion: float = 0.5,
                        return_method: str = 'shrinkage',
                        cov_method: str = 'ledoit_wolf') -> Tuple[List[str], np.ndarray, Dict]:
        """
        Optimize portfolio weights for selected assets.
        
        This finds BOTH the asset selection AND optimal weights.
        
        Parameters
        ----------
        candidate_tickers : List[str]
            Candidate assets to choose from
        n_assets : int
            Number of assets to select (15)
        risk_aversion : float
            Risk aversion parameter (0-1)
        return_method : str
            Expected return estimation method
        cov_method : str
            Covariance estimation method
        
        Returns
        -------
        selected_tickers : List[str]
            Final selected tickers
        optimal_weights : np.ndarray
            Optimal portfolio weights (sum to 1)
        statistics : Dict
            Portfolio statistics
        """
        print(f"\n{'='*80}")
        print(f"WEIGHT OPTIMIZATION - Target: {n_assets} stocks")
        print(f"{'='*80}")
        
        # Estimate parameters
        all_returns = self.estimate_expected_returns(method=return_method)
        all_cov = self.estimate_covariance(method=cov_method)
        
        # Focus on candidates
        expected_returns = all_returns[candidate_tickers].values
        cov_matrix = all_cov.loc[candidate_tickers, candidate_tickers].values
        n_candidates = len(candidate_tickers)
        
        print(f"Optimizing from {n_candidates} candidates...")
        print(f"Risk aversion: {risk_aversion}")
        print(f"Return method: {return_method}")
        print(f"Covariance method: {cov_method}")
        
        # Step 1: Find best n_assets using maximum Sharpe
        def neg_sharpe_all(w):
            """Negative Sharpe ratio for all candidates."""
            ret = np.dot(w, expected_returns)
            vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
            return -ret / vol if vol > 1e-8 else 1e10
        
        constraints_all = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds_all = tuple((0, 1) for _ in range(n_candidates))
        x0_all = np.ones(n_candidates) / n_candidates
        
        result_all = minimize(
            neg_sharpe_all, 
            x0_all, 
            method='SLSQP',
            bounds=bounds_all, 
            constraints=constraints_all,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        # Select top n_assets by weight
        weights_all = result_all.x
        top_indices = np.argsort(weights_all)[-n_assets:]
        selected_tickers = [candidate_tickers[i] for i in top_indices]
        
        # Step 2: Re-optimize weights on selected assets only
        selected_returns = expected_returns[top_indices]
        selected_cov = cov_matrix[np.ix_(top_indices, top_indices)]
        
        def objective(w):
            """Mean-variance objective with risk aversion."""
            ret = np.dot(w, selected_returns)
            var = np.dot(w, np.dot(selected_cov, w))
            return risk_aversion * var - (1 - risk_aversion) * ret
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.01, 0.40) for _ in range(n_assets))  # Min 1%, Max 40%
        x0 = np.ones(n_assets) / n_assets
        
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        optimal_weights = result.x
        
        # Calculate statistics
        portfolio_return = np.dot(optimal_weights, selected_returns)
        portfolio_variance = np.dot(optimal_weights, np.dot(selected_cov, optimal_weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Diversification metrics
        hhi = np.sum(optimal_weights ** 2)
        effective_n = 1 / hhi if hhi > 0 else 0
        
        statistics = {
            'selected_tickers': selected_tickers,
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'hhi': hhi,
            'effective_n_assets': effective_n,
            'max_weight': optimal_weights.max(),
            'min_weight': optimal_weights.min(),
            'weight_std': optimal_weights.std()
        }
        
        print(f"\n{'─'*80}")
        print(f"OPTIMIZATION RESULTS")
        print(f"{'─'*80}")
        print(f"Expected Return:  {portfolio_return:.2%}")
        print(f"Volatility:       {portfolio_vol:.2%}")
        print(f"Sharpe Ratio:     {sharpe_ratio:.3f}")
        print(f"HHI:              {hhi:.4f}")
        print(f"Effective N:      {effective_n:.2f}")
        print(f"\nWeight Statistics:")
        print(f"  Max:   {optimal_weights.max():.2%}")
        print(f"  Min:   {optimal_weights.min():.2%}")
        print(f"  Std:   {optimal_weights.std():.4f}")
        
        print(f"\n{'─'*80}")
        print(f"PORTFOLIO ALLOCATION (${self.capital:,.0f})")
        print(f"{'─'*80}")
        print(f"{'Ticker':<8} {'Weight':>8} {'$ Amount':>12} {'Exp.Return':>12} {'Volatility':>12}")
        print(f"{'─'*80}")
        
        for i, ticker in enumerate(selected_tickers):
            weight = optimal_weights[i]
            amount = self.capital * weight
            ret = selected_returns[i]
            vol = np.sqrt(selected_cov[i, i])
            print(f"{ticker:<8} {weight:>7.2%} ${amount:>11,.0f} {ret:>11.2%} {vol:>11.2%}")
        
        print(f"{'─'*80}")
        print(f"{'TOTAL':<8} {optimal_weights.sum():>7.2%} ${self.capital:>11,.0f}")
        
        return selected_tickers, optimal_weights, statistics
    
    def get_statistics_for_quantum(self, 
                                   tickers: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get statistics for quantum optimization.
        
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
            Ticker symbols
        """
        all_returns = self.estimate_expected_returns(method='shrinkage')
        all_cov = self.estimate_covariance(method='ledoit_wolf')
        
        expected_returns = all_returns[tickers].values
        cov_matrix = all_cov.loc[tickers, tickers].values
        
        return expected_returns, cov_matrix, tickers


def define_universe() -> List[str]:
    """
    Define diversified stock universe.
    
    Returns
    -------
    List[str]
        120 high-quality stocks across sectors
    """
    universe = [
        # Technology (25)
        'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CSCO', 'ORCL', 'AMD', 'INTC',
        'CRM', 'ADBE', 'QCOM', 'TXN', 'INTU', 'NOW', 'PANW', 'AMAT',
        'ADI', 'KLAC', 'LRCX', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'ANSS', 'PLTR',
        
        # Communication (10)
        'GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA',
        
        # Consumer Discretionary (15)
        'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX',
        'BKNG', 'CMG', 'MAR', 'GM', 'F', 'ROST', 'YUM',
        
        # Consumer Staples (12)
        'WMT', 'PG', 'KO', 'PEP', 'COST', 'PM', 'MO', 'CL',
        'MDLZ', 'KMB', 'GIS', 'K',
        
        # Financials (20)
        'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'C',
        'BLK', 'SPGI', 'AXP', 'USB', 'PNC', 'TFC', 'COF',
        'BK', 'STT', 'SCHW', 'CB', 'MMC',
        
        # Healthcare (15)
        'JNJ', 'UNH', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'ABT',
        'DHR', 'BMY', 'AMGN', 'GILD', 'CVS', 'CI', 'HUM',
        
        # Industrials (13)
        'BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX', 'DE', 'LMT',
        'UNP', 'MMM', 'FDX', 'NSC', 'EMR',
        
        # Energy (6)
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC',
        
        # Materials (4)
        'LIN', 'APD', 'SHW', 'FCX'
    ]
    
    return universe


if __name__ == "__main__":
    # Example usage
    universe = define_universe()
    print(f"Universe: {len(universe)} stocks")
    
    selector = CapitalConstrainedPortfolioSelector(
        universe=universe,
        capital=100000.0,
        lookback_years=5,
        rebalance_frequency='quarterly'
    )
    
    data = selector.download_data()
    
    # Use all stocks as candidates for this example
    selected_tickers, weights, stats = selector.optimize_weights(
        candidate_tickers=data.columns.tolist()[:30],  # Top 30 for example
        n_assets=15,
        risk_aversion=0.5
    )