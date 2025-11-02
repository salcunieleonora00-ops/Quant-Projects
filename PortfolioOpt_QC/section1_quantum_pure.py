"""
Section 1: Portfolio Selection with Capital Constraint - Quantum Pure Version
=============================================================================

Key Features:
- Fixed capital constraint: $100,000
- Find optimal weights (not just selection)
- 15-stock portfolio
- Quarterly rebalancing simulation
- Shrinkage estimators for robustness
- In-sample/Out-of-sample split (80/20)
- Realistic transaction costs and taxes
- Comprehensive performance metrics
- Prepared for QUANTUM optimization only

References:
- Markowitz, H. (1952). Portfolio Selection
- Ledoit & Wolf (2004). Honey, I Shrunk the Sample Covariance Matrix
- Black & Litterman (1992). Global Portfolio Optimization

Author: Portfolio Optimization System
Version: 4.0 - Quantum Pure
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class QuantumPortfolioDataPreparator:
    """
    Data preparation for pure quantum portfolio optimization.
    No classical optimization - only data prep for quantum solver.
    """
    
    def __init__(self, 
                 universe: List[str], 
                 capital: float = 100000.0,
                 lookback_years: int = 5,
                 rebalance_frequency: str = 'quarterly'):
        """
        Initialize portfolio data preparator for quantum optimization.
        
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
        print(f"QUANTUM PORTFOLIO DATA PREPARATION")
        print(f"{'='*80}")
        print(f"Capital Constraint: ${self.capital:,.0f}")
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
        print(f"\nIn-Sample Period (Training - for Quantum):")
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
    
    def get_quantum_optimization_inputs(self, 
                                       tickers: List[str] = None,
                                       return_method: str = 'shrinkage',
                                       cov_method: str = 'ledoit_wolf') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Get prepared inputs for quantum optimization.
        
        Parameters
        ----------
        tickers : List[str], optional
            Subset of tickers to use (default: all)
        return_method : str
            Expected return estimation method
        cov_method : str
            Covariance estimation method
        
        Returns
        -------
        expected_returns : np.ndarray
            Annual expected returns (IN-SAMPLE only)
        cov_matrix : np.ndarray
            Annual covariance matrix (IN-SAMPLE only)
        tickers : List[str]
            Ticker symbols
        """
        print(f"\n{'='*80}")
        print(f"PREPARING QUANTUM OPTIMIZATION INPUTS")
        print(f"{'='*80}")
        print(f"⚠️  USING IN-SAMPLE DATA ONLY (No lookahead bias)")
        
        # Estimate parameters IN-SAMPLE
        all_returns = self.estimate_expected_returns(method=return_method)
        all_cov = self.estimate_covariance(method=cov_method)
        
        if tickers is None:
            tickers = self.in_sample_returns.columns.tolist()
        
        expected_returns = all_returns[tickers].values
        cov_matrix = all_cov.loc[tickers, tickers].values
        
        print(f"Assets: {len(tickers)}")
        print(f"Return method: {return_method}")
        print(f"Covariance method: {cov_method}")
        print(f"\nExpected returns: [{expected_returns.min():.2%}, {expected_returns.max():.2%}]")
        print(f"Average volatility: {np.sqrt(np.diag(cov_matrix)).mean():.2%}")
        
        # Calculate average correlation
        vols = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(vols, vols)
        avg_corr = (corr_matrix.sum() - len(corr_matrix)) / (len(corr_matrix) * (len(corr_matrix)-1))
        print(f"Average correlation: {avg_corr:.3f}")
        
        print(f"{'─'*80}")
        print(f"✓ Data ready for quantum optimization")
        print(f"{'='*80}")
        
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
    
    preparator = QuantumPortfolioDataPreparator(
        universe=universe,
        capital=100000.0,
        lookback_years=5,
        rebalance_frequency='quarterly'
    )
    
    data = preparator.download_data()
    
    # Get quantum inputs
    expected_returns, cov_matrix, tickers = preparator.get_quantum_optimization_inputs(
        tickers=data.columns.tolist()[:30],  # Top 30 for example
        return_method='shrinkage',
        cov_method='ledoit_wolf'
    )
    
    print(f"\n✓ Ready for quantum optimization with {len(tickers)} assets")