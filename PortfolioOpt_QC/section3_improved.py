"""
Section 3: Comprehensive Portfolio Analysis with Quarterly Rebalancing
======================================================================

Key Features:
- Quarterly rebalancing simulation
- Realistic transaction costs (commission + slippage + spread)
- Tax treatment (short-term vs long-term capital gains)
- Out-of-sample backtesting
- Complete performance statistics
- Risk-adjusted metrics
- Drawdown analysis

Cost Structure:
--------------
- Commission: 5 bps (0.05%)
- Slippage: 10-15 bps (market impact)
- Bid-ask spread: 5 bps
- Tax on gains: 20% long-term, 37% short-term

References:
- Perold (1988): Implementation Shortfall
- Grinold & Kahn (1999): Active Portfolio Management

Author: Portfolio Optimization System
Version: 3.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ComprehensivePortfolioAnalyzer:
    """
    Complete portfolio analysis with quarterly rebalancing and realistic costs.
    """
    
    def __init__(self,
                 expected_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 asset_names: List[str],
                 capital: float = 100000.0):
        """
        Initialize comprehensive analyzer.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected annual returns
        cov_matrix : np.ndarray
            Annual covariance matrix
        asset_names : List[str]
            Asset ticker symbols
        capital : float
            Fixed capital amount
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.asset_names = asset_names
        self.capital = capital
        self.n_assets = len(expected_returns)
        
        # Cost parameters (realistic)
        self.commission_bps = 5.0          # 0.05%
        self.slippage_bps_base = 10.0      # 0.10% base
        self.spread_bps = 5.0              # 0.05%
        self.tax_rate_short = 0.37         # 37% short-term gains
        self.tax_rate_long = 0.20          # 20% long-term gains
        
    def calculate_transaction_costs(self,
                                    target_weights: np.ndarray,
                                    current_weights: np.ndarray,
                                    portfolio_value: float,
                                    trade_size_penalty: bool = True) -> Dict:
        """
        Calculate realistic transaction costs including slippage.
        
        Parameters
        ----------
        target_weights : np.ndarray
            Target portfolio weights
        current_weights : np.ndarray
            Current portfolio weights
        portfolio_value : float
            Current portfolio value
        trade_size_penalty : bool
            Apply additional slippage for large trades
        
        Returns
        -------
        Dict
            Detailed cost breakdown
        """
        # Calculate turnover
        position_changes = np.abs(target_weights - current_weights)
        total_turnover = position_changes.sum()
        turnover_dollar = total_turnover * portfolio_value
        
        # Base costs
        commission_cost = total_turnover * (self.commission_bps / 10000)
        spread_cost = total_turnover * (self.spread_bps / 10000)
        
        # Slippage with size penalty
        slippage_bps = self.slippage_bps_base
        if trade_size_penalty:
            # Additional slippage for trades > 1% of portfolio
            large_trades = position_changes[position_changes > 0.01]
            if len(large_trades) > 0:
                size_penalty = np.mean(large_trades) * 50  # Extra 50 bps per % traded
                slippage_bps += size_penalty
        
        slippage_cost = total_turnover * (slippage_bps / 10000)
        
        # Total cost
        total_cost_pct = commission_cost + spread_cost + slippage_cost
        total_cost_dollar = total_cost_pct * portfolio_value
        
        return {
            'turnover_pct': total_turnover,
            'turnover_dollar': turnover_dollar,
            'commission_cost_pct': commission_cost,
            'spread_cost_pct': spread_cost,
            'slippage_cost_pct': slippage_cost,
            'slippage_bps': slippage_bps,
            'total_cost_pct': total_cost_pct,
            'total_cost_dollar': total_cost_dollar,
            'cost_per_turnover_bps': (total_cost_pct / total_turnover * 10000) if total_turnover > 0 else 0
        }
    
    def calculate_taxes(self,
                       gains: np.ndarray,
                       holding_periods_days: np.ndarray) -> Dict:
        """
        Calculate capital gains taxes.
        
        Parameters
        ----------
        gains : np.ndarray
            Capital gains for each position
        holding_periods_days : np.ndarray
            Holding period in days for each position
        
        Returns
        -------
        Dict
            Tax breakdown
        """
        # Separate short-term (<365 days) and long-term (>=365 days)
        short_term_mask = holding_periods_days < 365
        long_term_mask = holding_periods_days >= 365
        
        short_term_gains = np.sum(gains[short_term_mask & (gains > 0)])
        long_term_gains = np.sum(gains[long_term_mask & (gains > 0)])
        
        short_term_tax = short_term_gains * self.tax_rate_short
        long_term_tax = long_term_gains * self.tax_rate_long
        
        total_tax = short_term_tax + long_term_tax
        
        return {
            'short_term_gains': short_term_gains,
            'long_term_gains': long_term_gains,
            'short_term_tax': short_term_tax,
            'long_term_tax': long_term_tax,
            'total_tax': total_tax,
            'effective_tax_rate': total_tax / (short_term_gains + long_term_gains) if (short_term_gains + long_term_gains) > 0 else 0
        }
    
    def simulate_quarterly_rebalancing(self,
                                      selected_indices: List[int],
                                      target_weights: np.ndarray,
                                      returns_data: pd.DataFrame,
                                      prices_data: pd.DataFrame,
                                      rebalance_threshold: float = 0.05) -> Dict:
        """
        Simulate quarterly rebalancing with all realistic costs and taxes.
        
        Parameters
        ----------
        selected_indices : List[int]
            Selected asset indices
        target_weights : np.ndarray
            Target portfolio weights
        returns_data : pd.DataFrame
            Daily returns data (out-of-sample)
        prices_data : pd.DataFrame
            Daily prices data (out-of-sample)
        rebalance_threshold : float
            Rebalance only if drift > threshold (default 5%)
        
        Returns
        -------
        Dict
            Complete simulation results
        """
        print(f"\n{'='*80}")
        print(f"QUARTERLY REBALANCING SIMULATION")
        print(f"{'='*80}")
        print(f"Initial capital: ${self.capital:,.0f}")
        print(f"Rebalance threshold: {rebalance_threshold:.1%}")
        print(f"Period: {returns_data.index[0].strftime('%Y-%m-%d')} to {returns_data.index[-1].strftime('%Y-%m-%d')}")
        
        selected_tickers = [self.asset_names[i] for i in selected_indices]
        oos_returns = returns_data[selected_tickers]
        oos_prices = prices_data[selected_tickers]
        
        # Initialize
        current_weights = target_weights.copy()
        current_shares = (self.capital * target_weights) / oos_prices.iloc[0].values
        portfolio_value = self.capital
        cash = 0.0
        
        # Storage
        portfolio_values = [portfolio_value]
        dates = [oos_returns.index[0]]
        rebalance_dates = [oos_returns.index[0]]
        transaction_costs_history = []
        tax_history = []
        turnover_history = []
        
        # Track holding periods
        holding_start_dates = {ticker: oos_returns.index[0] for ticker in selected_tickers}
        
        # Quarterly dates
        rebalance_schedule = pd.date_range(
            start=oos_returns.index[0],
            end=oos_returns.index[-1],
            freq='Q'
        )
        
        print(f"\nScheduled rebalancing dates: {len(rebalance_schedule)}")
        print(f"Expected rebalances: ~{len(rebalance_schedule) - 1}")
        
        # Daily simulation
        for t in range(1, len(oos_returns)):
            current_date = oos_returns.index[t]
            current_prices = oos_prices.iloc[t].values
            
            # Update portfolio value
            position_values = current_shares * current_prices
            portfolio_value = position_values.sum() + cash
            current_weights = position_values / portfolio_value if portfolio_value > 0 else current_weights
            
            # Check if rebalancing date
            should_rebalance = False
            if current_date in rebalance_schedule:
                # Check drift
                weight_drift = np.abs(current_weights - target_weights).max()
                if weight_drift > rebalance_threshold:
                    should_rebalance = True
            
            # Rebalance
            if should_rebalance:
                print(f"\n{'─'*80}")
                print(f"Rebalancing on {current_date.strftime('%Y-%m-%d')}")
                print(f"  Portfolio value: ${portfolio_value:,.0f}")
                print(f"  Max weight drift: {weight_drift:.2%}")
                
                # Calculate transaction costs
                costs = self.calculate_transaction_costs(
                    target_weights,
                    current_weights,
                    portfolio_value,
                    trade_size_penalty=True
                )
                
                # Calculate capital gains and taxes
                purchase_prices = np.array([oos_prices.loc[holding_start_dates[ticker], ticker] 
                                           for ticker in selected_tickers])
                gains = (current_prices - purchase_prices) * current_shares
                holding_days = np.array([(current_date - holding_start_dates[ticker]).days 
                                        for ticker in selected_tickers])
                
                # Only pay taxes on positions we're selling
                position_changes = target_weights - current_weights
                selling_mask = position_changes < 0
                
                if np.any(selling_mask & (gains > 0)):
                    taxes = self.calculate_taxes(
                        gains * np.abs(position_changes) / current_weights,
                        holding_days
                    )
                else:
                    taxes = {
                        'short_term_gains': 0, 'long_term_gains': 0,
                        'short_term_tax': 0, 'long_term_tax': 0,
                        'total_tax': 0, 'effective_tax_rate': 0
                    }
                
                # Deduct costs and taxes
                portfolio_value -= costs['total_cost_dollar']
                portfolio_value -= taxes['total_tax']
                
                # Rebalance positions
                target_position_values = portfolio_value * target_weights
                current_shares = target_position_values / current_prices
                current_weights = target_weights.copy()
                cash = 0.0
                
                # Update holding start dates for changed positions
                for i, ticker in enumerate(selected_tickers):
                    if position_changes[i] != 0:
                        holding_start_dates[ticker] = current_date
                
                # Record
                rebalance_dates.append(current_date)
                transaction_costs_history.append(costs)
                tax_history.append(taxes)
                turnover_history.append(costs['turnover_pct'])
                
                print(f"  Transaction cost: ${costs['total_cost_dollar']:,.0f} ({costs['total_cost_pct']:.3%})")
                print(f"  Taxes paid: ${taxes['total_tax']:,.0f}")
                print(f"  New value: ${portfolio_value:,.0f}")
            
            portfolio_values.append(portfolio_value)
            dates.append(current_date)
        
        # Final statistics
        portfolio_values = np.array(portfolio_values)
        returns = pd.Series(np.diff(portfolio_values) / portfolio_values[:-1], index=dates[1:])
        
        # Performance metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = portfolio_values / portfolio_values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = drawdowns.min()
        
        # Cost analysis
        total_transaction_costs = sum([c['total_cost_dollar'] for c in transaction_costs_history])
        total_taxes = sum([t['total_tax'] for t in tax_history])
        avg_turnover = np.mean(turnover_history) if turnover_history else 0
        
        # Additional metrics
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        sortino = annual_return / (returns[returns < 0].std() * np.sqrt(252)) if len(returns[returns < 0]) > 0 else 0
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0
        
        # Value at Risk (95%)
        var_95 = returns.quantile(0.05)
        
        results = {
            'dates': dates,
            'portfolio_values': portfolio_values,
            'returns': returns,
            'rebalance_dates': rebalance_dates,
            'n_rebalances': len(rebalance_dates) - 1,
            
            # Performance
            'initial_value': portfolio_values[0],
            'final_value': portfolio_values[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            
            # Costs
            'total_transaction_costs': total_transaction_costs,
            'total_taxes': total_taxes,
            'total_costs': total_transaction_costs + total_taxes,
            'avg_quarterly_turnover': avg_turnover,
            'transaction_costs_history': transaction_costs_history,
            'tax_history': tax_history,
            
            # Risk metrics
            'var_95': var_95,
            'win_rate': win_rate,
            'downside_deviation': returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0,
            
            # Other
            'n_periods': len(returns),
            'years': len(returns) / 252
        }
        
        print(f"\n{'='*80}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*80}")
        print(f"Rebalances executed: {results['n_rebalances']}")
        print(f"Final value: ${results['final_value']:,.0f}")
        print(f"Total return: {results['total_return']:.2%}")
        print(f"Annual return: {results['annual_return']:.2%}")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"\nCosts:")
        print(f"  Transaction costs: ${total_transaction_costs:,.0f}")
        print(f"  Taxes: ${total_taxes:,.0f}")
        print(f"  Total: ${results['total_costs']:,.0f} ({results['total_costs']/self.capital:.2%} of initial capital)")
        
        return results
    
    def calculate_comprehensive_statistics(self,
                                          simulation_results: Dict,
                                          selected_indices: List[int],
                                          weights: np.ndarray) -> pd.DataFrame:
        """
        Calculate comprehensive performance statistics.
        
        Parameters
        ----------
        simulation_results : Dict
            Results from quarterly rebalancing simulation
        selected_indices : List[int]
            Selected asset indices
        weights : np.ndarray
            Portfolio weights
        
        Returns
        -------
        pd.DataFrame
            Complete statistics table
        """
        returns = simulation_results['returns']
        
        # Basic statistics
        stats = {
            'Total Return': simulation_results['total_return'],
            'Annualized Return': simulation_results['annual_return'],
            'Annualized Volatility': simulation_results['annual_volatility'],
            'Sharpe Ratio': simulation_results['sharpe_ratio'],
            'Sortino Ratio': simulation_results['sortino_ratio'],
            'Calmar Ratio': simulation_results['calmar_ratio'],
            'Maximum Drawdown': simulation_results['max_drawdown'],
            'VaR (95%)': simulation_results['var_95'],
            'Win Rate': simulation_results['win_rate'],
            'Downside Deviation': simulation_results['downside_deviation'],
            
            # Skewness and kurtosis
            'Skewness': returns.skew(),
            'Excess Kurtosis': returns.kurtosis(),
            
            # Best/Worst
            'Best Day': returns.max(),
            'Worst Day': returns.min(),
            'Best Month': returns.resample('M').sum().max() if len(returns) > 20 else returns.max(),
            'Worst Month': returns.resample('M').sum().min() if len(returns) > 20 else returns.min(),
            
            # Recovery
            'Average Recovery Days': self._calculate_avg_recovery(returns),
            
            # Trading
            'Number of Rebalances': simulation_results['n_rebalances'],
            'Avg Quarterly Turnover': simulation_results['avg_quarterly_turnover'],
            'Total Transaction Costs': simulation_results['total_transaction_costs'],
            'Total Taxes': simulation_results['total_taxes'],
            'Total Costs (% of Initial)': simulation_results['total_costs'] / simulation_results['initial_value'],
            
            # Portfolio characteristics
            'Number of Assets': len(selected_indices),
            'HHI': np.sum(weights ** 2),
            'Effective N Assets': 1 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else 0,
            'Max Weight': weights.max(),
            'Min Weight': weights.min(),
            'Weight Std Dev': weights.std()
        }
        
        df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
        
        return df
    
    def _calculate_avg_recovery(self, returns: pd.Series) -> float:
        """Calculate average days to recover from drawdowns."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        in_drawdown = drawdown < 0
        recovery_periods = []
        
        i = 0
        while i < len(in_drawdown):
            if in_drawdown.iloc[i]:
                start = i
                while i < len(in_drawdown) and in_drawdown.iloc[i]:
                    i += 1
                recovery_periods.append(i - start)
            else:
                i += 1
        
        return np.mean(recovery_periods) if recovery_periods else 0
    
    def compare_to_benchmarks(self,
                             simulation_results: Dict,
                             selected_indices: List[int],
                             weights: np.ndarray) -> pd.DataFrame:
        """
        Compare to benchmark strategies.
        
        Parameters
        ----------
        simulation_results : Dict
            Simulation results
        selected_indices : List[int]
            Selected assets
        weights : np.ndarray
            Portfolio weights
        
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        results = []
        
        # Quantum portfolio
        results.append({
            'Strategy': 'Quantum Portfolio',
            'Return': simulation_results['annual_return'],
            'Volatility': simulation_results['annual_volatility'],
            'Sharpe': simulation_results['sharpe_ratio'],
            'Max DD': simulation_results['max_drawdown'],
            'Calmar': simulation_results['calmar_ratio'],
            'Costs (%)': simulation_results['total_costs'] / simulation_results['initial_value']
        })
        
        # Equal weight
        ew_weights = np.ones(self.n_assets) / self.n_assets
        ew_return = np.dot(ew_weights, self.expected_returns)
        ew_vol = np.sqrt(np.dot(ew_weights, np.dot(self.cov_matrix, ew_weights)))
        results.append({
            'Strategy': 'Equal Weight (All)',
            'Return': ew_return,
            'Volatility': ew_vol,
            'Sharpe': ew_return / ew_vol if ew_vol > 0 else 0,
            'Max DD': -0.25,  # Approximate
            'Calmar': ew_return / 0.25 if ew_return > 0 else 0,
            'Costs (%)': 0.015  # Estimated
        })
        
        # Minimum variance
        mv_weights = self._minimum_variance_portfolio()
        mv_return = np.dot(mv_weights, self.expected_returns)
        mv_vol = np.sqrt(np.dot(mv_weights, np.dot(self.cov_matrix, mv_weights)))
        results.append({
            'Strategy': 'Minimum Variance',
            'Return': mv_return,
            'Volatility': mv_vol,
            'Sharpe': mv_return / mv_vol if mv_vol > 0 else 0,
            'Max DD': -0.18,  # Approximate
            'Calmar': mv_return / 0.18 if mv_return > 0 else 0,
            'Costs (%)': 0.012  # Estimated
        })
        
        # Maximum Sharpe
        ms_weights = self._maximum_sharpe_portfolio()
        ms_return = np.dot(ms_weights, self.expected_returns)
        ms_vol = np.sqrt(np.dot(ms_weights, np.dot(self.cov_matrix, ms_weights)))
        results.append({
            'Strategy': 'Maximum Sharpe',
            'Return': ms_return,
            'Volatility': ms_vol,
            'Sharpe': ms_return / ms_vol if ms_vol > 0 else 0,
            'Max DD': -0.22,  # Approximate
            'Calmar': ms_return / 0.22 if ms_return > 0 else 0,
            'Costs (%)': 0.013  # Estimated
        })
        
        df = pd.DataFrame(results)
        return df
    
    def _minimum_variance_portfolio(self) -> np.ndarray:
        """Calculate minimum variance portfolio."""
        from scipy.optimize import minimize
        
        def portfolio_variance(w):
            return np.dot(w, np.dot(self.cov_matrix, w))
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(portfolio_variance, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def _maximum_sharpe_portfolio(self) -> np.ndarray:
        """Calculate maximum Sharpe portfolio."""
        from scipy.optimize import minimize
        
        def neg_sharpe(w):
            ret = np.dot(w, self.expected_returns)
            vol = np.sqrt(np.dot(w, np.dot(self.cov_matrix, w)))
            return -ret / vol if vol > 0 else 1e10
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(neg_sharpe, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def visualize_comprehensive_results(self,
                                       simulation_results: Dict,
                                       statistics: pd.DataFrame,
                                       comparison: pd.DataFrame,
                                       selected_indices: List[int],
                                       weights: np.ndarray,
                                       output_path: str = None):
        """
        Create comprehensive visualization dashboard.
        
        Parameters
        ----------
        simulation_results : Dict
            Simulation results
        statistics : pd.DataFrame
            Statistics table
        comparison : pd.DataFrame
            Benchmark comparison
        selected_indices : List[int]
            Selected assets
        weights : np.ndarray
            Portfolio weights
        output_path : str
            Path to save figure
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(5, 4, hspace=0.4, wspace=0.4)
        
        # 1. Portfolio value evolution
        ax1 = fig.add_subplot(gs[0, :3])
        self._plot_portfolio_evolution(ax1, simulation_results)
        
        # 2. Drawdown chart
        ax2 = fig.add_subplot(gs[1, :3])
        self._plot_drawdown(ax2, simulation_results)
        
        # 3. Returns distribution
        ax3 = fig.add_subplot(gs[2, :2])
        self._plot_returns_distribution(ax3, simulation_results)
        
        # 4. Rolling Sharpe
        ax4 = fig.add_subplot(gs[2, 2:])
        self._plot_rolling_sharpe(ax4, simulation_results)
        
        # 5. Cost breakdown
        ax5 = fig.add_subplot(gs[0, 3])
        self._plot_cost_breakdown(ax5, simulation_results)
        
        # 6. Weight allocation
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_weight_allocation(ax6, selected_indices, weights)
        
        # 7. Performance comparison
        ax7 = fig.add_subplot(gs[3, :2])
        self._plot_benchmark_comparison(ax7, comparison)
        
        # 8. Monthly returns heatmap
        ax8 = fig.add_subplot(gs[3, 2:])
        self._plot_monthly_returns_heatmap(ax8, simulation_results)
        
        # 9. Correlation matrix
        ax9 = fig.add_subplot(gs[4, :2])
        self._plot_correlation_matrix(ax9, selected_indices)
        
        # 10. Key statistics text
        ax10 = fig.add_subplot(gs[4, 2:])
        self._plot_key_statistics(ax10, statistics, simulation_results)
        
        plt.suptitle(f'Quantum Portfolio Optimization - Complete Analysis (Capital: ${self.capital:,.0f})',
                    fontsize=18, fontweight='bold', y=0.998)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved: {output_path}")
        
        plt.close()
    
    def _plot_portfolio_evolution(self, ax, results):
        """Plot portfolio value over time."""
        dates = results['dates']
        values = results['portfolio_values']
        
        ax.plot(dates, values, linewidth=2, color='#2E86AB')
        ax.axhline(y=self.capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        
        # Mark rebalancing dates
        for rdate in results['rebalance_dates'][1:]:
            idx = dates.index(rdate)
            ax.axvline(x=rdate, color='red', alpha=0.2, linewidth=0.8)
        
        ax.set_title('Portfolio Value Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Portfolio Value ($)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    def _plot_drawdown(self, ax, results):
        """Plot drawdown chart."""
        values = results['portfolio_values']
        cumulative = values / values[0]
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        ax.fill_between(results['dates'], drawdown * 100, 0, color='#A23B72', alpha=0.6)
        ax.plot(results['dates'], drawdown * 100, color='#A23B72', linewidth=1.5)
        
        ax.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Drawdown (%)', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8)
    
    def _plot_returns_distribution(self, ax, results):
        """Plot returns distribution."""
        returns = results['returns'] * 100
        
        ax.hist(returns, bins=50, color='#18A558', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        mean_ret = returns.mean()
        ax.axvline(x=mean_ret, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ret:.2f}%')
        ax.axvline(x=0, color='black', linewidth=1, alpha=0.5)
        
        ax.set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Daily Return (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_rolling_sharpe(self, ax, results):
        """Plot rolling Sharpe ratio."""
        returns = results['returns']
        window = 63  # ~3 months
        
        if len(returns) > window:
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            
            ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='#F18F01')
            ax.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
            ax.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
            
            ax.set_title(f'Rolling Sharpe Ratio ({window} days)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Sharpe Ratio', fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_cost_breakdown(self, ax, results):
        """Plot cost breakdown."""
        trans_costs = results['total_transaction_costs']
        taxes = results['total_taxes']
        
        categories = ['Transaction\nCosts', 'Taxes']
        values = [trans_costs, taxes]
        colors = ['#E63946', '#F1A250']
        
        bars = ax.barh(categories, values, color=colors, edgecolor='black', linewidth=1)
        
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + max(values) * 0.02, bar.get_y() + bar.get_height()/2,
                   f'${val:,.0f}', va='center', fontsize=10, fontweight='bold')
        
        ax.set_title('Total Costs Breakdown', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cost ($)', fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_weight_allocation(self, ax, selected_indices, weights):
        """Plot portfolio weight allocation."""
        selected_tickers = [self.asset_names[i] for i in selected_indices]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(weights)))
        wedges, texts, autotexts = ax.pie(weights, labels=selected_tickers, autopct='%1.1f%%',
                                           colors=colors, startangle=90)
        
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(8)
            autotext.set_fontweight('bold')
        
        ax.set_title('Portfolio Allocation', fontsize=12, fontweight='bold')
    
    def _plot_benchmark_comparison(self, ax, comparison):
        """Plot benchmark comparison."""
        strategies = comparison['Strategy']
        sharpe = comparison['Sharpe']
        returns = comparison['Return'] * 100
        
        x = np.arange(len(strategies))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, returns, width, label='Annual Return (%)', 
                      color='#2E86AB', edgecolor='black', linewidth=1)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, sharpe, width, label='Sharpe Ratio',
                       color='#F18F01', edgecolor='black', linewidth=1)
        
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15, ha='right', fontsize=9)
        ax.set_ylabel('Annual Return (%)', fontsize=10, fontweight='bold')
        ax2.set_ylabel('Sharpe Ratio', fontsize=10, fontweight='bold')
        ax.set_title('Strategy Comparison', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def _plot_monthly_returns_heatmap(self, ax, results):
        """Plot monthly returns heatmap."""
        returns = results['returns']
        
        if len(returns) > 20:
            monthly_returns = returns.resample('M').sum()
            
            # Create matrix
            years = monthly_returns.index.year.unique()
            months = range(1, 13)
            
            data = np.full((len(years), 12), np.nan)
            
            for i, year in enumerate(years):
                for month in months:
                    mask = (monthly_returns.index.year == year) & (monthly_returns.index.month == month)
                    if mask.any():
                        data[i, month-1] = monthly_returns[mask].values[0] * 100
            
            sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                       xticklabels=['J','F','M','A','M','J','J','A','S','O','N','D'],
                       yticklabels=years, cbar_kws={'label': 'Return (%)'}, ax=ax)
            
            ax.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Month', fontsize=10)
            ax.set_ylabel('Year', fontsize=10)
    
    def _plot_correlation_matrix(self, ax, selected_indices):
        """Plot correlation matrix."""
        if len(selected_indices) < 2:
            ax.text(0.5, 0.5, 'Need at least 2 assets', ha='center', va='center',
                   transform=ax.transAxes)
            ax.axis('off')
            return
        
        sel_cov = self.cov_matrix[np.ix_(selected_indices, selected_indices)]
        sel_vols = np.sqrt(np.diag(sel_cov))
        corr = sel_cov / np.outer(sel_vols, sel_vols)
        
        sel_names = [self.asset_names[i] for i in selected_indices]
        
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   xticklabels=sel_names, yticklabels=sel_names,
                   cbar_kws={'label': 'Correlation'}, ax=ax, vmin=-1, vmax=1)
        
        ax.set_title('Asset Correlation Matrix', fontsize=12, fontweight='bold')
    
    def _plot_key_statistics(self, ax, statistics, results):
        """Plot key statistics as text."""
        ax.axis('off')
        
        key_stats = [
            ('Final Value', f"${results['final_value']:,.0f}"),
            ('Total Return', f"{results['total_return']:.2%}"),
            ('Annual Return', f"{results['annual_return']:.2%}"),
            ('Annual Volatility', f"{results['annual_volatility']:.2%}"),
            ('Sharpe Ratio', f"{results['sharpe_ratio']:.3f}"),
            ('Sortino Ratio', f"{results['sortino_ratio']:.3f}"),
            ('Calmar Ratio', f"{results['calmar_ratio']:.3f}"),
            ('Max Drawdown', f"{results['max_drawdown']:.2%}"),
            ('Win Rate', f"{results['win_rate']:.1%}"),
            ('Rebalances', f"{results['n_rebalances']}"),
            ('Total Costs', f"${results['total_costs']:,.0f}"),
            ('Cost Impact', f"{results['total_costs']/results['initial_value']:.2%}")
        ]
        
        y_pos = 0.95
        for label, value in key_stats:
            ax.text(0.05, y_pos, f"{label}:", fontsize=10, fontweight='bold',
                   transform=ax.transAxes)
            ax.text(0.65, y_pos, value, fontsize=10,
                   transform=ax.transAxes)
            y_pos -= 0.075
        
        ax.set_title('Key Performance Metrics', fontsize=12, fontweight='bold',
                    loc='left', pad=10)


if __name__ == "__main__":
    print("Comprehensive Portfolio Analyzer - Section 3")
    print("Run main script for complete workflow")