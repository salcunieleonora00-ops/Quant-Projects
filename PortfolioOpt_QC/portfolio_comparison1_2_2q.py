"""
Portfolio Strategy Comparison Analysis - FIXED VERSION
=======================================================

Comprehensive comparison of multiple portfolio optimization strategies
with advanced visualizations and statistical analysis.

Fixes:
- Resolved values_df scope issue in plot_drawdowns
- Fixed empty comparison table issue
- Added better error handling
- Improved data loading

Author: Portfolio Analysis System
Version: 1.1 FIXED
Date: 2025-10-30
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class StrategyComparator:
    """
    Comprehensive strategy comparison tool.
    
    Automatically loads results from multiple strategy folders and generates
    detailed comparative analysis with visualizations.
    """
    
    def __init__(self, base_dir='.'):
        """
        Initialize comparator.
        
        Parameters
        ----------
        base_dir : str
            Base directory containing results folders
        """
        self.base_dir = base_dir
        self.strategies = {}
        self.sp500_data = None
        
    def load_strategy_results(self, folder_name, strategy_label):
        """
        Load all results from a strategy folder.
        
        Parameters
        ----------
        folder_name : str
            Folder name (e.g., 'results_quantum_fixed')
        strategy_label : str
            Human-readable label (e.g., 'Quantum Fixed')
        
        Returns
        -------
        dict
            Strategy data
        """
        folder_path = os.path.join(self.base_dir, folder_name)
        
        if not os.path.exists(folder_path):
            print(f"  Folder not found: {folder_path}")
            return None
        
        # Find latest files (most recent timestamp)
        files = os.listdir(folder_path)
        
        # Get latest timestamp
        timestamps = []
        for f in files:
            if '_' in f:
                parts = f.split('_')
                if len(parts) >= 2:
                    try:
                        ts = parts[-1].replace('.csv', '').replace('.json', '').replace('.txt', '').replace('.png', '')
                        if ts.isdigit() and len(ts) >= 6:
                            timestamps.append(ts)
                    except:
                        pass
        
        if not timestamps:
            print(f"  No valid files found in {folder_path}")
            return None
        
        latest_ts = max(set(timestamps))
        print(f"✓ Loading {strategy_label}: timestamp {latest_ts}")
        
        data = {
            'label': strategy_label,
            'folder': folder_name,
            'timestamp': latest_ts
        }
        
        # Load portfolio values
        values_file = f"portfolio_values_{latest_ts}.csv"
        values_path = os.path.join(folder_path, values_file)
        if os.path.exists(values_path):
            data['values'] = pd.read_csv(values_path)
            data['values']['Date'] = pd.to_datetime(data['values']['Date'])
            print(f"  ✓ Loaded portfolio values: {len(data['values'])} rows")
        else:
            print(f"    Portfolio values not found: {values_file}")
        
        # Load statistics
        stats_file = f"statistics_{latest_ts}.csv"
        stats_path = os.path.join(folder_path, stats_file)
        if os.path.exists(stats_path):
            data['statistics'] = pd.read_csv(stats_path, index_col=0)
            print(f"  ✓ Loaded statistics")
        else:
            print(f"    Statistics not found: {stats_file}")
        
        # Load benchmarks
        bench_file = f"benchmarks_{latest_ts}.csv"
        bench_path = os.path.join(folder_path, bench_file)
        if os.path.exists(bench_path):
            data['benchmarks'] = pd.read_csv(bench_path)
            print(f"  ✓ Loaded benchmarks")
        else:
            print(f"    Benchmarks not found: {bench_file}")
        
        # Load JSON report
        json_file = f"complete_report_{latest_ts}.json"
        json_path = os.path.join(folder_path, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data['report'] = json.load(f)
            print(f"  ✓ Loaded JSON report")
        else:
            print(f"    JSON report not found: {json_file}")
        
        # Load allocation
        alloc_file = f"portfolio_allocation_{latest_ts}.csv"
        alloc_path = os.path.join(folder_path, alloc_file)
        if os.path.exists(alloc_path):
            data['allocation'] = pd.read_csv(alloc_path)
            print(f"  ✓ Loaded allocation: {len(data['allocation'])} assets")
        else:
            print(f"    Allocation not found: {alloc_file}")
        
        return data
    
    def load_all_strategies(self, strategy_configs):
        """
        Load all strategies from config.
        
        Parameters
        ----------
        strategy_configs : list of tuples
            [(folder_name, label), ...]
        """
        print("\n" + "="*80)
        print("LOADING STRATEGY RESULTS")
        print("="*80)
        
        for folder, label in strategy_configs:
            data = self.load_strategy_results(folder, label)
            if data:
                self.strategies[label] = data
        
        print(f"\n✓ Successfully loaded {len(self.strategies)} strategies")
        
        if len(self.strategies) == 0:
            print("\n ERROR: No strategies loaded!")
            return False
        
        return True
    
    def calculate_returns(self, values_series):
        """Calculate returns from values."""
        return values_series.pct_change().dropna()
    
    def calculate_drawdown(self, values_series):
        """Calculate drawdown series."""
        cummax = values_series.cummax()
        drawdown = (values_series - cummax) / cummax
        return drawdown
    
    def calculate_rolling_sharpe(self, returns, window=60):
        """Calculate rolling Sharpe ratio (60-day window)."""
        rolling_mean = returns.rolling(window).mean() * 252
        rolling_std = returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_mean / rolling_std
        return rolling_sharpe
    
    def create_comparison_table(self):
        """
        Create comprehensive comparison table.
        
        Returns
        -------
        pd.DataFrame
            Comparison metrics
        """
        metrics = []
        
        for label, data in self.strategies.items():
            metric_dict = {'Strategy': label}
            
            # Try to get data from report (preferred)
            if 'report' in data:
                report = data['report']
                oos = report.get('out_of_sample', {})
                config = report.get('configuration', {})
                
                metric_dict.update({
                    'Method': 'Quantum' if config.get('use_quantum', False) else 'Classical',
                    'Final Value ($)': oos.get('final_value', 0),
                    'Total Return (%)': oos.get('total_return', 0) * 100,
                    'Annual Return (%)': oos.get('annual_return', 0) * 100,
                    'Sharpe Ratio': oos.get('sharpe_ratio', 0),
                    'Sortino Ratio': oos.get('sortino_ratio', 0),
                    'Calmar Ratio': oos.get('calmar_ratio', 0),
                    'Max Drawdown (%)': abs(oos.get('max_drawdown', 0)) * 100,
                    'Volatility (%)': oos.get('annual_volatility', 0) * 100,
                    'Win Rate (%)': oos.get('win_rate', 0) * 100,
                    'N Assets': config.get('n_assets', 0),
                    'Total Costs ($)': oos.get('total_costs', 0),
                    'N Rebalances': oos.get('n_rebalances', 0)
                })
            
            # Fallback: calculate from values
            elif 'values' in data:
                values_df = data['values']
                initial_value = values_df['Value'].iloc[0]
                final_value = values_df['Value'].iloc[-1]
                total_return = (final_value / initial_value) - 1
                
                returns = self.calculate_returns(values_df['Value'])
                annual_return = returns.mean() * 252
                annual_vol = returns.std() * np.sqrt(252)
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                
                dd = self.calculate_drawdown(values_df['Value'])
                max_dd = dd.min()
                
                metric_dict.update({
                    'Method': 'Unknown',
                    'Final Value ($)': final_value,
                    'Total Return (%)': total_return * 100,
                    'Annual Return (%)': annual_return * 100,
                    'Sharpe Ratio': sharpe,
                    'Sortino Ratio': 0,
                    'Calmar Ratio': 0,
                    'Max Drawdown (%)': abs(max_dd) * 100,
                    'Volatility (%)': annual_vol * 100,
                    'Win Rate (%)': (returns > 0).mean() * 100,
                    'N Assets': 0,
                    'Total Costs ($)': 0,
                    'N Rebalances': 0
                })
            
            if len(metric_dict) > 1:  # Has more than just 'Strategy'
                metrics.append(metric_dict)
        
        if len(metrics) == 0:
            print("\n  WARNING: No metrics could be extracted from strategies!")
            return pd.DataFrame()
        
        df = pd.DataFrame(metrics)
        return df
    
    def plot_equity_curves(self, output_path=None):
        """
        Plot equity curves comparison.
        
        Parameters
        ----------
        output_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        plotted = False
        for label, data in self.strategies.items():
            if 'values' not in data:
                continue
            
            values_df = data['values']
            ax.plot(values_df['Date'], values_df['Value'], 
                   label=label, linewidth=2.5, alpha=0.85)
            plotted = True
        
        if not plotted:
            print("  No equity curves to plot")
            plt.close(fig)
            return None
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
        ax.set_title('Portfolio Equity Curves Comparison', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Equity curves saved: {output_path}")
        
        return fig
    
    def plot_drawdowns(self, output_path=None):
        """Plot drawdown comparison - FIXED VERSION."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        plotted = False
        last_dates = None
        worst_dd = None
        
        for label, data in self.strategies.items():
            if 'values' not in data:
                continue
            
            values_df = data['values']
            dd = self.calculate_drawdown(values_df['Value'])
            ax.plot(values_df['Date'], dd * 100, 
                   label=label, linewidth=2.5, alpha=0.85)
            
            # Track for filling
            last_dates = values_df['Date']
            if worst_dd is None or dd.min() < worst_dd.min():
                worst_dd = dd
            
            plotted = True
        
        if not plotted:
            print("  No drawdowns to plot")
            plt.close(fig)
            return None
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Fill area under worst drawdown
        if last_dates is not None and worst_dd is not None:
            ax.fill_between(last_dates, 0, worst_dd * 100, alpha=0.15, color='red')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
        ax.set_title('Drawdown Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Drawdowns saved: {output_path}")
        
        return fig
    
    def plot_rolling_sharpe(self, window=60, output_path=None):
        """Plot rolling Sharpe ratio."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        plotted = False
        for label, data in self.strategies.items():
            if 'values' not in data:
                continue
            
            values_df = data['values']
            returns = self.calculate_returns(values_df['Value'])
            rolling_sharpe = self.calculate_rolling_sharpe(returns, window)
            
            # Skip NaN values
            valid_idx = rolling_sharpe.notna()
            if valid_idx.sum() == 0:
                continue
            
            ax.plot(values_df['Date'].iloc[window:], rolling_sharpe.iloc[window:], 
                   label=label, linewidth=2.5, alpha=0.85)
            plotted = True
        
        if not plotted:
            print("  No rolling Sharpe to plot")
            plt.close(fig)
            return None
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axhline(y=1, color='green', linestyle=':', alpha=0.5, label='Sharpe = 1')
        ax.axhline(y=2, color='darkgreen', linestyle=':', alpha=0.5, label='Sharpe = 2')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rolling Sharpe Ratio', fontsize=12, fontweight='bold')
        ax.set_title(f'Rolling Sharpe Ratio ({window}-day window)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Rolling Sharpe saved: {output_path}")
        
        return fig
    
    def plot_returns_distribution(self, output_path=None):
        """Plot returns distribution comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        ax1 = axes[0]
        plotted = False
        for label, data in self.strategies.items():
            if 'values' not in data:
                continue
            
            values_df = data['values']
            returns = self.calculate_returns(values_df['Value']) * 100
            ax1.hist(returns, bins=50, alpha=0.5, label=label, density=True)
            plotted = True
        
        if plotted:
            ax1.set_xlabel('Daily Returns (%)', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
            ax1.set_title('Returns Distribution', fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Box plot
        ax2 = axes[1]
        returns_data = []
        labels = []
        for label, data in self.strategies.items():
            if 'values' not in data:
                continue
            
            values_df = data['values']
            returns = self.calculate_returns(values_df['Value']) * 100
            returns_data.append(returns.dropna())
            labels.append(label)
        
        if len(returns_data) > 0:
            bp = ax2.boxplot(returns_data, labels=labels, showmeans=True, 
                           meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
            ax2.set_ylabel('Daily Returns (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Returns Box Plot', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Returns distribution saved: {output_path}")
        
        return fig
    
    def plot_risk_return_scatter(self, output_path=None):
        """Plot risk-return scatter plot."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        plotted = False
        for label, data in self.strategies.items():
            if 'report' not in data:
                continue
            
            oos = data['report'].get('out_of_sample', {})
            annual_return = oos.get('annual_return', 0) * 100
            volatility = oos.get('annual_volatility', 0) * 100
            sharpe = oos.get('sharpe_ratio', 0)
            
            # Size by Sharpe
            size = max(200, abs(sharpe) * 300)
            color = 'green' if sharpe > 1 else 'orange' if sharpe > 0 else 'red'
            
            ax.scatter(volatility, annual_return, s=size, alpha=0.6, 
                      label=label, c=color, edgecolors='black', linewidth=1.5)
            
            # Annotate
            ax.annotate(f'{label}\nSharpe: {sharpe:.2f}', 
                       (volatility, annual_return), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))
            plotted = True
        
        if not plotted:
            print("  No risk-return data to plot")
            plt.close(fig)
            return None
        
        ax.set_xlabel('Annualized Volatility (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Annualized Return (%)', fontsize=12, fontweight='bold')
        ax.set_title('Risk-Return Profile\n(bubble size = Sharpe Ratio)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Risk-return scatter saved: {output_path}")
        
        return fig
    
    def plot_portfolio_composition(self, output_path=None):
        """Plot portfolio composition comparison."""
        n_strategies = len([d for d in self.strategies.values() if 'allocation' in d])
        
        if n_strategies == 0:
            print("  No allocation data to plot")
            return None
        
        fig, axes = plt.subplots(1, n_strategies, figsize=(7*n_strategies, 7))
        
        if n_strategies == 1:
            axes = [axes]
        
        idx = 0
        for label, data in self.strategies.items():
            if 'allocation' not in data:
                continue
            
            alloc = data['allocation']
            ax = axes[idx]
            
            # Pie chart
            wedges, texts, autotexts = ax.pie(alloc['Weight'], labels=alloc['Ticker'], 
                                               autopct='%1.1f%%', startangle=90,
                                               textprops={'fontsize': 9})
            
            # Bold percentage text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title(f'{label}\nPortfolio Composition ({len(alloc)} assets)', 
                        fontsize=12, fontweight='bold')
            idx += 1
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Portfolio composition saved: {output_path}")
        
        return fig
    
    def plot_monthly_returns_heatmap(self, output_path=None):
        """Plot monthly returns heatmap for each strategy."""
        n_strategies = len([d for d in self.strategies.values() if 'values' in d])
        
        if n_strategies == 0:
            print("  No values data for heatmap")
            return None
        
        fig, axes = plt.subplots(n_strategies, 1, figsize=(14, 5*n_strategies))
        
        if n_strategies == 1:
            axes = [axes]
        
        idx = 0
        for label, data in self.strategies.items():
            if 'values' not in data:
                continue
            
            values_df = data['values'].copy()
            values_df.set_index('Date', inplace=True)
            
            # Calculate monthly returns
            monthly_values = values_df.resample('M').last()
            monthly_returns = monthly_values.pct_change() * 100
            
            # Pivot for heatmap
            monthly_returns['Year'] = monthly_returns.index.year
            monthly_returns['Month'] = monthly_returns.index.month
            heatmap_data = monthly_returns.pivot(index='Year', columns='Month', values='Value')
            
            # Plot
            ax = axes[idx]
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                       center=0, ax=ax, cbar_kws={'label': 'Return (%)'}, 
                       linewidths=0.5)
            ax.set_title(f'{label} - Monthly Returns (%)', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Month', fontsize=10)
            ax.set_ylabel('Year', fontsize=10)
            idx += 1
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Monthly returns heatmap saved: {output_path}")
        
        return fig
    
    def plot_costs_comparison(self, output_path=None):
        """Plot costs comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        labels_list = []
        transaction_costs = []
        taxes = []
        total_costs = []
        
        for label, data in self.strategies.items():
            if 'report' not in data:
                continue
            
            oos = data['report'].get('out_of_sample', {})
            labels_list.append(label)
            transaction_costs.append(oos.get('total_transaction_costs', 0))
            taxes.append(oos.get('total_taxes', 0))
            total_costs.append(oos.get('total_costs', 0))
        
        if len(labels_list) == 0:
            print("  No cost data to plot")
            plt.close(fig)
            return None
        
        x = np.arange(len(labels_list))
        width = 0.25
        
        ax.bar(x - width, transaction_costs, width, label='Transaction Costs', alpha=0.8)
        ax.bar(x, taxes, width, label='Taxes', alpha=0.8)
        ax.bar(x + width, total_costs, width, label='Total Costs', alpha=0.8)
        
        ax.set_ylabel('Cost ($)', fontsize=12, fontweight='bold')
        ax.set_title('Trading Costs Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_list, rotation=45, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Costs comparison saved: {output_path}")
        
        return fig
    
    def plot_cumulative_returns(self, output_path=None):
        """Plot cumulative returns comparison (normalized to 100)."""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        plotted = False
        for label, data in self.strategies.items():
            if 'values' not in data:
                continue
            
            values_df = data['values']
            # Normalize to start at 100
            normalized = (values_df['Value'] / values_df['Value'].iloc[0]) * 100
            ax.plot(values_df['Date'], normalized, 
                   label=label, linewidth=2.5, alpha=0.85)
            plotted = True
        
        if not plotted:
            print("  No cumulative returns to plot")
            plt.close(fig)
            return None
        
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.3, label='Initial Value')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Return (Base 100)', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Returns Comparison (Normalized)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ Cumulative returns saved: {output_path}")
        
        return fig
    
    def perform_statistical_tests(self):
        """
        Perform statistical tests on returns.
        
        Returns
        -------
        pd.DataFrame
            Test results
        """
        results = []
        strategy_returns = {}
        
        # Calculate returns for all strategies
        for label, data in self.strategies.items():
            if 'values' not in data:
                continue
            
            values_df = data['values']
            returns = self.calculate_returns(values_df['Value'])
            strategy_returns[label] = returns
        
        if len(strategy_returns) < 2:
            print("  Need at least 2 strategies for statistical tests")
            return pd.DataFrame()
        
        # Pairwise t-tests
        strategy_labels = list(strategy_returns.keys())
        for i in range(len(strategy_labels)):
            for j in range(i+1, len(strategy_labels)):
                label1 = strategy_labels[i]
                label2 = strategy_labels[j]
                
                returns1 = strategy_returns[label1].dropna()
                returns2 = strategy_returns[label2].dropna()
                
                # Align dates
                common_dates = returns1.index.intersection(returns2.index)
                if len(common_dates) == 0:
                    continue
                
                returns1_aligned = returns1.loc[common_dates]
                returns2_aligned = returns2.loc[common_dates]
                
                # T-test
                t_stat, p_value = stats.ttest_ind(returns1_aligned, returns2_aligned)
                
                # Mean difference
                mean_diff = (returns1_aligned.mean() - returns2_aligned.mean()) * 252  # Annualized
                
                results.append({
                    'Strategy 1': label1,
                    'Strategy 2': label2,
                    'Mean Diff (annual %)': mean_diff * 100,
                    'T-Statistic': t_stat,
                    'P-Value': p_value,
                    'Significant (5%)': 'Yes' if p_value < 0.05 else 'No',
                    'Significant (1%)': 'Yes' if p_value < 0.01 else 'No'
                })
        
        if len(results) == 0:
            return pd.DataFrame()
        
        return pd.DataFrame(results)
    
    def generate_comprehensive_report(self, output_dir='./comparison_results'):
        """
        Generate comprehensive comparison report.
        
        Parameters
        ----------
        output_dir : str
            Output directory for all results
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE COMPARISON REPORT")
        print("="*80)
        
        # 1. Comparison table
        print("\n[1/11] Creating comparison table...")
        comparison_df = self.create_comparison_table()
        
        if len(comparison_df) > 0:
            table_path = os.path.join(output_dir, f'comparison_table_{timestamp}.csv')
            comparison_df.to_csv(table_path, index=False)
            print(f"  ✓ Saved: {table_path}")
            
            print("\n" + "-"*80)
            print("PERFORMANCE SUMMARY")
            print("-"*80)
            print(comparison_df.to_string(index=False))
        else:
            print("    No comparison table generated (missing data)")
        
        # 2. Equity curves
        print("\n[2/11] Plotting equity curves...")
        self.plot_equity_curves(
            os.path.join(output_dir, f'equity_curves_{timestamp}.png')
        )
        
        # 3. Cumulative returns
        print("\n[3/11] Plotting cumulative returns...")
        self.plot_cumulative_returns(
            os.path.join(output_dir, f'cumulative_returns_{timestamp}.png')
        )
        
        # 4. Drawdowns
        print("\n[4/11] Plotting drawdowns...")
        self.plot_drawdowns(
            os.path.join(output_dir, f'drawdowns_{timestamp}.png')
        )
        
        # 5. Rolling Sharpe
        print("\n[5/11] Plotting rolling Sharpe...")
        self.plot_rolling_sharpe(
            window=60,
            output_path=os.path.join(output_dir, f'rolling_sharpe_{timestamp}.png')
        )
        
        # 6. Returns distribution
        print("\n[6/11] Plotting returns distribution...")
        self.plot_returns_distribution(
            os.path.join(output_dir, f'returns_distribution_{timestamp}.png')
        )
        
        # 7. Risk-return scatter
        print("\n[7/11] Plotting risk-return scatter...")
        self.plot_risk_return_scatter(
            os.path.join(output_dir, f'risk_return_{timestamp}.png')
        )
        
        # 8. Portfolio composition
        print("\n[8/11] Plotting portfolio composition...")
        self.plot_portfolio_composition(
            os.path.join(output_dir, f'portfolio_composition_{timestamp}.png')
        )
        
        # 9. Monthly returns heatmap
        print("\n[9/11] Plotting monthly returns heatmap...")
        self.plot_monthly_returns_heatmap(
            os.path.join(output_dir, f'monthly_returns_{timestamp}.png')
        )
        
        # 10. Costs comparison
        print("\n[10/11] Plotting costs comparison...")
        self.plot_costs_comparison(
            os.path.join(output_dir, f'costs_comparison_{timestamp}.png')
        )
        
        # 11. Statistical tests
        print("\n[11/11] Performing statistical tests...")
        stats_df = self.perform_statistical_tests()
        
        if len(stats_df) > 0:
            stats_path = os.path.join(output_dir, f'statistical_tests_{timestamp}.csv')
            stats_df.to_csv(stats_path, index=False)
            print(f"  ✓ Saved: {stats_path}")
            
            print("\n" + "-"*80)
            print("STATISTICAL TESTS")
            print("-"*80)
            print(stats_df.to_string(index=False))
        else:
            print("    No statistical tests performed (insufficient data)")
        
        # 12. Generate text report
        print("\n[BONUS] Generating comprehensive text report...")
        self._generate_text_report(comparison_df, stats_df, output_dir, timestamp)
        
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved to: {output_dir}/")
        print(f"Timestamp: {timestamp}")
        
        return comparison_df
    
    def _generate_text_report(self, comparison_df, stats_df, output_dir, timestamp):
        """Generate detailed text report."""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("PORTFOLIO STRATEGIES COMPREHENSIVE COMPARISON REPORT")
        report_lines.append("="*80)
        report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of Strategies Compared: {len(self.strategies)}")
        
        if len(comparison_df) > 0:
            report_lines.append("\n" + "-"*80)
            report_lines.append("PERFORMANCE SUMMARY")
            report_lines.append("-"*80)
            
            for _, row in comparison_df.iterrows():
                report_lines.append(f"\n{row['Strategy']} ({row['Method']}):")
                report_lines.append(f"  Final Value:        ${row['Final Value ($)']:,.0f}")
                report_lines.append(f"  Total Return:       {row['Total Return (%)']:.2f}%")
                report_lines.append(f"  Annualized Return:  {row['Annual Return (%)']:.2f}%")
                report_lines.append(f"  Sharpe Ratio:       {row['Sharpe Ratio']:.3f}")
                report_lines.append(f"  Sortino Ratio:      {row['Sortino Ratio']:.3f}")
                report_lines.append(f"  Calmar Ratio:       {row['Calmar Ratio']:.3f}")
                report_lines.append(f"  Max Drawdown:       {row['Max Drawdown (%)']:.2f}%")
                report_lines.append(f"  Volatility:         {row['Volatility (%)']:.2f}%")
                report_lines.append(f"  Win Rate:           {row['Win Rate (%)']:.1f}%")
                report_lines.append(f"  Number of Assets:   {row['N Assets']}")
                report_lines.append(f"  Total Costs:        ${row['Total Costs ($)']:,.0f}")
                report_lines.append(f"  Rebalances:         {row['N Rebalances']}")
            
            # Rankings
            report_lines.append("\n" + "-"*80)
            report_lines.append("RANKINGS")
            report_lines.append("-"*80)
            
            best_sharpe_idx = comparison_df['Sharpe Ratio'].idxmax()
            best_return_idx = comparison_df['Annual Return (%)'].idxmax()
            lowest_dd_idx = comparison_df['Max Drawdown (%)'].idxmin()
            lowest_vol_idx = comparison_df['Volatility (%)'].idxmin()
            
            report_lines.append(f"\nBest Sharpe Ratio:     {comparison_df.loc[best_sharpe_idx, 'Strategy']}")
            report_lines.append(f"                       ({comparison_df.loc[best_sharpe_idx, 'Sharpe Ratio']:.3f})")
            report_lines.append(f"\nHighest Return:        {comparison_df.loc[best_return_idx, 'Strategy']}")
            report_lines.append(f"                       ({comparison_df.loc[best_return_idx, 'Annual Return (%)']:.2f}%)")
            report_lines.append(f"\nLowest Drawdown:       {comparison_df.loc[lowest_dd_idx, 'Strategy']}")
            report_lines.append(f"                       ({comparison_df.loc[lowest_dd_idx, 'Max Drawdown (%)']:.2f}%)")
            report_lines.append(f"\nLowest Volatility:     {comparison_df.loc[lowest_vol_idx, 'Strategy']}")
            report_lines.append(f"                       ({comparison_df.loc[lowest_vol_idx, 'Volatility (%)']:.2f}%)")
        
        if len(stats_df) > 0:
            report_lines.append("\n" + "-"*80)
            report_lines.append("STATISTICAL SIGNIFICANCE TESTS")
            report_lines.append("-"*80)
            report_lines.append("\nPairwise T-Tests (H0: returns are equal):")
            
            for _, row in stats_df.iterrows():
                report_lines.append(f"\n{row['Strategy 1']} vs {row['Strategy 2']}:")
                report_lines.append(f"  Mean Difference:  {row['Mean Diff (annual %)']:.2f}% (annualized)")
                report_lines.append(f"  T-Statistic:      {row['T-Statistic']:.4f}")
                report_lines.append(f"  P-Value:          {row['P-Value']:.4f}")
                report_lines.append(f"  Significant (5%): {row['Significant (5%)']}")
                report_lines.append(f"  Significant (1%): {row['Significant (1%)']}")
        
        report_lines.append("\n" + "="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        # Save report
        report_path = os.path.join(output_dir, f'comprehensive_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✓ Saved: {report_path}")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("PORTFOLIO STRATEGIES COMPARISON TOOL - FIXED VERSION v1.1")
    print("="*80)
    
    # Initialize comparator
    comparator = StrategyComparator(base_dir='.')
    
    # Define strategies to compare
    # MODIFY THIS LIST TO ADD/REMOVE STRATEGIES
    strategies_to_compare = [
        ('results_quantum_fixed', 'Quantum Fixed'),
        ('results_realistic_classical', 'Realistic Classical'),
        # Add more strategies here:
        # ('results_another_strategy', 'Another Strategy Label'),
    ]
    
    print("\nStrategies configured for comparison:")
    for folder, label in strategies_to_compare:
        print(f"  - {label} (folder: {folder})")
    
    # Load all strategies
    success = comparator.load_all_strategies(strategies_to_compare)
    
    if not success:
        print("\n ERROR: Failed to load strategies. Check folder names and file availability.")
        return 1
    
    # Generate comprehensive report
    try:
        comparison_df = comparator.generate_comprehensive_report(
            output_dir='./comparison_results'
        )
        
        print("\n" + "="*80)
        print("✓ ANALYSIS COMPLETE!")
        print("="*80)
        print("\nCheck ./comparison_results/ for all outputs:")
        print("  - comparison_table_*.csv")
        print("  - equity_curves_*.png")
        print("  - drawdowns_*.png")
        print("  - rolling_sharpe_*.png")
        print("  - returns_distribution_*.png")
        print("  - risk_return_*.png")
        print("  - portfolio_composition_*.png")
        print("  - monthly_returns_*.png")
        print("  - costs_comparison_*.png")
        print("  - statistical_tests_*.csv")
        print("  - comprehensive_report_*.txt")
        
        return 0
        
    except Exception as e:
        print(f"\n ERROR during report generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n\n FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)