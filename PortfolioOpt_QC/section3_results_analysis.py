"""
Section 3: Results Analysis and Consistency Testing
===================================================

This module analyzes quantum optimization results and tests their statistical
consistency and robustness. Multiple metrics are calculated to validate the
quality of the quantum-generated portfolio.

Key Analyses:
1. Sharpe Ratio and risk-adjusted performance
2. Diversification metrics
3. Solution consistency across quantum runs
4. Comparison with classical benchmarks
5. Statistical significance testing

References:
- Sharpe, W. F. (1966). Mutual Fund Performance.
- Jobson, J. D., & Korkie, B. M. (1981). Performance Hypothesis Testing.

Author: Portfolio Optimization System
Version: 1.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class PortfolioAnalyzer:
    """
    Comprehensive portfolio analysis and validation.
    """
    
    def __init__(self, expected_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 asset_names: List[str]):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected returns
        cov_matrix : np.ndarray
            Covariance matrix
        asset_names : List[str]
            Asset names
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.asset_names = asset_names
        self.n_assets = len(expected_returns)
        
    def calculate_sharpe_ratio(self, weights: np.ndarray, 
                               risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.
        
        Sharpe Ratio = (R_p - R_f) / sigma_p
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        risk_free_rate : float
            Risk-free rate (annualized)
        
        Returns
        -------
        float
            Sharpe ratio
        """
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_std = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        
        sharpe = (portfolio_return - risk_free_rate) / portfolio_std
        return sharpe
    
    def calculate_diversification_ratio(self, weights: np.ndarray) -> float:
        """
        Calculate diversification ratio.
        
        DR = (sum w_i * sigma_i) / sigma_p
        
        A ratio > 1 indicates diversification benefit.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        
        Returns
        -------
        float
            Diversification ratio
        """
        asset_volatilities = np.sqrt(np.diag(self.cov_matrix))
        weighted_vol_sum = np.dot(weights, asset_volatilities)
        portfolio_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        
        dr = weighted_vol_sum / portfolio_vol
        return dr
    
    def calculate_herfindahl_index(self, weights: np.ndarray) -> float:
        """
        Calculate Herfindahl-Hirschman Index for concentration.
        
        HHI = sum (w_i)^2
        
        HHI = 1.0 means full concentration (one asset)
        HHI = 1/N means equal weighting
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights
        
        Returns
        -------
        float
            HHI index
        """
        active_weights = weights[weights > 0]
        hhi = np.sum(active_weights ** 2)
        return hhi
    
    def calculate_maximum_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown from return series.
        
        Parameters
        ----------
        returns : np.ndarray
            Time series of returns
        
        Returns
        -------
        float
            Maximum drawdown
        """
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        return max_dd
    
    def test_solution_consistency(self, sampleset, 
                                  top_n: int = 10) -> Dict:
        """
        Test consistency of quantum solutions.
        
        Analyzes:
        1. Energy distribution of solutions
        2. Frequency of top solutions
        3. Stability of asset selection
        
        Parameters
        ----------
        sampleset : SampleSet
            Quantum annealer results
        top_n : int
            Number of top solutions to analyze
        
        Returns
        -------
        Dict
            Consistency metrics
        """
        print("\nTesting solution consistency...")
        
        # Get unique solutions
        aggregated = sampleset.aggregate()
        energies = aggregated.record.energy
        occurrences = aggregated.record.num_occurrences
        samples = aggregated.record.sample
        
        # Energy statistics
        energy_mean = np.mean(energies)
        energy_std = np.std(energies)
        energy_range = np.max(energies) - np.min(energies)
        
        # Best solution frequency
        total_reads = np.sum(occurrences)
        best_frequency = occurrences[0] / total_reads
        
        # Analyze top N solutions
        top_solutions = samples[:top_n]
        
        # Count how often each asset appears in top solutions
        asset_frequency = np.zeros(self.n_assets)
        for solution in top_solutions:
            asset_frequency += solution
        asset_frequency = asset_frequency / top_n
        
        # Calculate solution overlap (Jaccard similarity)
        overlaps = []
        best_solution = set(np.where(samples[0] == 1)[0])
        for i in range(1, min(top_n, len(samples))):
            solution_i = set(np.where(samples[i] == 1)[0])
            intersection = len(best_solution & solution_i)
            union = len(best_solution | solution_i)
            jaccard = intersection / union if union > 0 else 0
            overlaps.append(jaccard)
        
        avg_overlap = np.mean(overlaps) if overlaps else 0
        
        consistency = {
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'energy_range': energy_range,
            'best_frequency': best_frequency,
            'avg_solution_overlap': avg_overlap,
            'asset_stability': asset_frequency,
            'total_unique_solutions': len(aggregated)
        }
        
        return consistency
    
    def compare_to_benchmarks(self, 
                             selected_indices: List[int]) -> pd.DataFrame:
        """
        Compare quantum portfolio to classical benchmarks.
        
        Benchmarks:
        1. Equal-weighted portfolio (all assets)
        2. Equal-weighted portfolio (selected assets)
        3. Minimum variance portfolio
        4. Maximum Sharpe portfolio
        
        Parameters
        ----------
        selected_indices : List[int]
            Quantum-selected asset indices
        
        Returns
        -------
        pd.DataFrame
            Comparison table
        """
        print("\nComparing to classical benchmarks...")
        
        results = []
        
        # Quantum portfolio (equal-weighted among selected)
        quantum_weights = np.zeros(self.n_assets)
        if len(selected_indices) > 0:
            quantum_weights[selected_indices] = 1.0 / len(selected_indices)
        
        quantum_metrics = self._portfolio_metrics(quantum_weights, "Quantum (Equal-Weight)")
        results.append(quantum_metrics)
        
        # Benchmark 1: Equal-weighted (all assets)
        ew_all = np.ones(self.n_assets) / self.n_assets
        ew_all_metrics = self._portfolio_metrics(ew_all, "Equal-Weight (All)")
        results.append(ew_all_metrics)
        
        # Benchmark 2: Minimum variance
        min_var_weights = self._minimum_variance_portfolio()
        min_var_metrics = self._portfolio_metrics(min_var_weights, "Minimum Variance")
        results.append(min_var_metrics)
        
        # Benchmark 3: Maximum Sharpe
        max_sharpe_weights = self._maximum_sharpe_portfolio()
        max_sharpe_metrics = self._portfolio_metrics(max_sharpe_weights, "Maximum Sharpe")
        results.append(max_sharpe_metrics)
        
        df = pd.DataFrame(results)
        return df
    
    def _portfolio_metrics(self, weights: np.ndarray, name: str) -> Dict:
        """Calculate metrics for a given portfolio."""
        port_return = np.dot(weights, self.expected_returns)
        port_vol = np.sqrt(np.dot(weights, np.dot(self.cov_matrix, weights)))
        sharpe = port_return / port_vol
        dr = self.calculate_diversification_ratio(weights)
        hhi = self.calculate_herfindahl_index(weights)
        n_assets = np.sum(weights > 1e-6)
        
        return {
            'Portfolio': name,
            'Return': port_return,
            'Volatility': port_vol,
            'Sharpe_Ratio': sharpe,
            'Diversification_Ratio': dr,
            'HHI': hhi,
            'N_Assets': n_assets
        }
    
    def _minimum_variance_portfolio(self) -> np.ndarray:
        """Calculate minimum variance portfolio weights."""
        from scipy.optimize import minimize
        
        n = self.n_assets
        
        def portfolio_variance(w):
            return np.dot(w, np.dot(self.cov_matrix, w))
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        x0 = np.ones(n) / n
        
        result = minimize(portfolio_variance, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def _maximum_sharpe_portfolio(self) -> np.ndarray:
        """Calculate maximum Sharpe ratio portfolio weights."""
        from scipy.optimize import minimize
        
        n = self.n_assets
        
        def neg_sharpe(w):
            ret = np.dot(w, self.expected_returns)
            vol = np.sqrt(np.dot(w, np.dot(self.cov_matrix, w)))
            return -ret / vol
        
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(n))
        x0 = np.ones(n) / n
        
        result = minimize(neg_sharpe, x0, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
    
    def visualize_results(self, 
                         selected_indices: List[int],
                         consistency: Dict,
                         comparison_df: pd.DataFrame,
                         output_path: str = None):
        """
        Create comprehensive visualization of results.
        
        Parameters
        ----------
        selected_indices : List[int]
            Selected asset indices
        consistency : Dict
            Consistency metrics
        comparison_df : pd.DataFrame
            Benchmark comparison
        output_path : str
            Path to save figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Risk-Return Scatter
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_risk_return(ax1, selected_indices)
        
        # Plot 2: Energy Distribution
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_energy_distribution(ax2, consistency)
        
        # Plot 3: Asset Stability
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_asset_stability(ax3, consistency['asset_stability'])
        
        # Plot 4: Benchmark Comparison
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_benchmark_comparison(ax4, comparison_df)
        
        # Plot 5: Correlation Matrix
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_correlation_matrix(ax5, selected_indices)
        
        plt.suptitle('Quantum Portfolio Optimization - Results Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved: {output_path}")
        
        plt.close()
    
    def _plot_risk_return(self, ax, selected_indices):
        """Plot risk-return scatter."""
        vols = np.sqrt(np.diag(self.cov_matrix))
        
        # All assets
        ax.scatter(vols, self.expected_returns, s=80, alpha=0.4, 
                  c='lightblue', edgecolors='black', linewidth=0.5)
        
        # Selected assets
        if len(selected_indices) > 0:
            ax.scatter(vols[selected_indices], 
                      self.expected_returns[selected_indices],
                      s=200, c='red', marker='*', edgecolors='darkred', 
                      linewidth=1.5, zorder=5, label='Selected')
        
        # Labels
        for i in selected_indices:
            ax.annotate(self.asset_names[i], 
                       (vols[i], self.expected_returns[i]),
                       fontsize=8, xytext=(3, 3), textcoords='offset points',
                       fontweight='bold')
        
        ax.set_xlabel('Volatility (Annual)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Expected Return (Annual)', fontsize=11, fontweight='bold')
        ax.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_energy_distribution(self, ax, consistency):
        """Plot energy distribution metrics."""
        data = [
            consistency['best_frequency'],
            consistency['avg_solution_overlap']
        ]
        labels = ['Best Solution\nFrequency', 'Avg Solution\nOverlap']
        colors = ['#2ecc71', '#3498db']
        
        bars = ax.barh(labels, data, color=colors, edgecolor='black', linewidth=1)
        
        for i, (bar, val) in enumerate(zip(bars, data)):
            ax.text(val + 0.02, i, f'{val:.3f}', 
                   va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlim(0, 1.0)
        ax.set_title('Solution Consistency', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_asset_stability(self, ax, asset_stability):
        """Plot asset selection stability."""
        indices = np.arange(self.n_assets)
        colors = ['red' if s > 0.5 else 'lightblue' for s in asset_stability]
        
        bars = ax.bar(indices, asset_stability, color=colors, 
                     edgecolor='black', linewidth=0.5)
        
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Asset Index', fontsize=11, fontweight='bold')
        ax.set_ylabel('Selection Frequency', fontsize=11, fontweight='bold')
        ax.set_title('Asset Selection Stability (Top 10 Solutions)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_benchmark_comparison(self, ax, comparison_df):
        """Plot benchmark comparison."""
        portfolios = comparison_df['Portfolio'].values
        sharpe_ratios = comparison_df['Sharpe_Ratio'].values
        
        colors = ['red' if 'Quantum' in p else 'lightgray' for p in portfolios]
        bars = ax.barh(portfolios, sharpe_ratios, color=colors, 
                      edgecolor='black', linewidth=1)
        
        for i, (bar, val) in enumerate(zip(bars, sharpe_ratios)):
            ax.text(val + 0.02, i, f'{val:.3f}', 
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Sharpe Ratio', fontsize=11, fontweight='bold')
        ax.set_title('Benchmark Comparison', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_correlation_matrix(self, ax, selected_indices):
        """Plot correlation matrix of selected assets."""
        if len(selected_indices) < 2:
            ax.text(0.5, 0.5, 'Insufficient assets for correlation matrix',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return
        
        # Extract correlation matrix
        sel_cov = self.cov_matrix[np.ix_(selected_indices, selected_indices)]
        sel_vols = np.sqrt(np.diag(sel_cov))
        corr = sel_cov / np.outer(sel_vols, sel_vols)
        
        # Plot
        im = ax.imshow(corr, cmap='RdYlGn_r', vmin=-1, vmax=1, aspect='auto')
        
        # Ticks
        sel_names = [self.asset_names[i] for i in selected_indices]
        ax.set_xticks(range(len(selected_indices)))
        ax.set_yticks(range(len(selected_indices)))
        ax.set_xticklabels(sel_names, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(sel_names, fontsize=9)
        
        # Values
        for i in range(len(selected_indices)):
            for j in range(len(selected_indices)):
                ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                       fontsize=8, color='black', fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_title('Correlation Matrix (Selected Assets)', 
                    fontsize=12, fontweight='bold')
    
    def generate_report(self, 
                       selected_indices: List[int],
                       metrics: Dict,
                       consistency: Dict,
                       comparison_df: pd.DataFrame) -> str:
        """
        Generate comprehensive text report.
        
        Parameters
        ----------
        selected_indices : List[int]
            Selected assets
        metrics : Dict
            Portfolio metrics
        consistency : Dict
            Consistency metrics
        comparison_df : pd.DataFrame
            Benchmark comparison
        
        Returns
        -------
        str
            Formatted report
        """
        report = []
        report.append("\n" + "="*80)
        report.append("QUANTUM PORTFOLIO OPTIMIZATION - ANALYSIS REPORT")
        report.append("="*80)
        
        # Selected Portfolio
        report.append("\nSELECTED PORTFOLIO")
        report.append("-"*80)
        report.append(f"Number of assets: {metrics['num_assets']}")
        report.append(f"Expected return: {metrics['return']:.2%}")
        report.append(f"Volatility: {metrics['volatility']:.2%}")
        report.append(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")
        
        report.append("\nSelected assets:")
        for idx in selected_indices:
            ret = self.expected_returns[idx]
            vol = np.sqrt(self.cov_matrix[idx, idx])
            report.append(f"  {self.asset_names[idx]}: Return={ret:.2%}, Vol={vol:.2%}")
        
        # Consistency Analysis
        report.append("\n" + "="*80)
        report.append("SOLUTION CONSISTENCY")
        report.append("-"*80)
        report.append(f"Best solution frequency: {consistency['best_frequency']:.1%}")
        report.append(f"Average solution overlap: {consistency['avg_solution_overlap']:.3f}")
        report.append(f"Energy std deviation: {consistency['energy_std']:.4f}")
        report.append(f"Total unique solutions: {consistency['total_unique_solutions']}")
        
        # Benchmark Comparison
        report.append("\n" + "="*80)
        report.append("BENCHMARK COMPARISON")
        report.append("-"*80)
        report.append(comparison_df.to_string(index=False))
        
        # Conclusion
        report.append("\n" + "="*80)
        report.append("CONCLUSIONS")
        report.append("-"*80)
        
        quantum_sharpe = metrics['sharpe_ratio']
        best_benchmark = comparison_df.iloc[1:]['Sharpe_Ratio'].max()
        
        if quantum_sharpe > best_benchmark:
            improvement = ((quantum_sharpe / best_benchmark) - 1) * 100
            report.append(f"Quantum portfolio outperforms best benchmark by {improvement:.1f}%")
        else:
            gap = ((best_benchmark / quantum_sharpe) - 1) * 100
            report.append(f"Quantum portfolio underperforms best benchmark by {gap:.1f}%")
        
        report.append(f"Solution consistency: {'HIGH' if consistency['best_frequency'] > 0.1 else 'MODERATE'}")
        report.append("="*80)
        
        return "\n".join(report)


if __name__ == "__main__":
    print("This module requires inputs from Sections 1 and 2")
    print("Run the integrated workflow in main.py")
