"""
Quantum Portfolio Optimization - Main Execution Script
======================================================

Complete workflow integrating capital-constrained portfolio optimization
with quantum computing and comprehensive backtesting.

Features:
---------
- Fixed capital constraint: $100,000
- Optimal weight determination for 15 stocks
- Quarterly rebalancing with realistic costs
- Transaction costs: commission, slippage, bid-ask spread
- Tax modeling: short-term vs long-term capital gains
- In-sample/out-of-sample data split (80/20)
- Quantum optimization via D-Wave (optional)
- Comprehensive performance analysis

Usage:
------
python main.py --capital 100000 --n_assets 15 --risk_aversion 0.5

Arguments:
----------
--capital : float
    Fixed capital amount in dollars (default: 100000)
--n_assets : int
    Number of stocks in final portfolio (default: 15)
--risk_aversion : float
    Risk aversion parameter, 0=return focus, 1=risk focus (default: 0.5)
--use_quantum : flag
    Use D-Wave quantum annealer for optimization
--num_reads : int
    Number of quantum annealer samples (default: 2000)
--rebalance_threshold : float
    Rebalance if weight drift exceeds threshold (default: 0.05)
--output_dir : str
    Output directory for results (default: ./results_quantum_portfolio)

References:
-----------
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Ledoit & Wolf (2004). Honey, I Shrunk the Sample Covariance Matrix.
- Perold, A. F. (1988). The Implementation Shortfall.

Author: Portfolio Optimization System
Version: 3.0
Date: 2025
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from section1_improved import (
    CapitalConstrainedPortfolioSelector,
    define_universe
)
from section2_improved import (
    QuantumWeightOptimizer
)
from section3_improved import (
    ComprehensivePortfolioAnalyzer
)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Quantum Portfolio Optimization with Capital Constraint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Capital and portfolio parameters
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Fixed capital amount in dollars'
    )
    
    parser.add_argument(
        '--n_assets',
        type=int,
        default=15,
        help='Number of stocks in portfolio'
    )
    
    parser.add_argument(
        '--candidate_pool',
        type=int,
        default=30,
        help='Size of candidate pool for initial screening'
    )
    
    # Optimization parameters
    parser.add_argument(
        '--risk_aversion',
        type=float,
        default=0.5,
        help='Risk aversion parameter (0-1)'
    )
    
    parser.add_argument(
        '--use_quantum',
        action='store_true',
        help='Use D-Wave quantum annealer'
    )
    
    parser.add_argument(
        '--num_reads',
        type=int,
        default=2000,
        help='Number of quantum annealer reads'
    )
    
    # Data parameters
    parser.add_argument(
        '--lookback_years',
        type=int,
        default=5,
        help='Years of historical data'
    )
    
    parser.add_argument(
        '--universe_size',
        type=int,
        default=100,
        help='Size of stock universe'
    )
    
    # Rebalancing parameters
    parser.add_argument(
        '--rebalance_threshold',
        type=float,
        default=0.05,
        help='Rebalance if weight drift exceeds threshold'
    )
    
    # Cost parameters
    parser.add_argument(
        '--commission_bps',
        type=float,
        default=5.0,
        help='Commission in basis points'
    )
    
    parser.add_argument(
        '--slippage_bps',
        type=float,
        default=10.0,
        help='Slippage in basis points'
    )
    
    parser.add_argument(
        '--spread_bps',
        type=float,
        default=5.0,
        help='Bid-ask spread in basis points'
    )
    
    # Output parameters
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results_quantum_portfolio',
        help='Output directory'
    )
    
    return parser.parse_args()


def print_section_header(title, width=80, char='='):
    """
    Print formatted section header.
    
    Parameters
    ----------
    title : str
        Section title
    width : int
        Width of header line
    char : str
        Character for header line
    """
    print(f"\n{char * width}")
    print(title.center(width))
    print(f"{char * width}")


def print_subsection(title, width=80, char='-'):
    """
    Print formatted subsection header.
    
    Parameters
    ----------
    title : str
        Subsection title
    width : int
        Width of header line
    char : str
        Character for header line
    """
    print(f"\n{char * width}")
    print(title)
    print(f"{char * width}")


def main():
    """
    Main execution workflow.
    
    Workflow:
    ---------
    1. Parse arguments and initialize
    2. Download and prepare data (in-sample/out-of-sample split)
    3. Optimize portfolio weights (quantum or classical)
    4. Simulate quarterly rebalancing with realistic costs
    5. Generate comprehensive performance analysis
    6. Save all results and visualizations
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for error)
    """
    args = parse_arguments()
    
    # Print execution header
    print_section_header("QUANTUM PORTFOLIO OPTIMIZATION SYSTEM")
    print(f"\nExecution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Capital Constraint: ${args.capital:,.0f}")
    print(f"Target Portfolio Size: {args.n_assets} stocks")
    print(f"Rebalancing Frequency: Quarterly")
    print(f"Optimization Method: {'Quantum (D-Wave)' if args.use_quantum else 'Classical'}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # =========================================================================
    # SECTION 1: DATA PREPARATION AND PORTFOLIO SELECTION
    # =========================================================================
    print_section_header("SECTION 1: DATA PREPARATION")
    
    # Define stock universe
    universe = define_universe()[:args.universe_size]
    print(f"\nStock Universe: {len(universe)} stocks across all major sectors")
    
    # Initialize selector with capital constraint
    selector = CapitalConstrainedPortfolioSelector(
        universe=universe,
        capital=args.capital,
        lookback_years=args.lookback_years,
        rebalance_frequency='quarterly'
    )
    
    # Download historical data
    print("\nDownloading historical market data...")
    try:
        data = selector.download_data()
    except Exception as e:
        print(f"\nERROR: Failed to download data - {e}")
        print("Please check internet connection and try again.")
        return 1
    
    # Use available tickers as candidates
    candidate_tickers = data.columns.tolist()[:min(args.candidate_pool, len(data.columns))]
    print(f"\nCandidate Pool: {len(candidate_tickers)} stocks selected for optimization")
    
    # Get in-sample statistics for optimization
    expected_returns, cov_matrix, asset_names = selector.get_statistics_for_quantum(
        candidate_tickers
    )
    
    print_subsection("IN-SAMPLE STATISTICS (Training Data)")
    print(f"Expected Returns: [{expected_returns.min():.2%}, {expected_returns.max():.2%}]")
    print(f"Average Volatility: {np.sqrt(np.diag(cov_matrix)).mean():.2%}")
    corr_mean = (cov_matrix.sum() - np.trace(cov_matrix)) / (len(cov_matrix) * (len(cov_matrix)-1)) / (np.sqrt(np.diag(cov_matrix)).mean()**2 + 1e-10)
    print(f"Average Correlation: {corr_mean:.3f}")
    
    # =========================================================================
    # SECTION 2: PORTFOLIO WEIGHT OPTIMIZATION
    # =========================================================================
    print_section_header("SECTION 2: PORTFOLIO OPTIMIZATION")
    
    if args.use_quantum:
        print("\nInitializing quantum optimizer (D-Wave)...")
        
        try:
            optimizer = QuantumWeightOptimizer(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                asset_names=asset_names,
                capital=args.capital,
                is_in_sample=True
            )
            
            # Formulate QUBO problem
            print("\nFormulating QUBO problem...")
            bqm = optimizer.formulate_qubo_with_weights(
                risk_aversion=args.risk_aversion,
                n_select=args.n_assets,
                penalty_strength=20.0
            )
            
            # Solve on quantum annealer
            print("\nSubmitting to quantum annealer...")
            quantum_results = optimizer.solve_quantum(
                bqm=bqm,
                num_reads=args.num_reads,
                annealing_time=20
            )
            
            selected_indices = quantum_results['selected_assets']
            optimal_weights = quantum_results['optimal_weights']
            
        except Exception as e:
            print(f"\nERROR: Quantum optimization failed - {e}")
            print("Falling back to classical optimization...")
            args.use_quantum = False
    
    if not args.use_quantum:
        # Classical optimization
        print("\nUsing classical optimization (Markowitz mean-variance)...")
        
        selected_tickers, optimal_weights, opt_stats = selector.optimize_weights(
            candidate_tickers=candidate_tickers,
            n_assets=args.n_assets,
            risk_aversion=args.risk_aversion,
            return_method='shrinkage',
            cov_method='ledoit_wolf'
        )
        
        selected_indices = [asset_names.index(t) for t in selected_tickers]
        
        quantum_results = {
            'selected_tickers': selected_tickers,
            'optimal_weights': optimal_weights,
            'selected_assets': selected_indices,
            'n_selected': len(selected_tickers),
            'qpu_info': {'chip_id': 'Classical', 'topology': 'CPU'}
        }
    
    # Display optimized portfolio
    print_subsection("OPTIMIZED PORTFOLIO")
    print(f"Assets Selected: {len(selected_indices)}")
    print(f"\n{'Ticker':<8} {'Weight':>10} {'$ Amount':>14} {'Exp.Return':>12} {'Volatility':>12}")
    print("-" * 80)
    
    for i, (ticker, weight) in enumerate(zip(quantum_results['selected_tickers'], optimal_weights)):
        idx = selected_indices[i]
        amount = args.capital * weight
        ret = expected_returns[idx]
        vol = np.sqrt(cov_matrix[idx, idx])
        print(f"{ticker:<8} {weight:>9.2%} ${amount:>13,.0f} {ret:>11.2%} {vol:>11.2%}")
    
    print("-" * 80)
    print(f"{'TOTAL':<8} {optimal_weights.sum():>9.2%} ${args.capital:>13,.0f}")
    
    # Calculate portfolio-level metrics
    portfolio_return = np.dot(optimal_weights, expected_returns[selected_indices])
    portfolio_vol = np.sqrt(np.dot(
        optimal_weights,
        np.dot(cov_matrix[np.ix_(selected_indices, selected_indices)], optimal_weights)
    ))
    portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
    print_subsection("PORTFOLIO METRICS (In-Sample)")
    print(f"Expected Return:     {portfolio_return:.2%}")
    print(f"Expected Volatility: {portfolio_vol:.2%}")
    print(f"Expected Sharpe:     {portfolio_sharpe:.3f}")
    print(f"HHI:                 {np.sum(optimal_weights**2):.4f}")
    print(f"Effective N Assets:  {1/np.sum(optimal_weights**2):.2f}")
    
    # Save portfolio allocation
    allocation_df = pd.DataFrame({
        'Ticker': quantum_results['selected_tickers'],
        'Weight': optimal_weights,
        'Dollar_Amount': optimal_weights * args.capital,
        'Expected_Return': expected_returns[selected_indices],
        'Volatility': np.sqrt(np.diag(cov_matrix))[selected_indices]
    })
    alloc_path = os.path.join(args.output_dir, f'portfolio_allocation_{timestamp}.csv')
    allocation_df.to_csv(alloc_path, index=False)
    print(f"\nPortfolio allocation saved: {alloc_path}")
    
    # =========================================================================
    # SECTION 3: OUT-OF-SAMPLE BACKTESTING WITH QUARTERLY REBALANCING
    # =========================================================================
    print_section_header("SECTION 3: OUT-OF-SAMPLE BACKTESTING")
    
    # Initialize analyzer
    analyzer = ComprehensivePortfolioAnalyzer(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        asset_names=asset_names,
        capital=args.capital
    )
    
    # Update cost parameters
    analyzer.commission_bps = args.commission_bps
    analyzer.slippage_bps_base = args.slippage_bps
    analyzer.spread_bps = args.spread_bps
    
    # Run quarterly rebalancing simulation
    print("\nSimulating quarterly rebalancing with realistic costs...")
    
    try:
        simulation_results = analyzer.simulate_quarterly_rebalancing(
            selected_indices=selected_indices,
            target_weights=optimal_weights,
            returns_data=selector.out_sample_returns,
            prices_data=selector.out_sample_prices,
            rebalance_threshold=args.rebalance_threshold
        )
    except Exception as e:
        print(f"\nERROR: Simulation failed - {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Calculate comprehensive statistics
    print("\nCalculating comprehensive performance statistics...")
    statistics = analyzer.calculate_comprehensive_statistics(
        simulation_results=simulation_results,
        selected_indices=selected_indices,
        weights=optimal_weights
    )
    
    # Benchmark comparison
    print("\nComparing to benchmark strategies...")
    benchmark_comparison = analyzer.compare_to_benchmarks(
        simulation_results=simulation_results,
        selected_indices=selected_indices,
        weights=optimal_weights
    )
    
    print_subsection("BENCHMARK COMPARISON")
    print(benchmark_comparison.to_string(index=False))
    
    # =========================================================================
    # GENERATE COMPREHENSIVE OUTPUTS
    # =========================================================================
    print_section_header("GENERATING REPORTS AND VISUALIZATIONS")
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, f'statistics_{timestamp}.csv')
    statistics.to_csv(stats_path)
    print(f"\nStatistics saved: {stats_path}")
    
    # Save benchmark comparison
    bench_path = os.path.join(args.output_dir, f'benchmarks_{timestamp}.csv')
    benchmark_comparison.to_csv(bench_path, index=False)
    print(f"Benchmarks saved: {bench_path}")
    
    # Save portfolio time series
    timeseries_df = pd.DataFrame({
        'Date': simulation_results['dates'],
        'Portfolio_Value': simulation_results['portfolio_values']
    })
    timeseries_path = os.path.join(args.output_dir, f'portfolio_values_{timestamp}.csv')
    timeseries_df.to_csv(timeseries_path, index=False)
    print(f"Time series saved: {timeseries_path}")
    
    # Generate comprehensive visualization
    print("\nGenerating comprehensive visualization dashboard...")
    viz_path = os.path.join(args.output_dir, f'analysis_dashboard_{timestamp}.png')
    analyzer.visualize_comprehensive_results(
        simulation_results=simulation_results,
        statistics=statistics,
        comparison=benchmark_comparison,
        selected_indices=selected_indices,
        weights=optimal_weights,
        output_path=viz_path
    )
    
    # Generate JSON report
    json_report = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'execution_time': timestamp,
            'version': '3.0'
        },
        'configuration': {
            'capital': args.capital,
            'n_assets': args.n_assets,
            'risk_aversion': args.risk_aversion,
            'lookback_years': args.lookback_years,
            'use_quantum': args.use_quantum,
            'commission_bps': args.commission_bps,
            'slippage_bps': args.slippage_bps,
            'spread_bps': args.spread_bps,
            'rebalance_threshold': args.rebalance_threshold
        },
        'portfolio': {
            'tickers': quantum_results['selected_tickers'],
            'weights': optimal_weights.tolist(),
            'expected_return': float(portfolio_return),
            'expected_volatility': float(portfolio_vol),
            'expected_sharpe': float(portfolio_sharpe)
        },
        'out_of_sample_performance': {
            'final_value': float(simulation_results['final_value']),
            'total_return': float(simulation_results['total_return']),
            'annual_return': float(simulation_results['annual_return']),
            'annual_volatility': float(simulation_results['annual_volatility']),
            'sharpe_ratio': float(simulation_results['sharpe_ratio']),
            'sortino_ratio': float(simulation_results['sortino_ratio']),
            'calmar_ratio': float(simulation_results['calmar_ratio']),
            'max_drawdown': float(simulation_results['max_drawdown']),
            'win_rate': float(simulation_results['win_rate']),
            'var_95': float(simulation_results['var_95'])
        },
        'costs': {
            'total_transaction_costs': float(simulation_results['total_transaction_costs']),
            'total_taxes': float(simulation_results['total_taxes']),
            'total_costs': float(simulation_results['total_costs']),
            'avg_quarterly_turnover': float(simulation_results['avg_quarterly_turnover']),
            'n_rebalances': simulation_results['n_rebalances']
        },
        'quantum_info': {
            'qpu_chip': quantum_results['qpu_info']['chip_id'],
            'topology': quantum_results['qpu_info']['topology']
        }
    }
    
    json_path = os.path.join(args.output_dir, f'complete_report_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(json_report, f, indent=2)
    print(f"JSON report saved: {json_path}")
    
    # Generate text report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("QUANTUM PORTFOLIO OPTIMIZATION - FINAL REPORT")
    report_lines.append("="*80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    report_lines.append(f"\n{'-'*80}")
    report_lines.append("CONFIGURATION")
    report_lines.append(f"{'-'*80}")
    report_lines.append(f"Capital:               ${args.capital:,.0f}")
    report_lines.append(f"Portfolio Size:        {args.n_assets} stocks")
    report_lines.append(f"Optimization:          {'Quantum (D-Wave)' if args.use_quantum else 'Classical'}")
    report_lines.append(f"Risk Aversion:         {args.risk_aversion}")
    report_lines.append(f"Rebalancing:           Quarterly (threshold: {args.rebalance_threshold:.1%})")
    
    report_lines.append(f"\n{'-'*80}")
    report_lines.append("PORTFOLIO COMPOSITION")
    report_lines.append(f"{'-'*80}")
    for ticker, weight in zip(quantum_results['selected_tickers'], optimal_weights):
        amount = args.capital * weight
        report_lines.append(f"{ticker:<8} {weight:>7.2%}  ${amount:>11,.0f}")
    
    report_lines.append(f"\n{'-'*80}")
    report_lines.append("OUT-OF-SAMPLE PERFORMANCE")
    report_lines.append(f"{'-'*80}")
    report_lines.append(f"Test Period:           {simulation_results['n_periods']} days ({simulation_results['years']:.2f} years)")
    report_lines.append(f"Initial Value:         ${simulation_results['initial_value']:,.0f}")
    report_lines.append(f"Final Value:           ${simulation_results['final_value']:,.0f}")
    report_lines.append(f"Total Return:          {simulation_results['total_return']:.2%}")
    report_lines.append(f"Annualized Return:     {simulation_results['annual_return']:.2%}")
    report_lines.append(f"Annualized Volatility: {simulation_results['annual_volatility']:.2%}")
    report_lines.append(f"Sharpe Ratio:          {simulation_results['sharpe_ratio']:.3f}")
    report_lines.append(f"Sortino Ratio:         {simulation_results['sortino_ratio']:.3f}")
    report_lines.append(f"Calmar Ratio:          {simulation_results['calmar_ratio']:.3f}")
    report_lines.append(f"Maximum Drawdown:      {simulation_results['max_drawdown']:.2%}")
    report_lines.append(f"Win Rate:              {simulation_results['win_rate']:.1%}")
    report_lines.append(f"VaR (95%):             {simulation_results['var_95']:.2%}")
    
    report_lines.append(f"\n{'-'*80}")
    report_lines.append("COSTS AND REBALANCING")
    report_lines.append(f"{'-'*80}")
    report_lines.append(f"Number of Rebalances:  {simulation_results['n_rebalances']}")
    report_lines.append(f"Avg Quarterly Turnover: {simulation_results['avg_quarterly_turnover']:.1%}")
    report_lines.append(f"Transaction Costs:     ${simulation_results['total_transaction_costs']:,.0f}")
    report_lines.append(f"Taxes Paid:            ${simulation_results['total_taxes']:,.0f}")
    report_lines.append(f"Total Costs:           ${simulation_results['total_costs']:,.0f} ({simulation_results['total_costs']/args.capital:.2%} of capital)")
    
    report_lines.append(f"\n{'-'*80}")
    report_lines.append("COMPARISON TO BENCHMARKS")
    report_lines.append(f"{'-'*80}")
    report_lines.append(benchmark_comparison.to_string(index=False))
    
    # Conclusion
    quantum_sharpe = simulation_results['sharpe_ratio']
    best_benchmark_sharpe = benchmark_comparison.iloc[1:]['Sharpe'].max()
    
    report_lines.append(f"\n{'-'*80}")
    report_lines.append("CONCLUSION")
    report_lines.append(f"{'-'*80}")
    
    if quantum_sharpe > best_benchmark_sharpe:
        improvement = ((quantum_sharpe / best_benchmark_sharpe) - 1) * 100
        report_lines.append(f"Portfolio OUTPERFORMS best benchmark by {improvement:.1f}%")
        report_lines.append(f"(Sharpe: {quantum_sharpe:.3f} vs {best_benchmark_sharpe:.3f})")
    else:
        gap = ((best_benchmark_sharpe / quantum_sharpe) - 1) * 100
        report_lines.append(f"Portfolio underperforms best benchmark by {gap:.1f}%")
        report_lines.append(f"(Sharpe: {quantum_sharpe:.3f} vs {best_benchmark_sharpe:.3f})")
    
    report_lines.append(f"\nTotal cost impact: {simulation_results['total_costs']/args.capital:.2%} of initial capital")
    report_lines.append(f"Net profit: ${simulation_results['final_value'] - args.capital:,.0f}")
    
    report_lines.append(f"\n{'='*80}")
    report_lines.append("END OF REPORT")
    report_lines.append(f"{'='*80}")
    
    report_text = '\n'.join(report_lines)
    
    # Save text report
    text_report_path = os.path.join(args.output_dir, f'final_report_{timestamp}.txt')
    with open(text_report_path, 'w') as f:
        f.write(report_text)
    print(f"Text report saved: {text_report_path}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_section_header("EXECUTION SUMMARY")
    
    print(f"\n{'Final Portfolio Value:':<30} ${simulation_results['final_value']:,.0f}")
    print(f"{'Net Profit/Loss:':<30} ${simulation_results['final_value'] - args.capital:,.0f}")
    print(f"{'Total Return:':<30} {simulation_results['total_return']:.2%}")
    print(f"{'Annualized Return:':<30} {simulation_results['annual_return']:.2%}")
    print(f"{'Sharpe Ratio:':<30} {simulation_results['sharpe_ratio']:.3f}")
    print(f"{'Max Drawdown:':<30} {simulation_results['max_drawdown']:.2%}")
    print(f"{'Total Costs:':<30} ${simulation_results['total_costs']:,.0f}")
    
    print(f"\n{'All outputs saved to:':<30} {args.output_dir}/")
    print(f"{'Execution completed:':<30} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)