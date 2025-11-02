"""
Quantum Portfolio Optimization - Main Workflow
==============================================

This is the main execution script that integrates all three sections:
1. Portfolio Selection (Markowitz or Momentum)
2. Quantum Optimization (QUBO formulation and D-Wave execution)
3. Results Analysis and Consistency Testing

Usage:
------
python main.py --method markowitz --budget 5 --risk_aversion 0.5 --num_reads 2000

Arguments:
----------
--method : str
    Portfolio selection method ('markowitz' or 'momentum')
--budget : int
    Number of assets to select in quantum optimization
--risk_aversion : float
    Risk aversion parameter lambda in [0, 1]
--num_reads : int
    Number of quantum annealer reads
--output : str
    Output directory for results

Author: Portfolio Optimization System
Version: 1.0
Date: 2025
"""

import argparse
import os
import sys
from datetime import datetime

# Import modules from sections
from section1_portfolio_selection import (
    PortfolioSelector, 
    define_sp500_universe
)
from section2_quantum_optimization import (
    QuantumPortfolioOptimizer
)
from section3_results_analysis import (
    PortfolioAnalyzer
)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Quantum Portfolio Optimization System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='markowitz',
        choices=['markowitz', 'momentum'],
        help='Portfolio selection method'
    )
    
    parser.add_argument(
        '--budget',
        type=int,
        default=5,
        help='Number of assets to select'
    )
    
    parser.add_argument(
        '--risk_aversion',
        type=float,
        default=0.5,
        help='Risk aversion parameter (0=return only, 1=risk only)'
    )
    
    parser.add_argument(
        '--num_reads',
        type=int,
        default=2000,
        help='Number of quantum annealer reads'
    )
    
    parser.add_argument(
        '--penalty_strength',
        type=float,
        default=15.0,
        help='Penalty strength for budget constraint'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--universe_size',
        type=int,
        default=50,
        help='Number of stocks in initial universe'
    )
    
    parser.add_argument(
        '--candidate_assets',
        type=int,
        default=15,
        help='Number of candidate assets from selection method'
    )
    
    return parser.parse_args()


def main():
    """
    Main execution workflow.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Print header
    print("\n" + "="*80)
    print("QUANTUM PORTFOLIO OPTIMIZATION SYSTEM")
    print("="*80)
    print(f"\nExecution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nConfiguration:")
    print(f"  Selection method: {args.method}")
    print(f"  Universe size: {args.universe_size}")
    print(f"  Candidate assets: {args.candidate_assets}")
    print(f"  Budget (final selection): {args.budget}")
    print(f"  Risk aversion: {args.risk_aversion}")
    print(f"  Quantum reads: {args.num_reads}")
    print(f"  Output directory: {args.output}")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # =========================================================================
    # SECTION 1: PORTFOLIO SELECTION
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 1: PORTFOLIO SELECTION")
    print("="*80)
    
    # Define universe
    universe = define_sp500_universe()[:args.universe_size]
    print(f"\nDefined universe of {len(universe)} stocks")
    
    # Initialize selector
    selector = PortfolioSelector(universe, lookback_years=3)
    
    # Download data
    try:
        selector.download_data()
    except Exception as e:
        print(f"\nError downloading data: {e}")
        print("Please check your internet connection and try again.")
        sys.exit(1)
    
    # Select portfolio using chosen method
    if args.method == 'markowitz':
        print("\nApplying Markowitz mean-variance optimization...")
        selected_tickers, selection_stats = selector.markowitz_portfolio(
            n_assets=args.candidate_assets
        )
    else:
        print("\nApplying momentum strategy...")
        selected_tickers, selection_stats = selector.momentum_portfolio(
            n_assets=args.candidate_assets
        )
    
    print("\nCandidate Portfolio:")
    print(selection_stats.to_string(index=False))
    
    # Get statistics for quantum optimization
    expected_returns, cov_matrix, asset_names = selector.get_statistics(selected_tickers)
    
    print(f"\nPortfolio statistics computed:")
    print(f"  Assets: {len(asset_names)}")
    print(f"  Expected return range: {expected_returns.min():.2%} to {expected_returns.max():.2%}")
    print(f"  Average volatility: {np.sqrt(np.diag(cov_matrix)).mean():.2%}")
    
    # Save selection results
    selection_stats.to_csv(
        os.path.join(args.output, 'candidate_portfolio.csv'),
        index=False
    )
    
    # =========================================================================
    # SECTION 2: QUANTUM OPTIMIZATION
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 2: QUANTUM OPTIMIZATION")
    print("="*80)
    
    # Initialize optimizer
    optimizer = QuantumPortfolioOptimizer(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        asset_names=asset_names
    )
    
    # Formulate QUBO
    bqm = optimizer.formulate_qubo(
        risk_aversion=args.risk_aversion,
        budget=args.budget,
        penalty_strength=args.penalty_strength
    )
    
    # Solve on quantum annealer
    try:
        quantum_results = optimizer.solve_quantum(
            bqm=bqm,
            num_reads=args.num_reads
        )
    except Exception as e:
        print(f"\nError executing quantum optimization: {e}")
        print("\nPossible causes:")
        print("  1. D-Wave credentials not configured (run: dwave config create)")
        print("  2. Insufficient QPU time available")
        print("  3. Network connectivity issues")
        sys.exit(1)
    
    # Calculate portfolio metrics
    portfolio_metrics = optimizer.calculate_portfolio_metrics(
        quantum_results['selected_assets']
    )
    
    print("\n" + "-"*80)
    print("QUANTUM OPTIMIZATION RESULTS")
    print("-"*80)
    print(f"Best energy: {quantum_results['best_energy']:.4f}")
    print(f"Selected assets: {portfolio_metrics['num_assets']}")
    print(f"\nOptimal portfolio:")
    for ticker in quantum_results['selected_tickers']:
        idx = asset_names.index(ticker)
        print(f"  {ticker}: Return={expected_returns[idx]:.2%}, "
              f"Vol={np.sqrt(cov_matrix[idx, idx]):.2%}")
    
    print(f"\nPortfolio metrics:")
    print(f"  Expected return: {portfolio_metrics['return']:.2%}")
    print(f"  Volatility: {portfolio_metrics['volatility']:.2%}")
    print(f"  Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
    
    # =========================================================================
    # SECTION 3: RESULTS ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("SECTION 3: RESULTS ANALYSIS AND VALIDATION")
    print("="*80)
    
    # Initialize analyzer
    analyzer = PortfolioAnalyzer(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        asset_names=asset_names
    )
    
    # Test solution consistency
    consistency_metrics = analyzer.test_solution_consistency(
        quantum_results['sampleset'],
        top_n=10
    )
    
    # Compare to benchmarks
    benchmark_comparison = analyzer.compare_to_benchmarks(
        quantum_results['selected_assets']
    )
    
    print("\nBenchmark comparison:")
    print(benchmark_comparison.to_string(index=False))
    
    # Generate comprehensive report
    report = analyzer.generate_report(
        selected_indices=quantum_results['selected_assets'],
        metrics=portfolio_metrics,
        consistency=consistency_metrics,
        comparison_df=benchmark_comparison
    )
    
    print(report)
    
    # Save report
    report_path = os.path.join(args.output, 'optimization_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")
    
    # Create visualizations
    viz_path = os.path.join(args.output, 'results_analysis.png')
    analyzer.visualize_results(
        selected_indices=quantum_results['selected_assets'],
        consistency=consistency_metrics,
        comparison_df=benchmark_comparison,
        output_path=viz_path
    )
    
    # Save detailed results
    results_dict = {
        'selected_tickers': quantum_results['selected_tickers'],
        'expected_return': portfolio_metrics['return'],
        'volatility': portfolio_metrics['volatility'],
        'sharpe_ratio': portfolio_metrics['sharpe_ratio'],
        'best_energy': quantum_results['best_energy'],
        'num_unique_solutions': consistency_metrics['total_unique_solutions'],
        'best_solution_frequency': consistency_metrics['best_frequency'],
        'qpu_chip': quantum_results['qpu_info']['chip_id']
    }
    
    import json
    results_path = os.path.join(args.output, 'optimization_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"Results saved: {results_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("EXECUTION COMPLETED")
    print("="*80)
    print(f"\nFinal portfolio: {len(quantum_results['selected_tickers'])} assets")
    print(f"Sharpe ratio: {portfolio_metrics['sharpe_ratio']:.3f}")
    print(f"\nAll results saved to: {args.output}")
    print(f"Execution finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
if __name__ == "__main__":
    import numpy as np  # Required for main execution
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)