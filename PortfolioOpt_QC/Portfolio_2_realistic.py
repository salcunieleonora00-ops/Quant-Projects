"""
Portfolio Optimization System - Advanced Realistic Implementation
=================================================================

This is the main execution script for the IMPROVED portfolio optimization
system (Version 2) using the enhanced sections.

Differences from PortfolioOpt_Quantum_Computer.py (Version 1):
---------------------------------------------------------------
- Capital constraint: $100,000 fixed
- Optimal weight finding (not just selection)
- Quarterly rebalancing simulation
- Full cost modeling (commission + slippage + spread + taxes)
- In-sample/out-of-sample split (80/20)
- Management fees
- Shrinkage estimators for robustness
- Multiple return/covariance estimation methods

Usage:
------
python Portfolio_2_realistic.py --budget 15 --risk_aversion 0.5

Output:
-------
Results saved in: ./results_realistic/
- portfolio_allocation_[timestamp].csv
- statistics_[timestamp].csv
- benchmarks_[timestamp].csv
- portfolio_values_[timestamp].csv
- analysis_dashboard_[timestamp].png
- complete_report_[timestamp].json
- final_report_[timestamp].txt

Author: Portfolio Optimization System
Version: 2.0
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Advanced Portfolio Optimization with Capital Constraint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Fixed capital amount ($)')
    parser.add_argument('--budget', type=int, default=15,
                       help='Number of assets in portfolio')
    parser.add_argument('--candidate_pool', type=int, default=30,
                       help='Candidate pool size')
    parser.add_argument('--risk_aversion', type=float, default=0.5,
                       help='Risk aversion (0-1)')
    parser.add_argument('--use_quantum', action='store_true',
                       help='Use D-Wave quantum annealer')
    parser.add_argument('--num_reads', type=int, default=2000,
                       help='Quantum reads')
    parser.add_argument('--lookback_years', type=int, default=5,
                       help='Years of data')
    parser.add_argument('--universe_size', type=int, default=100,
                       help='Universe size')
    parser.add_argument('--rebalance_threshold', type=float, default=0.05,
                       help='Rebalance threshold')
    parser.add_argument('--commission_bps', type=float, default=5.0,
                       help='Commission (bps)')
    parser.add_argument('--slippage_bps', type=float, default=10.0,
                       help='Slippage (bps)')
    parser.add_argument('--spread_bps', type=float, default=5.0,
                       help='Spread (bps)')
    parser.add_argument('--output', type=str, default='./results_realistic',
                       help='Output directory')
    
    return parser.parse_args()


def print_header(title, char='='):
    """Print formatted header."""
    print(f"\n{char*80}")
    print(title.center(80))
    print(f"{char*80}")


def main():
    """Main execution workflow."""
    args = parse_arguments()
    
    print_header("PORTFOLIO OPTIMIZATION V2 - REALISTIC IMPLEMENTATION")
    print(f"\nExecution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Portfolio Size: {args.budget} stocks")
    print(f"Method: {'Quantum' if args.use_quantum else 'Classical'}")
    
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # =========================================================================
    # SECTION 1: DATA PREPARATION
    # =========================================================================
    print_header("SECTION 1: DATA PREPARATION", '-')
    
    universe = define_universe()[:args.universe_size]
    print(f"\nUniverse: {len(universe)} stocks")
    
    selector = CapitalConstrainedPortfolioSelector(
        universe=universe,
        capital=args.capital,
        lookback_years=args.lookback_years,
        rebalance_frequency='quarterly'
    )
    
    print("Downloading data...")
    try:
        data = selector.download_data()
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    candidate_tickers = data.columns.tolist()[:min(args.candidate_pool, len(data.columns))]
    expected_returns, cov_matrix, asset_names = selector.get_statistics_for_quantum(candidate_tickers)
    
    print(f"\nCandidates: {len(candidate_tickers)}")
    print(f"Expected returns: [{expected_returns.min():.2%}, {expected_returns.max():.2%}]")
    
    # =========================================================================
    # SECTION 2: OPTIMIZATION
    # =========================================================================
    print_header("SECTION 2: OPTIMIZATION", '-')
    
    if args.use_quantum:
        print("\nQuantum optimization...")
        try:
            optimizer = QuantumWeightOptimizer(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                asset_names=asset_names,
                capital=args.capital,
                is_in_sample=True
            )
            
            bqm = optimizer.formulate_qubo_with_weights(
                risk_aversion=args.risk_aversion,
                n_select=args.budget,
                penalty_strength=20.0
            )
            
            results = optimizer.solve_quantum(bqm=bqm, num_reads=args.num_reads)
            selected_indices = results['selected_assets']
            optimal_weights = results['optimal_weights']
            
        except Exception as e:
            print(f"\nQuantum failed: {e}")
            print("Using classical...")
            args.use_quantum = False
    
    if not args.use_quantum:
        print("\nClassical optimization...")
        selected_tickers, optimal_weights, _ = selector.optimize_weights(
            candidate_tickers=candidate_tickers,
            n_assets=args.budget,
            risk_aversion=args.risk_aversion
        )
        selected_indices = [asset_names.index(t) for t in selected_tickers]
        results = {
            'selected_tickers': selected_tickers,
            'optimal_weights': optimal_weights,
            'selected_assets': selected_indices,
            'qpu_info': {'chip_id': 'Classical'}
        }
    
    print(f"\nSelected {len(selected_indices)} assets:")
    for ticker, weight in zip(results['selected_tickers'], optimal_weights):
        print(f"  {ticker}: {weight:.2%} (${args.capital*weight:,.0f})")
    
    # Portfolio metrics
    port_ret = np.dot(optimal_weights, expected_returns[selected_indices])
    port_vol = np.sqrt(np.dot(optimal_weights, 
                              np.dot(cov_matrix[np.ix_(selected_indices, selected_indices)], 
                                    optimal_weights)))
    port_sharpe = port_ret / port_vol if port_vol > 0 else 0
    
    print(f"\nIn-Sample Metrics:")
    print(f"  Return: {port_ret:.2%}")
    print(f"  Volatility: {port_vol:.2%}")
    print(f"  Sharpe: {port_sharpe:.3f}")
    
    # Save allocation
    allocation_df = pd.DataFrame({
        'Ticker': results['selected_tickers'],
        'Weight': optimal_weights,
        'Amount': optimal_weights * args.capital,
        'Expected_Return': expected_returns[selected_indices],
        'Volatility': np.sqrt(np.diag(cov_matrix))[selected_indices]
    })
    allocation_df.to_csv(f"{args.output}/portfolio_allocation_{timestamp}.csv", index=False)
    
    # =========================================================================
    # SECTION 3: BACKTESTING
    # =========================================================================
    print_header("SECTION 3: OUT-OF-SAMPLE BACKTESTING", '-')
    
    analyzer = ComprehensivePortfolioAnalyzer(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        asset_names=asset_names,
        capital=args.capital
    )
    
    analyzer.commission_bps = args.commission_bps
    analyzer.slippage_bps_base = args.slippage_bps
    analyzer.spread_bps = args.spread_bps
    
    print("Running quarterly rebalancing simulation...")
    try:
        sim_results = analyzer.simulate_quarterly_rebalancing(
            selected_indices=selected_indices,
            target_weights=optimal_weights,
            returns_data=selector.out_sample_returns,
            prices_data=selector.out_sample_prices,
            rebalance_threshold=args.rebalance_threshold
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    # Statistics
    statistics = analyzer.calculate_comprehensive_statistics(
        simulation_results=sim_results,
        selected_indices=selected_indices,
        weights=optimal_weights
    )
    
    # Benchmarks
    benchmarks = analyzer.compare_to_benchmarks(
        simulation_results=sim_results,
        selected_indices=selected_indices,
        weights=optimal_weights
    )
    
    print("\nOut-of-Sample Results:")
    print(f"  Final Value: ${sim_results['final_value']:,.0f}")
    print(f"  Total Return: {sim_results['total_return']:.2%}")
    print(f"  Annual Return: {sim_results['annual_return']:.2%}")
    print(f"  Sharpe: {sim_results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {sim_results['max_drawdown']:.2%}")
    print(f"  Rebalances: {sim_results['n_rebalances']}")
    print(f"  Total Costs: ${sim_results['total_costs']:,.0f}")
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print_header("SAVING OUTPUTS", '-')
    
    statistics.to_csv(f"{args.output}/statistics_{timestamp}.csv")
    benchmarks.to_csv(f"{args.output}/benchmarks_{timestamp}.csv", index=False)
    
    pd.DataFrame({
        'Date': sim_results['dates'],
        'Value': sim_results['portfolio_values']
    }).to_csv(f"{args.output}/portfolio_values_{timestamp}.csv", index=False)
    
    # Visualization
    print("\nGenerating dashboard...")
    analyzer.visualize_comprehensive_results(
        simulation_results=sim_results,
        statistics=statistics,
        comparison=benchmarks,
        selected_indices=selected_indices,
        weights=optimal_weights,
        output_path=f"{args.output}/analysis_dashboard_{timestamp}.png"
    )
    
    # JSON report
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': '2.0_realistic',
        'configuration': {
            'capital': args.capital,
            'n_assets': args.budget,
            'risk_aversion': args.risk_aversion,
            'use_quantum': args.use_quantum,
            'rebalance_threshold': args.rebalance_threshold
        },
        'portfolio': {
            'tickers': results['selected_tickers'],
            'weights': optimal_weights.tolist(),
            'in_sample_return': float(port_ret),
            'in_sample_volatility': float(port_vol),
            'in_sample_sharpe': float(port_sharpe)
        },
        'out_of_sample': {
            'final_value': float(sim_results['final_value']),
            'total_return': float(sim_results['total_return']),
            'annual_return': float(sim_results['annual_return']),
            'sharpe_ratio': float(sim_results['sharpe_ratio']),
            'max_drawdown': float(sim_results['max_drawdown']),
            'n_rebalances': sim_results['n_rebalances'],
            'total_costs': float(sim_results['total_costs'])
        },
        'quantum_info': results['qpu_info']
    }
    
    with open(f"{args.output}/complete_report_{timestamp}.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Text report
    text_report = []
    text_report.append("="*80)
    text_report.append("PORTFOLIO OPTIMIZATION V2 - FINAL REPORT")
    text_report.append("="*80)
    text_report.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    text_report.append(f"Capital: ${args.capital:,.0f}")
    text_report.append(f"Method: {'Quantum' if args.use_quantum else 'Classical'}")
    text_report.append(f"\nPORTFOLIO COMPOSITION ({args.budget} assets):")
    for ticker, weight in zip(results['selected_tickers'], optimal_weights):
        text_report.append(f"  {ticker}: {weight:.2%} (${args.capital*weight:,.0f})")
    text_report.append(f"\nOUT-OF-SAMPLE PERFORMANCE:")
    text_report.append(f"  Final Value: ${sim_results['final_value']:,.0f}")
    text_report.append(f"  Net P&L: ${sim_results['final_value']-args.capital:,.0f}")
    text_report.append(f"  Total Return: {sim_results['total_return']:.2%}")
    text_report.append(f"  Annual Return: {sim_results['annual_return']:.2%}")
    text_report.append(f"  Sharpe Ratio: {sim_results['sharpe_ratio']:.3f}")
    text_report.append(f"  Max Drawdown: {sim_results['max_drawdown']:.2%}")
    text_report.append(f"\nCOSTS:")
    text_report.append(f"  Rebalances: {sim_results['n_rebalances']}")
    text_report.append(f"  Transaction Costs: ${sim_results['total_transaction_costs']:,.0f}")
    text_report.append(f"  Taxes: ${sim_results['total_taxes']:,.0f}")
    text_report.append(f"  Total: ${sim_results['total_costs']:,.0f}")
    text_report.append("\n" + "="*80)
    
    with open(f"{args.output}/final_report_{timestamp}.txt", 'w') as f:
        f.write('\n'.join(text_report))
    
    print(f"\nAll outputs saved to: {args.output}/")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("EXECUTION SUMMARY")
    
    print(f"\nFinal Value: ${sim_results['final_value']:,.0f}")
    print(f"Net P&L: ${sim_results['final_value']-args.capital:,.0f}")
    print(f"Sharpe: {sim_results['sharpe_ratio']:.3f}")
    
    best_bench = benchmarks.iloc[1:]['Sharpe'].max()
    if sim_results['sharpe_ratio'] > best_bench:
        improvement = ((sim_results['sharpe_ratio']/best_bench) - 1) * 100
        print(f"\nOUTPERFORMS best benchmark by {improvement:.1f}%")
    else:
        gap = ((best_bench/sim_results['sharpe_ratio']) - 1) * 100
        print(f"\nUnderperforms best benchmark by {gap:.1f}%")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)