"""
Portfolio Optimization System - Advanced Realistic Implementation
=================================================================

FIXED VERSION FOR FAIR COMPARISON WITH V2.5
- Slippage: 5 bps (was 10)
- All cost parameters aligned
- Added explicit cost logging

Version: 2.0_COMPARABLE
Date: 2025-10-30
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
    """Parse command line arguments with ALIGNED defaults for comparison."""
    parser = argparse.ArgumentParser(
        description='Advanced Portfolio Optimization - COMPARABLE VERSION',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Portfolio parameters
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Fixed capital amount ($)')
    parser.add_argument('--budget', type=int, default=15,
                       help='Number of assets in portfolio')
    parser.add_argument('--candidate_pool', type=int, default=28,
                       help='Candidate pool size (aligned with V2.5)')
    
    # Optimization parameters
    parser.add_argument('--risk_aversion', type=float, default=0.5,
                       help='Risk aversion (0-1)')
    parser.add_argument('--use_quantum', action='store_true',
                       help='Use D-Wave quantum annealer')
    parser.add_argument('--num_reads', type=int, default=2000,
                       help='Quantum reads (aligned with V2.5)')
    
    # Data parameters
    parser.add_argument('--lookback_years', type=int, default=5,
                       help='Years of historical data')
    parser.add_argument('--universe_size', type=int, default=80,
                       help='Universe size (aligned with V2.5)')
    
    # Rebalancing parameters
    parser.add_argument('--rebalance_threshold', type=float, default=0.05,
                       help='Rebalance threshold (5%)')
    
    # Cost parameters - ALIGNED WITH V2.5
    parser.add_argument('--commission_bps', type=float, default=5.0,
                       help='Commission (5 bps) - ALIGNED')
    parser.add_argument('--slippage_bps', type=float, default=5.0,
                       help='Slippage (5 bps) - ALIGNED WITH V2.5')
    parser.add_argument('--spread_bps', type=float, default=5.0,
                       help='Spread (5 bps) - ALIGNED')
    
    # Output
    parser.add_argument('--output', type=str, default='./results_realistic_comparable',
                       help='Output directory')
    
    return parser.parse_args()


def print_header(title, char='='):
    """Print formatted header."""
    width = 80
    print(f"\n{char*width}")
    print(title.center(width))
    print(f"{char*width}")


def print_subheader(title, char='-'):
    """Print formatted subheader."""
    width = 80
    print(f"\n{char*width}")
    print(title)
    print(f"{char*width}")


def main():
    """Main execution workflow with aligned parameters."""
    args = parse_arguments()
    
    print_header("PORTFOLIO OPTIMIZATION V2.0 - COMPARABLE VERSION")
    print(f"\nExecution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Portfolio Size: {args.budget} stocks")
    print(f"Candidate Pool: {args.candidate_pool} stocks")
    print(f"Risk Aversion: {args.risk_aversion}")
    print(f"Method: {'Quantum' if args.use_quantum else 'Classical'}")
    
    # Print cost parameters for verification
    print(f"\n{'='*80}")
    print("COST PARAMETERS (aligned with V2.5):")
    print(f"{'='*80}")
    print(f"Commission:  {args.commission_bps} bps")
    print(f"Slippage:    {args.slippage_bps} bps")
    print(f"Spread:      {args.spread_bps} bps")
    print(f"Total/Trade: {args.commission_bps + args.slippage_bps + args.spread_bps} bps (0.15%)")
    print(f"{'='*80}")
    
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # =========================================================================
    # SECTION 1: DATA PREPARATION
    # =========================================================================
    print_header("SECTION 1: DATA PREPARATION", '-')
    
    universe = define_universe()[:args.universe_size]
    print(f"\nStock Universe: {len(universe)} stocks")
    
    selector = CapitalConstrainedPortfolioSelector(
        universe=universe,
        capital=args.capital,
        lookback_years=args.lookback_years,
        rebalance_frequency='quarterly'
    )
    
    print("\nDownloading historical data...")
    try:
        data = selector.download_data()
    except Exception as e:
        print(f"\n ERROR: Failed to download data - {e}")
        return 1
    
    # Get exactly candidate_pool candidates
    candidate_tickers = data.columns.tolist()[:min(args.candidate_pool, len(data.columns))]
    print(f"\nSelected {len(candidate_tickers)} candidates")
    
    expected_returns, cov_matrix, asset_names = selector.get_statistics_for_quantum(candidate_tickers)
    
    print_subheader("IN-SAMPLE STATISTICS")
    print(f"Expected Returns: [{expected_returns.min():.2%}, {expected_returns.max():.2%}]")
    print(f"Average Volatility: {np.sqrt(np.diag(cov_matrix)).mean():.2%}")
    print(f"Average Correlation: {np.corrcoef(cov_matrix)[np.triu_indices_from(np.corrcoef(cov_matrix), k=1)].mean():.3f}")
    
    # =========================================================================
    # SECTION 2: PORTFOLIO OPTIMIZATION
    # =========================================================================
    print_header("SECTION 2: PORTFOLIO OPTIMIZATION", '-')
    
    quantum_used = False
    
    if args.use_quantum:
        print("\nAttempting quantum optimization...")
        try:
            optimizer = QuantumWeightOptimizer(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                asset_names=asset_names,
                capital=args.capital,
                is_in_sample=True
            )
            
            print("Formulating QUBO problem...")
            bqm = optimizer.formulate_qubo_with_weights(
                risk_aversion=args.risk_aversion,
                n_select=args.budget,
                penalty_strength=20.0
            )
            
            print("Submitting to D-Wave QPU...")
            results = optimizer.solve_quantum(bqm=bqm, num_reads=args.num_reads)
            selected_indices = results['selected_assets']
            optimal_weights = results['optimal_weights']
            quantum_used = True
            
            print(f"\n✓ QUANTUM OPTIMIZATION SUCCESSFUL")
            if 'qpu_info' in results and 'chip_id' in results['qpu_info']:
                print(f"  QPU Chip: {results['qpu_info']['chip_id']}")
            
        except Exception as e:
            print(f"\n  Quantum optimization failed: {e}")
            print("Falling back to classical optimization...")
            args.use_quantum = False
    
    # Classical optimization (fallback or direct)
    if not args.use_quantum or not quantum_used:
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
            'qpu_info': {'chip_id': 'Classical_SLSQP'}
        }
        print(f"✓ Classical optimization completed")
    
    # Display portfolio
    print_subheader("OPTIMIZED PORTFOLIO")
    print(f"Assets selected: {len(selected_indices)}")
    print(f"Optimization method: {'QUANTUM' if quantum_used else 'CLASSICAL'}")
    print(f"\n{'Ticker':<8} {'Weight':>10} {'Amount':>14} {'Exp.Return':>12} {'Volatility':>12}")
    print("-" * 80)
    
    for ticker, weight in zip(results['selected_tickers'], optimal_weights):
        idx = results['selected_assets'][results['selected_tickers'].index(ticker)]
        amount = args.capital * weight
        ret = expected_returns[idx]
        vol = np.sqrt(cov_matrix[idx, idx])
        print(f"{ticker:<8} {weight:>9.2%} ${amount:>13,.0f} {ret:>11.2%} {vol:>11.2%}")
    
    print("-" * 80)
    print(f"{'TOTAL':<8} {optimal_weights.sum():>9.2%} ${args.capital:>13,.0f}")
    
    # Calculate portfolio metrics
    port_ret = np.dot(optimal_weights, expected_returns[selected_indices])
    port_cov = cov_matrix[np.ix_(selected_indices, selected_indices)]
    port_vol = np.sqrt(np.dot(optimal_weights, np.dot(port_cov, optimal_weights)))
    port_sharpe = port_ret / port_vol if port_vol > 0 else 0
    hhi = np.sum(optimal_weights ** 2)
    effective_n = 1 / hhi if hhi > 0 else 0
    
    print_subheader("IN-SAMPLE METRICS")
    print(f"Expected Return:     {port_ret:.2%}")
    print(f"Expected Volatility: {port_vol:.2%}")
    print(f"Expected Sharpe:     {port_sharpe:.3f}")
    print(f"HHI:                 {hhi:.4f}")
    print(f"Effective N Assets:  {effective_n:.2f}")
    
    # Save allocation
    allocation_df = pd.DataFrame({
        'Ticker': results['selected_tickers'],
        'Weight': optimal_weights,
        'Amount': optimal_weights * args.capital,
        'Expected_Return': expected_returns[selected_indices],
        'Volatility': np.sqrt(np.diag(cov_matrix))[selected_indices]
    })
    alloc_path = f"{args.output}/portfolio_allocation_{timestamp}.csv"
    allocation_df.to_csv(alloc_path, index=False)
    print(f"\nPortfolio allocation saved: {alloc_path}")
    
    # =========================================================================
    # SECTION 3: OUT-OF-SAMPLE BACKTESTING
    # =========================================================================
    print_header("SECTION 3: OUT-OF-SAMPLE BACKTESTING", '-')
    
    analyzer = ComprehensivePortfolioAnalyzer(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        asset_names=asset_names,
        capital=args.capital
    )
    
    # Set cost parameters (ALIGNED)
    analyzer.commission_bps = args.commission_bps
    analyzer.slippage_bps_base = args.slippage_bps
    analyzer.spread_bps = args.spread_bps
    
    print(f"\nCost Configuration:")
    print(f"  Commission: {analyzer.commission_bps} bps")
    print(f"  Slippage:   {analyzer.slippage_bps_base} bps")
    print(f"  Spread:     {analyzer.spread_bps} bps")
    print(f"  Total:      {analyzer.commission_bps + analyzer.slippage_bps_base + analyzer.spread_bps} bps per trade")
    
    print("\nRunning quarterly rebalancing simulation...")
    
    try:
        sim_results = analyzer.simulate_quarterly_rebalancing(
            selected_indices=selected_indices,
            target_weights=optimal_weights,
            returns_data=selector.out_sample_returns,
            prices_data=selector.out_sample_prices,
            rebalance_threshold=args.rebalance_threshold
        )
    except Exception as e:
        print(f"\n ERROR: Simulation failed - {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Calculate statistics
    statistics = analyzer.calculate_comprehensive_statistics(
        simulation_results=sim_results,
        selected_indices=selected_indices,
        weights=optimal_weights
    )
    
    # Benchmark comparison
    benchmarks = analyzer.compare_to_benchmarks(
        simulation_results=sim_results,
        selected_indices=selected_indices,
        weights=optimal_weights
    )
    
    print_subheader("OUT-OF-SAMPLE PERFORMANCE")
    print(f"Final Value:         ${sim_results['final_value']:,.0f}")
    print(f"Net P&L:             ${sim_results['final_value']-args.capital:,.0f}")
    print(f"Total Return:        {sim_results['total_return']:.2%}")
    print(f"Annual Return:       {sim_results['annual_return']:.2%}")
    print(f"Annual Volatility:   {sim_results['annual_volatility']:.2%}")
    print(f"Sharpe Ratio:        {sim_results['sharpe_ratio']:.3f}")
    print(f"Sortino Ratio:       {sim_results['sortino_ratio']:.3f}")
    print(f"Calmar Ratio:        {sim_results['calmar_ratio']:.3f}")
    print(f"Max Drawdown:        {sim_results['max_drawdown']:.2%}")
    print(f"Win Rate:            {sim_results['win_rate']:.1%}")
    print(f"\nRebalances:          {sim_results['n_rebalances']}")
    print(f"Transaction Costs:   ${sim_results['total_transaction_costs']:,.0f}")
    print(f"Taxes:               ${sim_results['total_taxes']:,.0f}")
    print(f"Total Costs:         ${sim_results['total_costs']:,.0f}")
    print(f"Cost Impact:         {sim_results['total_costs']/args.capital:.2%} of capital")
    
    print_subheader("BENCHMARK COMPARISON")
    print(benchmarks.to_string(index=False))
    
    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================
    print_header("SAVING OUTPUTS", '-')
    
    stats_path = f"{args.output}/statistics_{timestamp}.csv"
    statistics.to_csv(stats_path)
    print(f"\nStatistics saved: {stats_path}")
    
    bench_path = f"{args.output}/benchmarks_{timestamp}.csv"
    benchmarks.to_csv(bench_path, index=False)
    print(f"Benchmarks saved: {bench_path}")
    
    timeseries_df = pd.DataFrame({
        'Date': sim_results['dates'],
        'Value': sim_results['portfolio_values']
    })
    ts_path = f"{args.output}/portfolio_values_{timestamp}.csv"
    timeseries_df.to_csv(ts_path, index=False)
    print(f"Time series saved: {ts_path}")
    
    print("\nGenerating comprehensive dashboard...")
    viz_path = f"{args.output}/analysis_dashboard_{timestamp}.png"
    try:
        analyzer.visualize_comprehensive_results(
            simulation_results=sim_results,
            statistics=statistics,
            comparison=benchmarks,
            selected_indices=selected_indices,
            weights=optimal_weights,
            output_path=viz_path
        )
        print(f"✓ Visualization saved: {viz_path}")
    except Exception as e:
        print(f"  Visualization failed: {e}")
    
    # Generate reports
    print("\nGenerating reports...")
    
    # JSON report
    report = {
        'timestamp': datetime.now().isoformat(),
        'version': '2.0_COMPARABLE',
        'configuration': {
            'capital': args.capital,
            'n_assets': args.budget,
            'candidate_pool': args.candidate_pool,
            'risk_aversion': args.risk_aversion,
            'use_quantum': quantum_used,
            'num_reads': args.num_reads if quantum_used else None,
            'rebalance_threshold': args.rebalance_threshold,
            'commission_bps': args.commission_bps,
            'slippage_bps': args.slippage_bps,
            'spread_bps': args.spread_bps,
            'total_cost_bps': args.commission_bps + args.slippage_bps + args.spread_bps
        },
        'portfolio': {
            'tickers': results['selected_tickers'],
            'weights': optimal_weights.tolist(),
            'in_sample_return': float(port_ret),
            'in_sample_volatility': float(port_vol),
            'in_sample_sharpe': float(port_sharpe),
            'hhi': float(hhi),
            'effective_n': float(effective_n)
        },
        'out_of_sample': {
            'final_value': float(sim_results['final_value']),
            'total_return': float(sim_results['total_return']),
            'annual_return': float(sim_results['annual_return']),
            'sharpe_ratio': float(sim_results['sharpe_ratio']),
            'sortino_ratio': float(sim_results['sortino_ratio']),
            'calmar_ratio': float(sim_results['calmar_ratio']),
            'max_drawdown': float(sim_results['max_drawdown']),
            'win_rate': float(sim_results['win_rate']),
            'n_rebalances': sim_results['n_rebalances'],
            'total_costs': float(sim_results['total_costs'])
        },
        'quantum_info': results['qpu_info']
    }
    
    json_path = f"{args.output}/complete_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved: {json_path}")
    
    # Text report
    text_report = []
    text_report.append("="*80)
    text_report.append("PORTFOLIO OPTIMIZATION V2.0 - COMPARABLE VERSION")
    text_report.append("="*80)
    text_report.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    text_report.append(f"Version: 2.0_COMPARABLE")
    text_report.append(f"Method: {'Quantum' if quantum_used else 'Classical'}")
    
    text_report.append(f"\n{'-'*80}")
    text_report.append("CONFIGURATION:")
    text_report.append(f"{'-'*80}")
    text_report.append(f"Capital:         ${args.capital:,.0f}")
    text_report.append(f"Portfolio Size:  {args.budget} assets")
    text_report.append(f"Candidate Pool:  {args.candidate_pool} assets")
    text_report.append(f"Risk Aversion:   {args.risk_aversion}")
    
    text_report.append(f"\n{'-'*80}")
    text_report.append("COST PARAMETERS (aligned with V2.5):")
    text_report.append(f"{'-'*80}")
    text_report.append(f"Commission:  {args.commission_bps} bps")
    text_report.append(f"Slippage:    {args.slippage_bps} bps")
    text_report.append(f"Spread:      {args.spread_bps} bps")
    text_report.append(f"Total/Trade: {args.commission_bps + args.slippage_bps + args.spread_bps} bps")
    
    text_report.append(f"\n{'-'*80}")
    text_report.append("PORTFOLIO COMPOSITION:")
    text_report.append(f"{'-'*80}")
    for ticker, weight in zip(results['selected_tickers'], optimal_weights):
        text_report.append(f"  {ticker}: {weight:.2%} (${args.capital*weight:,.0f})")
    
    text_report.append(f"\n{'-'*80}")
    text_report.append("OUT-OF-SAMPLE PERFORMANCE:")
    text_report.append(f"{'-'*80}")
    text_report.append(f"Final Value:    ${sim_results['final_value']:,.0f}")
    text_report.append(f"Net P&L:        ${sim_results['final_value']-args.capital:,.0f}")
    text_report.append(f"Total Return:   {sim_results['total_return']:.2%}")
    text_report.append(f"Annual Return:  {sim_results['annual_return']:.2%}")
    text_report.append(f"Sharpe Ratio:   {sim_results['sharpe_ratio']:.3f}")
    text_report.append(f"Sortino Ratio:  {sim_results['sortino_ratio']:.3f}")
    text_report.append(f"Max Drawdown:   {sim_results['max_drawdown']:.2%}")
    
    text_report.append(f"\n{'-'*80}")
    text_report.append("COSTS:")
    text_report.append(f"{'-'*80}")
    text_report.append(f"Rebalances:      {sim_results['n_rebalances']}")
    text_report.append(f"Transaction:     ${sim_results['total_transaction_costs']:,.0f}")
    text_report.append(f"Taxes:           ${sim_results['total_taxes']:,.0f}")
    text_report.append(f"Total:           ${sim_results['total_costs']:,.0f}")
    text_report.append(f"Cost Impact:     {sim_results['total_costs']/args.capital:.2%} of capital")
    
    text_report.append("\n" + "="*80)
    
    text_path = f"{args.output}/final_report_{timestamp}.txt"
    with open(text_path, 'w') as f:
        f.write('\n'.join(text_report))
    print(f"Text report saved: {text_path}")
    
    print(f"\nAll outputs saved to: {args.output}/")
    
    # =========================================================================
    # EXECUTION SUMMARY
    # =========================================================================
    print_header("EXECUTION SUMMARY")
    
    print(f"\nOptimization Method:     {'QUANTUM' if quantum_used else 'CLASSICAL'}")
    if quantum_used and 'qpu_info' in results and 'chip_id' in results['qpu_info']:
        print(f"QPU Chip:                {results['qpu_info']['chip_id']}")
    
    print(f"\nFinal Portfolio Value:   ${sim_results['final_value']:,.0f}")
    print(f"Net Profit/Loss:         ${sim_results['final_value']-args.capital:,.0f}")
    print(f"Total Return:            {sim_results['total_return']:.2%}")
    print(f"Annual Return:           {sim_results['annual_return']:.2%}")
    print(f"Sharpe Ratio:            {sim_results['sharpe_ratio']:.3f}")
    print(f"Max Drawdown:            {sim_results['max_drawdown']:.2%}")
    
    print(f"\nTotal Costs:             ${sim_results['total_costs']:,.0f}")
    print(f"Cost Impact:             {sim_results['total_costs']/args.capital:.2%} of capital")
    
    # Compare to benchmarks
    best_bench_sharpe = benchmarks.iloc[1:]['Sharpe'].max()
    if sim_results['sharpe_ratio'] > best_bench_sharpe:
        improvement = ((sim_results['sharpe_ratio']/best_bench_sharpe) - 1) * 100
        print(f"\nPerformance:             Outperforms best benchmark by {improvement:.1f}%")
    else:
        gap = ((best_bench_sharpe/sim_results['sharpe_ratio']) - 1) * 100
        print(f"\nPerformance:             Underperforms best benchmark by {gap:.1f}%")
    
    print(f"\nCompleted:               {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)