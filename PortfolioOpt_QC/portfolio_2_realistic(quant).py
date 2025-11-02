"""
Portfolio Optimization System - FIXED & OPTIMIZED VERSION
==========================================================

Fixes:
- Reduced quantum parameters to prevent timeout (2000 reads, 20us)
- Better error handling and retry logic
- Adaptive penalty strength with validation
- Fallback to classical if quantum fails
- Optimized QUBO formulation

Version: 2.5_FIXED
Date: 2025-10-29
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

from section1_quantum_pure import (
    QuantumPortfolioDataPreparator,
    define_universe
)
from section2_quantum_pure import (
    PureQuantumWeightOptimizer
)
from section3_quantum_pure import (
    ComprehensivePortfolioAnalyzer
)


def parse_arguments():
    """Parse command line arguments with STABLE defaults."""
    parser = argparse.ArgumentParser(
        description='FIXED Portfolio Optimization with Quantum Computing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Portfolio parameters
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Fixed capital amount ($)')
    parser.add_argument('--budget', type=int, default=15,
                       help='Number of assets')
    parser.add_argument('--candidate_pool', type=int, default=30,
                       help='Candidate pool (auto-reduced to 28 if needed)')
    
    # Optimization parameters - FIXED VALUES
    parser.add_argument('--risk_aversion', type=float, default=0.5,
                       help='Risk aversion (0.5 = balanced)')
    parser.add_argument('--no_quantum', action='store_true', default=False,
                       help='Disable quantum and use classical')
    parser.add_argument('--num_reads', type=int, default=2000,
                       help='Quantum reads (2000 for stability)')
    parser.add_argument('--annealing_time', type=int, default=20,
                       help='Annealing time (20us for reliability)')
    parser.add_argument('--penalty_strength', type=float, default=50.0,
                       help='QUBO penalty (50 = moderate)')
    parser.add_argument('--chain_strength', type=float, default=2.0,
                       help='Chain strength (2.0 to reduce breaks)')
    
    # Data parameters
    parser.add_argument('--lookback_years', type=int, default=5,
                       help='Years of data')
    parser.add_argument('--universe_size', type=int, default=80,
                       help='Universe size')
    
    # Rebalancing parameters
    parser.add_argument('--rebalance_threshold', type=float, default=0.05,
                       help='Rebalance at 5% drift')
    
    # Cost parameters
    parser.add_argument('--commission_bps', type=float, default=5.0,
                       help='Commission (5bps)')
    parser.add_argument('--slippage_bps', type=float, default=5.0,
                       help='Slippage (5bps)')
    parser.add_argument('--spread_bps', type=float, default=5.0,
                       help='Spread (5bps)')
    
    # Output
    parser.add_argument('--output', type=str, default='./results_quantum_fixed',
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


def validate_quantum_solution(results, target_n_assets, tolerance=3):
    """
    Validate quantum solution quality with relaxed tolerance.
    
    Parameters
    ----------
    results : dict
        Quantum optimization results
    target_n_assets : int
        Target number of assets
    tolerance : int
        Allowed deviation from target (relaxed to 3)
    
    Returns
    -------
    bool, str
        True/False and reason message
    """
    n_selected = results['n_selected']
    
    # Check if close to target
    if abs(n_selected - target_n_assets) > tolerance:
        return False, f"Selected {n_selected} assets, target was {target_n_assets} (±{tolerance})"
    
    # Check if weights are reasonable
    weights = results['optimal_weights']
    if len(weights) == 0:
        return False, "No weights found"
    
    # Check weight constraints (relaxed)
    if weights.max() > 0.50:
        return False, f"Max weight {weights.max():.2%} exceeds 50%"
    
    if weights.min() < 0.005:
        return False, f"Min weight {weights.min():.2%} below 0.5%"
    
    # Check diversification (relaxed)
    hhi = np.sum(weights ** 2)
    effective_n = 1 / hhi
    if effective_n < target_n_assets * 0.5:  # More relaxed
        return False, f"Low diversification (Effective N: {effective_n:.1f})"
    
    msg = f"✓ VALID: {n_selected} assets, HHI={hhi:.4f}, Effective N={effective_n:.1f}"
    return True, msg


def optimize_quantum_with_retry(preparator, candidate_tickers, expected_returns,
                                 cov_matrix, asset_names, args, max_attempts=3):
    """
    Quantum optimization with adaptive retry logic and auto-reduction.
    
    Automatically reduces problem size if embedding fails.
    Tries multiple penalty values and parameters if initial attempt fails.
    
    Parameters
    ----------
    preparator : QuantumPortfolioDataPreparator
        Data preparator
    candidate_tickers : list
        Candidate tickers
    expected_returns : np.ndarray
        Expected returns
    cov_matrix : np.ndarray
        Covariance matrix
    asset_names : list
        Asset names
    args : Namespace
        Arguments
    max_attempts : int
        Maximum optimization attempts
    
    Returns
    -------
    dict or None
        Optimization results or None if all attempts fail
    """
    # CRITICAL: Auto-reduce problem size if too large
    n_candidates = len(candidate_tickers)
    embedding_failed = False
    
    # Check if problem is too large for embedding
    if n_candidates > 30:
        print(f"\n{'!'*80}")
        print(f"WARNING: {n_candidates} candidates may be too large for quantum embedding")
        print(f"Auto-reducing to 28 candidates for stability...")
        print(f"{'!'*80}")
        
        # Select top 28 by Sharpe ratio
        candidate_sharpes = expected_returns / (np.sqrt(np.diag(cov_matrix)) + 1e-8)
        top_indices = np.argsort(candidate_sharpes)[-28:]
        
        # Update all arrays
        candidate_tickers = [candidate_tickers[i] for i in top_indices]
        expected_returns = expected_returns[top_indices]
        cov_matrix = cov_matrix[np.ix_(top_indices, top_indices)]
        
        print(f"✓ Reduced to {len(candidate_tickers)} candidates")
        print(f"  New Sharpe range: [{candidate_sharpes[top_indices].min():.2f}, {candidate_sharpes[top_indices].max():.2f}]")
    
    # Try different penalty strengths
    penalty_values = [
        args.penalty_strength,           # Base: 50
        args.penalty_strength * 1.5,     # Try: 75
        args.penalty_strength * 2.0      # Try: 100
    ]
    
    # Try different num_reads if needed
    reads_values = [
        args.num_reads,     # Base: 2000
        1500,               # Reduced
        1000                # Minimal
    ]
    
    # Try reducing candidates further if embedding keeps failing
    candidate_reduction = [
        len(candidate_tickers),  # Original (28 or less)
        24,                      # Reduced
        20                       # Minimal
    ]
    
    for attempt in range(max_attempts):
        penalty = penalty_values[min(attempt, len(penalty_values)-1)]
        num_reads = reads_values[min(attempt, len(reads_values)-1)]
        target_candidates = candidate_reduction[min(attempt, len(candidate_reduction)-1)]
        
        # Further reduce candidates if needed on retry
        if attempt > 0 and len(candidate_tickers) > target_candidates:
            print(f"\n{'!'*80}")
            print(f"RETRY {attempt+1}: Reducing candidates from {len(candidate_tickers)} to {target_candidates}")
            print(f"{'!'*80}")
            
            candidate_sharpes = expected_returns / (np.sqrt(np.diag(cov_matrix)) + 1e-8)
            top_indices = np.argsort(candidate_sharpes)[-target_candidates:]
            
            candidate_tickers = [candidate_tickers[i] for i in top_indices]
            expected_returns = expected_returns[top_indices]
            cov_matrix = cov_matrix[np.ix_(top_indices, top_indices)]
            
            print(f"✓ Now using {len(candidate_tickers)} candidates")
        
        print(f"\n{'='*80}")
        print(f"QUANTUM OPTIMIZATION ATTEMPT {attempt+1}/{max_attempts}")
        print(f"{'='*80}")
        print(f"Candidates: {len(candidate_tickers)}")
        print(f"Target assets: {args.budget}")
        print(f"Penalty strength: {penalty:.1f}")
        print(f"Num reads: {num_reads}")
        print(f"Annealing time: {args.annealing_time}us")
        print(f"Chain strength: {args.chain_strength}")
        print(f"Variables: {len(candidate_tickers) * 5} (candidates × weight levels)")
        
        try:
            # Initialize optimizer with current reduced set
            optimizer = PureQuantumWeightOptimizer(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                asset_names=candidate_tickers,  # Use reduced tickers
                capital=args.capital,
                is_in_sample=True
            )
            
            print("\nFormulating QUBO problem...")
            bqm = optimizer.formulate_qubo(
                risk_aversion=args.risk_aversion,
                n_select=min(args.budget, len(candidate_tickers)),  # Adjust if needed
                penalty_strength=penalty
            )
            
            print("Submitting to D-Wave QPU...")
            results = optimizer.solve_quantum(
                bqm=bqm,
                num_reads=num_reads,
                annealing_time=args.annealing_time,
                chain_strength=args.chain_strength
            )
            
            # Validate solution
            is_valid, msg = validate_quantum_solution(results, args.budget, tolerance=3)
            print(f"\nValidation: {msg}")
            
            if is_valid:
                print(f"\n{'='*80}")
                print(f"SUCCESS ON ATTEMPT {attempt+1}")
                print(f"{'='*80}")
                
                # Map back to original asset_names indices
                selected_tickers = results['selected_tickers']
                selected_indices = [asset_names.index(t) for t in selected_tickers]
                results['selected_assets'] = selected_indices
                
                return results
            else:
                print(f"\n  Attempt {attempt+1} failed validation")
                if attempt < max_attempts - 1:
                    print("Retrying with adjusted parameters...")
                
        except Exception as e:
            print(f"\n Attempt {attempt+1} failed with error:")
            print(f"   {type(e).__name__}: {str(e)}")
            
            # Print more details for debugging
            error_str = str(e).lower()
            if "timeout" in error_str or "time" in error_str:
                print("   → Possible timeout. Reducing reads for next attempt.")
            elif "embedding" in error_str or "no embedding found" in error_str:
                print("   → Embedding failed. Problem too large.")
                print("   → Will reduce candidates and try again...")
                embedding_failed = True
            elif "chain" in error_str:
                print("   → Chain breaks detected. Will adjust parameters.")
            
            if attempt < max_attempts - 1:
                print("   Retrying...")
            continue
    
    print(f"\n{'='*80}")
    print("ALL QUANTUM ATTEMPTS FAILED")
    print(f"{'='*80}")
    return None


def optimize_classical_fallback(candidate_tickers, expected_returns,
                                cov_matrix, asset_names, args):
    """
    Enhanced classical optimization fallback.
    
    Uses scipy optimization with proper constraints.
    
    Parameters
    ----------
    candidate_tickers : list
        Candidate tickers
    expected_returns : np.ndarray
        Expected returns
    cov_matrix : np.ndarray
        Covariance matrix
    asset_names : list
        Asset names
    args : Namespace
        Arguments
    
    Returns
    -------
    dict
        Optimization results
    """
    from scipy.optimize import minimize
    
    print(f"\n{'='*80}")
    print("CLASSICAL OPTIMIZATION (ENHANCED)")
    print(f"{'='*80}")
    
    # Regularize covariance matrix
    epsilon = 1e-6
    cov_reg = cov_matrix + epsilon * np.eye(len(cov_matrix))
    
    n_candidates = len(candidate_tickers)
    
    # Step 1: Find maximum Sharpe portfolio on all candidates
    print("\nStep 1: Finding maximum Sharpe portfolio...")
    
    def neg_sharpe(w):
        ret = np.dot(w, expected_returns)
        vol = np.sqrt(np.dot(w, np.dot(cov_reg, w)))
        return -ret / vol if vol > 1e-8 else 1e10
    
    constraints_all = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds_all = tuple((0, 1) for _ in range(n_candidates))
    x0_all = np.ones(n_candidates) / n_candidates
    
    result_all = minimize(
        neg_sharpe,
        x0_all,
        method='SLSQP',
        bounds=bounds_all,
        constraints=constraints_all,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    weights_all = result_all.x
    
    # Select top N assets by weight
    top_indices = np.argsort(weights_all)[-args.budget:]
    selected_tickers = [candidate_tickers[i] for i in top_indices]
    selected_indices = [asset_names.index(t) for t in selected_tickers]
    
    print(f"Selected {len(selected_tickers)} assets")
    
    # Step 2: Optimize weights on selected assets
    print("Step 2: Optimizing weights on selected assets...")
    
    selected_returns = expected_returns[top_indices]
    selected_cov = cov_reg[np.ix_(top_indices, top_indices)]
    
    def objective(w):
        ret = np.dot(w, selected_returns)
        var = np.dot(w, np.dot(selected_cov, w))
        return args.risk_aversion * var - (1 - args.risk_aversion) * ret
    
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = tuple((0.01, 0.40) for _ in range(args.budget))  # 1% to 40%
    x0 = np.ones(args.budget) / args.budget
    
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    optimal_weights = result.x
    
    # Normalize weights
    optimal_weights = optimal_weights / optimal_weights.sum()
    
    print(f"✓ Optimization completed")
    print(f"  Weight range: [{optimal_weights.min():.2%}, {optimal_weights.max():.2%}]")
    print(f"  Sum of weights: {optimal_weights.sum():.4f}")
    
    return {
        'selected_tickers': selected_tickers,
        'selected_assets': selected_indices,
        'optimal_weights': optimal_weights,
        'n_selected': len(selected_tickers),
        'qpu_info': {
            'chip_id': 'Classical_SLSQP',
            'topology': 'CPU',
            'num_qubits': 0
        }
    }


def main():
    """Main execution with fixed quantum parameters."""
    args = parse_arguments()
    
    use_quantum = not args.no_quantum
    
    print_header("PORTFOLIO OPTIMIZATION V2.5 - QUANTUM FIXED")
    print(f"\nExecution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Portfolio Size: {args.budget} stocks")
    print(f"Risk Aversion: {args.risk_aversion}")
    print(f"Method: {'Quantum (2000 reads, 20us)' if use_quantum else 'Classical'}")
    
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # =========================================================================
    # SECTION 1: DATA PREPARATION
    # =========================================================================
    print_header("SECTION 1: DATA PREPARATION", '-')
    
    universe = define_universe()[:args.universe_size]
    print(f"\nStock Universe: {len(universe)} stocks")
    
    preparator = QuantumPortfolioDataPreparator(
        universe=universe,
        capital=args.capital,
        lookback_years=args.lookback_years,
        rebalance_frequency='quarterly'
    )
    
    print("\nDownloading historical data...")
    try:
        data = preparator.download_data()
    except Exception as e:
        print(f"\n ERROR: Failed to download data - {e}")
        return 1
    
    # Pre-screen candidates by Sharpe ratio
    print(f"\nPre-screening top {args.candidate_pool} candidates by Sharpe ratio...")
    all_returns = preparator.in_sample_returns
    candidate_sharpes = (all_returns.mean() * 252) / (all_returns.std() * np.sqrt(252))
    top_sharpe_tickers = candidate_sharpes.nlargest(args.candidate_pool).index.tolist()
    
    print(f"Selected {len(top_sharpe_tickers)} candidates")
    print(f"Sharpe range: [{candidate_sharpes[top_sharpe_tickers].min():.2f}, "
          f"{candidate_sharpes[top_sharpe_tickers].max():.2f}]")
    
    # Get optimization inputs
    expected_returns, cov_matrix, asset_names = preparator.get_quantum_optimization_inputs(
        tickers=top_sharpe_tickers,
        return_method='shrinkage',
        cov_method='ledoit_wolf'
    )
    
    print_subheader("IN-SAMPLE STATISTICS")
    print(f"Expected Returns: [{expected_returns.min():.2%}, {expected_returns.max():.2%}]")
    print(f"Average Volatility: {np.sqrt(np.diag(cov_matrix)).mean():.2%}")
    print(f"Average Correlation: {np.corrcoef(cov_matrix)[np.triu_indices_from(np.corrcoef(cov_matrix), k=1)].mean():.3f}")
    
    # =========================================================================
    # SECTION 2: PORTFOLIO OPTIMIZATION
    # =========================================================================
    print_header("SECTION 2: PORTFOLIO OPTIMIZATION", '-')
    
    quantum_used = False
    results = None
    
    if use_quantum:
        print("\nAttempting quantum optimization with retry logic...")
        results = optimize_quantum_with_retry(
            preparator, top_sharpe_tickers, expected_returns,
            cov_matrix, asset_names, args, max_attempts=3
        )
        
        if results is not None:
            quantum_used = True
            selected_indices = results['selected_assets']
            optimal_weights = results['optimal_weights']
            print(f"\n✓ QUANTUM OPTIMIZATION SUCCESSFUL")
            print(f"  QPU Chip: {results['qpu_info']['chip_id']}")
            if 'qpu_time' in results['qpu_info']:
                print(f"  QPU Time: {results['qpu_info']['qpu_time']:.2f} μs")
        else:
            print("\n⚠️  All quantum attempts failed")
            print("Falling back to classical optimization...")
    
    # Classical fallback
    if not quantum_used or results is None:
        results = optimize_classical_fallback(
            top_sharpe_tickers, expected_returns,
            cov_matrix, asset_names, args
        )
        selected_indices = results['selected_assets']
        optimal_weights = results['optimal_weights']
        quantum_used = False
    
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
    
    # Set cost parameters
    analyzer.commission_bps = args.commission_bps
    analyzer.slippage_bps_base = args.slippage_bps
    analyzer.spread_bps = args.spread_bps
    
    print("\nRunning quarterly rebalancing simulation...")
    
    try:
        sim_results = analyzer.simulate_quarterly_rebalancing(
            selected_indices=selected_indices,
            target_weights=optimal_weights,
            returns_data=preparator.out_sample_returns,
            prices_data=preparator.out_sample_prices,
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
        'version': '2.5_FIXED',
        'configuration': {
            'capital': args.capital,
            'n_assets': args.budget,
            'risk_aversion': args.risk_aversion,
            'use_quantum': quantum_used,
            'num_reads': args.num_reads if quantum_used else None,
            'annealing_time': args.annealing_time if quantum_used else None,
            'penalty_strength': args.penalty_strength if quantum_used else None,
            'chain_strength': args.chain_strength if quantum_used else None
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
            'max_drawdown': float(sim_results['max_drawdown'])
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
    text_report.append("PORTFOLIO OPTIMIZATION - FINAL REPORT")
    text_report.append("="*80)
    text_report.append(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    text_report.append(f"Version: 2.5 FIXED")
    text_report.append(f"Method: {'Quantum' if quantum_used else 'Classical'}")
    
    text_report.append(f"\n{'-'*80}")
    text_report.append("OUT-OF-SAMPLE PERFORMANCE:")
    text_report.append(f"{'-'*80}")
    text_report.append(f"Final Value:    ${sim_results['final_value']:,.0f}")
    text_report.append(f"Total Return:   {sim_results['total_return']:.2%}")
    text_report.append(f"Annual Return:  {sim_results['annual_return']:.2%}")
    text_report.append(f"Sharpe Ratio:   {sim_results['sharpe_ratio']:.3f}")
    text_report.append(f"Max Drawdown:   {sim_results['max_drawdown']:.2%}")
    
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
    if quantum_used:
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