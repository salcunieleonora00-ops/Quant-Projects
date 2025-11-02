"""
Section 2: Pure Quantum Portfolio Optimization
==============================================

Key Features:
- QUBO formulation for BOTH selection and weight optimization
- Capital constraint: $100,000
- 15-stock portfolio with optimal weights
- In-sample data only (no lookahead bias)
- D-Wave quantum annealer (NO classical fallback)
- Improved numerical stability
- Compatible with latest dimod/dwave versions

Mathematical Formulation:
------------------------
minimize: f(w) = lambda * risk(w) - (1-lambda) * return(w)

where:
- w_i: continuous weights (converted to binary via discretization)
- risk(w) = w^T * Sigma * w
- return(w) = mu^T * w

Constraints:
- sum(w_i) = 1  (budget constraint)
- Exactly N assets selected
- w_i >= w_min for selected assets

References:
- Markowitz (1952): Portfolio Selection
- Lucas (2014): Ising formulations of NP problems
- Venturelli & Kondratyev (2019): Reverse Quantum Annealing

Author: Portfolio Optimization System
Version: 4.0 - Pure Quantum
"""

import numpy as np
from typing import List, Dict, Tuple
from dimod import BinaryQuadraticModel, BINARY
from dwave.system import DWaveSampler, EmbeddingComposite
import warnings
warnings.filterwarnings('ignore')


class PureQuantumWeightOptimizer:
    """
    Pure quantum optimizer - NO classical fallback.
    Finds both asset selection AND optimal weights using D-Wave only.
    """
    
    def __init__(self,
                 expected_returns: np.ndarray,
                 cov_matrix: np.ndarray,
                 asset_names: List[str],
                 capital: float = 100000.0,
                 is_in_sample: bool = True):
        """
        Initialize pure quantum weight optimizer.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected annual returns (in-sample)
        cov_matrix : np.ndarray
            Annual covariance matrix (in-sample)
        asset_names : List[str]
            Asset ticker symbols
        capital : float
            Fixed capital amount
        is_in_sample : bool
            Flag to ensure in-sample data only
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.asset_names = asset_names
        self.capital = capital
        self.n_assets = len(expected_returns)
        self.is_in_sample = is_in_sample
        
        if not is_in_sample:
            raise ValueError("❌ Quantum optimization MUST use in-sample data only!")
        
        # Weight discretization parameters
        self.n_weight_levels = 5  # Discretize weights into 5 levels
        self.min_weight = 0.01    # Minimum 1%
        self.max_weight = 0.40    # Maximum 40%
        
        print(f"\n{'='*80}")
        print(f"PURE QUANTUM OPTIMIZER INITIALIZED")
        print(f"{'='*80}")
        print(f"✓ Mode: Quantum ONLY (no classical fallback)")
        print(f"✓ Assets: {self.n_assets}")
        print(f"✓ Capital: ${self.capital:,.0f}")
        print(f"✓ Weight levels: {self.n_weight_levels}")
        print(f"✓ Weight range: [{self.min_weight:.2%}, {self.max_weight:.2%}]")
        
    def formulate_qubo(self,
                      risk_aversion: float = 0.5,
                      n_select: int = 15,
                      penalty_strength: float = 20.0) -> BinaryQuadraticModel:
        """
        Formulate QUBO for asset selection AND weight optimization.
        
        Strategy:
        ---------
        1. Each asset has K binary variables: x_{i,k} for k=1..K weight levels
        2. Constraint: sum_k x_{i,k} <= 1 (asset selected at most once)
        3. Constraint: sum_i sum_k x_{i,k} = N (exactly N assets selected)
        4. Weight of asset i: w_i = sum_k (k/K) * x_{i,k}
        
        Parameters
        ----------
        risk_aversion : float
            Risk aversion (0=return focus, 1=risk focus)
        n_select : int
            Number of assets to select (15)
        penalty_strength : float
            Penalty for constraint violations
        
        Returns
        -------
        BinaryQuadraticModel
            QUBO ready for quantum annealer
        """
        print(f"\n{'='*80}")
        print(f"QUBO FORMULATION - Pure Quantum")
        print(f"{'='*80}")
        print(f"Data source: {'IN-SAMPLE ✓' if self.is_in_sample else 'OUT-OF-SAMPLE ✗'}")
        print(f"Risk aversion: {risk_aversion}")
        print(f"Assets to select: {n_select}")
        print(f"Weight levels: {self.n_weight_levels}")
        print(f"Weight range: [{self.min_weight:.2%}, {self.max_weight:.2%}]")
        print(f"Penalty strength: {penalty_strength}")
        
        # Normalize for numerical stability
        mu_max = np.abs(self.expected_returns).max()
        sigma_max = np.abs(self.cov_matrix).max()
        
        mu_norm = self.expected_returns / (mu_max + 1e-10)
        sigma_norm = self.cov_matrix / (sigma_max + 1e-10)
        
        # Weight levels (normalized)
        weight_levels = np.linspace(self.min_weight, self.max_weight, self.n_weight_levels)
        weight_levels = weight_levels / weight_levels.sum()  # Normalize
        
        print(f"\nDiscretized weights: {[f'{w:.3f}' for w in weight_levels]}")
        
        # Total variables: n_assets * n_weight_levels
        n_vars = self.n_assets * self.n_weight_levels
        
        # Initialize BQM (using BINARY vartype)
        bqm = BinaryQuadraticModel(vartype=BINARY)
        
        # Helper function: variable index for asset i, weight level k
        def var_idx(i, k):
            return f"x_{i}_{k}"  # Use string labels instead of integers
        
        print(f"\nBuilding QUBO with {n_vars} variables...")
        
        # Objective function
        print("  → Adding objective function...")
        
        # Linear terms: -return(w) + risk(w) diagonal
        for i in range(self.n_assets):
            for k in range(self.n_weight_levels):
                var = var_idx(i, k)
                w_ik = weight_levels[k]
                
                # Return term (negative to maximize)
                h_return = -(1 - risk_aversion) * mu_norm[i] * w_ik
                
                # Risk term (diagonal)
                h_risk = risk_aversion * sigma_norm[i, i] * (w_ik ** 2)
                
                # Add to BQM
                bqm.add_variable(var, h_return + h_risk)
        
        # Quadratic terms: risk(w) off-diagonal
        print("  → Adding risk interactions...")
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                for k1 in range(self.n_weight_levels):
                    for k2 in range(self.n_weight_levels):
                        var_i = var_idx(i, k1)
                        var_j = var_idx(j, k2)
                        
                        w_i = weight_levels[k1]
                        w_j = weight_levels[k2]
                        
                        J_ij = 2 * risk_aversion * sigma_norm[i, j] * w_i * w_j
                        
                        # Set interaction directly
                        bqm.set_quadratic(var_i, var_j, J_ij)
        
        # Constraint 1: Each asset selected at most once
        print("  → Adding constraint: one weight per asset...")
        
        for i in range(self.n_assets):
            for k1 in range(self.n_weight_levels):
                var1 = var_idx(i, k1)
                
                # Linear: discourage multiple selections
                current = bqm.get_linear(var1)
                bqm.set_linear(var1, current + penalty_strength * 0.5)
                
                for k2 in range(k1 + 1, self.n_weight_levels):
                    var2 = var_idx(i, k2)
                    
                    # Quadratic: penalize selecting multiple weights for same asset
                    current_quad = bqm.get_quadratic(var1, var2, default=0)
                    bqm.set_quadratic(var1, var2, current_quad + penalty_strength)
        
        # Constraint 2: Exactly n_select assets selected
        print(f"  → Adding constraint: exactly {n_select} assets...")
        
        for i in range(self.n_assets):
            for k in range(self.n_weight_levels):
                var = var_idx(i, k)
                current = bqm.get_linear(var)
                bqm.set_linear(var, current + penalty_strength * (1 - 2*n_select))
        
        for i1 in range(self.n_assets):
            for k1 in range(self.n_weight_levels):
                for i2 in range(i1, self.n_assets):
                    k_start = k1 + 1 if i2 == i1 else 0
                    for k2 in range(k_start, self.n_weight_levels):
                        var1 = var_idx(i1, k1)
                        var2 = var_idx(i2, k2)
                        
                        if var1 != var2:
                            current_quad = bqm.get_quadratic(var1, var2, default=0)
                            bqm.set_quadratic(var1, var2, current_quad + 2 * penalty_strength)
        
        print(f"\n{'─'*80}")
        print(f"QUBO Statistics:")
        print(f"  Total variables: {len(bqm.variables)}")
        print(f"  Linear terms: {len(bqm.linear)}")
        print(f"  Quadratic terms: {len(bqm.quadratic)}")
        print(f"{'─'*80}")
        
        return bqm
    
    def solve_quantum(self,
                     bqm: BinaryQuadraticModel,
                     num_reads: int = 2000,
                     chain_strength: float = None,
                     annealing_time: int = 20) -> Dict:
        """
        Solve QUBO on D-Wave quantum annealer (PURE quantum, no fallback).
        
        Parameters
        ----------
        bqm : BinaryQuadraticModel
            QUBO problem
        num_reads : int
            Number of annealer samples
        chain_strength : float
            Chain strength (None=auto)
        annealing_time : int
            Annealing time (microseconds)
        
        Returns
        -------
        Dict
            Results with selected assets, weights, and metrics
        """
        print(f"\n{'='*80}")
        print(f"QUANTUM ANNEALING - D-Wave QPU (PURE QUANTUM)")
        print(f"{'='*80}")
        print("Connecting to D-Wave quantum computer...")
        
        try:
            sampler = EmbeddingComposite(DWaveSampler())
            
            qpu_info = sampler.child.properties
            print(f"✓ Connected to QPU: {qpu_info['chip_id']}")
            print(f"  Topology: {qpu_info['topology']['type']}")
            print(f"  Available qubits: {qpu_info['num_qubits']}")
            
            print(f"\nSubmitting job ({num_reads} reads, {annealing_time}μs)...")
            
            kwargs = {
                'num_reads': num_reads,
                'label': 'Portfolio_Pure_Quantum_V4',
                'answer_mode': 'histogram',
                'annealing_time': annealing_time
            }
            
            if chain_strength is not None:
                kwargs['chain_strength'] = chain_strength
            
            sampleset = sampler.sample(bqm, **kwargs)
            
            qpu_time = sampleset.info.get('timing', {})
            
            print(f"✓ Quantum annealing completed")
            print(f"  QPU access time: {qpu_time.get('qpu_access_time', 'N/A')} μs")
            print(f"  Total samples: {len(sampleset)}")
            
            # Aggregate to get unique solutions
            aggregated = sampleset.aggregate()
            print(f"  Unique solutions: {len(aggregated)}")
            
        except Exception as e:
            print(f"\n❌ QUANTUM ANNEALING FAILED: {e}")
            print("⚠️  NO CLASSICAL FALLBACK AVAILABLE")
            raise RuntimeError(f"Pure quantum optimization failed: {e}")
        
        # Parse best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        # Decode weights
        selected_assets = []
        optimal_weights = []
        
        weight_levels = np.linspace(self.min_weight, self.max_weight, self.n_weight_levels)
        
        for i in range(self.n_assets):
            asset_weight = 0.0
            for k in range(self.n_weight_levels):
                var = f"x_{i}_{k}"
                if var in best_sample and best_sample[var] == 1:
                    asset_weight = weight_levels[k]
                    break
            
            if asset_weight > 0:
                selected_assets.append(i)
                optimal_weights.append(asset_weight)
        
        # Normalize weights to sum to 1
        if len(optimal_weights) > 0:
            optimal_weights = np.array(optimal_weights)
            optimal_weights = optimal_weights / optimal_weights.sum()
        else:
            optimal_weights = np.array([])
            print("\n⚠️  WARNING: No assets selected by quantum solver!")
        
        selected_tickers = [self.asset_names[i] for i in selected_assets]
        
        if hasattr(sampleset.first, 'chain_break_fraction'):
            print(f"  Chain break fraction: {sampleset.first.chain_break_fraction:.3f}")
        
        results = {
            'sampleset': sampleset,
            'best_solution': best_sample,
            'best_energy': best_energy,
            'selected_assets': selected_assets,
            'selected_tickers': selected_tickers,
            'optimal_weights': optimal_weights,
            'n_selected': len(selected_assets),
            'qpu_info': {
                'chip_id': qpu_info['chip_id'],
                'topology': qpu_info['topology']['type'],
                'num_qubits': qpu_info['num_qubits'],
                'qpu_access_time': qpu_time.get('qpu_access_time', None)
            }
        }
        
        print(f"\n{'─'*80}")
        print(f"QUANTUM SOLUTION")
        print(f"{'─'*80}")
        print(f"Assets selected: {len(selected_assets)}")
        print(f"Best energy: {best_energy:.4f}")
        
        if len(selected_assets) > 0:
            print(f"\nOptimal Portfolio Weights:")
            print(f"{'Ticker':<8} {'Weight':>10} {'$ Amount':>12}")
            print(f"{'─'*80}")
            for ticker, weight in zip(selected_tickers, optimal_weights):
                amount = self.capital * weight
                print(f"{ticker:<8} {weight:>9.2%} ${amount:>11,.0f}")
            print(f"{'─'*80}")
            print(f"{'TOTAL':<8} {optimal_weights.sum():>9.2%} ${self.capital:>11,.0f}")
        
        return results
    
    def calculate_portfolio_metrics(self,
                                    selected_indices: List[int],
                                    weights: np.ndarray) -> Dict:
        """
        Calculate portfolio metrics for selected assets with weights.
        
        Parameters
        ----------
        selected_indices : List[int]
            Indices of selected assets
        weights : np.ndarray
            Portfolio weights
        
        Returns
        -------
        Dict
            Portfolio performance metrics
        """
        if len(selected_indices) == 0 or len(weights) == 0:
            return {
                'num_assets': 0,
                'return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'weights': np.array([]),
                'hhi': 0.0,
                'effective_n_assets': 0.0,
                'max_weight': 0.0,
                'min_weight': 0.0
            }
        
        # Extract relevant parameters
        selected_returns = self.expected_returns[selected_indices]
        selected_cov = self.cov_matrix[np.ix_(selected_indices, selected_indices)]
        
        # Portfolio metrics
        portfolio_return = np.dot(weights, selected_returns)
        portfolio_variance = np.dot(weights, np.dot(selected_cov, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
        
        # Diversification
        hhi = np.sum(weights ** 2)
        effective_n = 1 / hhi if hhi > 0 else 0
        
        return {
            'num_assets': len(selected_indices),
            'return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'weights': weights,
            'hhi': hhi,
            'effective_n_assets': effective_n,
            'max_weight': weights.max(),
            'min_weight': weights.min()
        }


if __name__ == "__main__":
    print("Pure Quantum Weight Optimizer - Section 2")
    print("⚠️  Requires D-Wave access - NO classical fallback")
    print("Run main script for complete workflow")