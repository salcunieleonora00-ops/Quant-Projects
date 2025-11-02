"""
Section 2: QUBO Formulation and Quantum Optimization
====================================================

This module formulates the portfolio optimization problem as a Quadratic 
Unconstrained Binary Optimization (QUBO) problem and solves it using the
D-Wave quantum annealer via API.

Mathematical Formulation:
------------------------
minimize: f(x) = lambda * risk(x) - (1 - lambda) * return(x)
subject to: sum(x_i) = K (budget constraint)

where:
- x_i in {0, 1} indicates selection of asset i
- risk(x) = x^T * Sigma * x (portfolio variance)
- return(x) = mu^T * x (expected return)
- lambda: risk aversion parameter [0, 1]
- K: number of assets to select

The budget constraint is enforced via penalty method:
penalty = P * (sum(x_i) - K)^2

References:
- Markowitz, H. (1952). Portfolio Selection.
- Lucas, A. (2014). Ising formulations of many NP problems.

Author: Portfolio Optimization System
Version: 1.0
"""

import numpy as np
from typing import Tuple, List, Dict
from dimod import BinaryQuadraticModel
from dwave.system import DWaveSampler, EmbeddingComposite


class QuantumPortfolioOptimizer:
    """
    Quantum portfolio optimization using D-Wave annealer.
    """
    
    def __init__(self, expected_returns: np.ndarray, 
                 cov_matrix: np.ndarray,
                 asset_names: List[str]):
        """
        Initialize quantum optimizer.
        
        Parameters
        ----------
        expected_returns : np.ndarray
            Expected annual returns for each asset
        cov_matrix : np.ndarray
            Annual covariance matrix
        asset_names : List[str]
            Asset ticker symbols
        """
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.asset_names = asset_names
        self.n_assets = len(expected_returns)
        
    def formulate_qubo(self, 
                       risk_aversion: float = 0.5,
                       budget: int = 5,
                       penalty_strength: float = 10.0) -> BinaryQuadraticModel:
        """
        Formulate portfolio optimization as QUBO problem.
        
        Mathematical Details:
        -------------------
        Objective: E(x) = sum_i h_i * x_i + sum_{i<j} J_ij * x_i * x_j
        
        where:
        h_i = -lambda * mu_i + (1-lambda) * Sigma_ii (linear: return and own variance)
        J_ij = 2 * (1-lambda) * Sigma_ij (quadratic: covariance between assets)
        
        Budget constraint (penalty method):
        E_constraint = P * (sum_i x_i - K)^2
                     = P * (sum_i x_i^2 + 2*sum_{i<j} x_i*x_j - 2K*sum_i x_i + K^2)
        
        Since x_i^2 = x_i for binary variables:
        h_i += P * (1 - 2K)
        J_ij += 2P
        
        Parameters
        ----------
        risk_aversion : float
            Risk aversion parameter lambda in [0, 1]
            0 = maximize return only, 1 = minimize risk only
        budget : int
            Number of assets to select
        penalty_strength : float
            Penalty parameter P for budget constraint
        
        Returns
        -------
        BinaryQuadraticModel
            QUBO formulation ready for quantum annealer
        """
        print("\nFormulating QUBO problem...")
        print(f"  Risk aversion (lambda): {risk_aversion}")
        print(f"  Budget constraint: {budget} assets")
        print(f"  Penalty strength: {penalty_strength}")
        
        # Normalize data for numerical stability
        mu_norm = self.expected_returns / np.abs(self.expected_returns).max()
        sigma_norm = self.cov_matrix / np.abs(self.cov_matrix).max()
        
        # Initialize Binary Quadratic Model
        bqm = BinaryQuadraticModel('BINARY')
        
        # Linear terms: h_i = -lambda * mu_i + (1-lambda) * Sigma_ii
        # We want to maximize return (negative coefficient)
        # And include diagonal variance in linear terms
        for i in range(self.n_assets):
            h_i = -risk_aversion * mu_norm[i] + (1 - risk_aversion) * sigma_norm[i, i]
            bqm.add_variable(i, h_i)
        
        # Quadratic terms: J_ij = 2 * (1-lambda) * Sigma_ij for i < j
        # This represents the covariance between different assets
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                # Off-diagonal: covariance
                # Factor of 2 because x_i * x_j appears once but represents symmetric interaction
                J_ij = 2 * (1 - risk_aversion) * sigma_norm[i, j]
                bqm.add_interaction(i, j, J_ij)
        
        # Add budget constraint penalty
        # Expand (sum x_i - K)^2 = sum x_i^2 + 2*sum_{i<j} x_i*x_j - 2K*sum x_i + K^2
        # K^2 term is constant and can be ignored
        
        for i in range(self.n_assets):
            # From x_i^2 term (which equals x_i for binary) and linear term -2K*x_i
            current_linear = bqm.get_linear(i)
            bqm.set_linear(i, current_linear + penalty_strength * (1 - 2*budget))
        
        for i in range(self.n_assets):
            for j in range(i + 1, self.n_assets):
                # From 2*x_i*x_j term
                current_quad = bqm.get_quadratic(i, j)
                bqm.set_quadratic(i, j, current_quad + 2 * penalty_strength)
        
        print(f"  QUBO size: {self.n_assets} variables, {len(bqm.quadratic)} interactions")
        
        return bqm
    
    def solve_quantum(self, 
                     bqm: BinaryQuadraticModel,
                     num_reads: int = 1000,
                     chain_strength: float = None) -> Dict:
        """
        Solve QUBO on D-Wave quantum annealer via API.
        
        Parameters
        ----------
        bqm : BinaryQuadraticModel
            QUBO problem to solve
        num_reads : int
            Number of samples from quantum annealer
        chain_strength : float
            Strength of chains in embedding (None = auto)
        
        Returns
        -------
        Dict
            Results including best solution, energy, and all samples
        """
        print("\nConnecting to D-Wave quantum annealer...")
        
        # Initialize sampler with automatic embedding
        sampler = EmbeddingComposite(DWaveSampler())
        
        # Get QPU information
        qpu_info = sampler.child.properties
        print(f"  QPU: {qpu_info['chip_id']}")
        print(f"  Topology: {qpu_info['topology']['type']}")
        print(f"  Available qubits: {qpu_info['num_qubits']}")
        
        print(f"\nSubmitting to quantum annealer ({num_reads} reads)...")
        
        # Prepare sampling parameters
        kwargs = {
            'num_reads': num_reads,
            'label': 'Portfolio_Optimization',
            'answer_mode': 'histogram'
        }
        
        if chain_strength is not None:
            kwargs['chain_strength'] = chain_strength
        
        # Execute on quantum hardware
        sampleset = sampler.sample(bqm, **kwargs)
        
        print(f"  Quantum annealing completed")
        print(f"  Total samples received: {len(sampleset)}")
        print(f"  Unique solutions: {len(sampleset.aggregate())}")
        
        # Extract best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        selected_assets = [i for i, val in best_sample.items() if val == 1]
        
        results = {
            'sampleset': sampleset,
            'best_solution': best_sample,
            'best_energy': best_energy,
            'selected_assets': selected_assets,
            'selected_tickers': [self.asset_names[i] for i in selected_assets],
            'qpu_info': {
                'chip_id': qpu_info['chip_id'],
                'topology': qpu_info['topology']['type'],
                'num_qubits': qpu_info['num_qubits']
            }
        }
        
        return results
    
    def calculate_portfolio_metrics(self, selected_indices: List[int]) -> Dict:
        """
        Calculate performance metrics for selected portfolio.
        
        Parameters
        ----------
        selected_indices : List[int]
            Indices of selected assets
        
        Returns
        -------
        Dict
            Portfolio metrics including return, risk, and Sharpe ratio
        """
        if len(selected_indices) == 0:
            return {
                'num_assets': 0,
                'return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0
            }
        
        # Equal-weighted portfolio
        n = len(selected_indices)
        weights = np.zeros(self.n_assets)
        weights[selected_indices] = 1.0 / n
        
        # Calculate metrics
        portfolio_return = np.dot(weights, self.expected_returns)
        portfolio_variance = np.dot(weights, np.dot(self.cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = portfolio_return / portfolio_volatility
        
        metrics = {
            'num_assets': n,
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'weights': weights
        }
        
        return metrics


def run_quantum_optimization(expected_returns: np.ndarray,
                            cov_matrix: np.ndarray,
                            asset_names: List[str],
                            risk_aversion: float = 0.5,
                            budget: int = 5,
                            num_reads: int = 1000) -> Tuple[Dict, Dict]:
    """
    Complete quantum portfolio optimization workflow.
    
    Parameters
    ----------
    expected_returns : np.ndarray
        Expected returns
    cov_matrix : np.ndarray
        Covariance matrix
    asset_names : List[str]
        Asset names
    risk_aversion : float
        Risk aversion parameter
    budget : int
        Number of assets to select
    num_reads : int
        QPU samples
    
    Returns
    -------
    results : Dict
        Quantum annealer results
    metrics : Dict
        Portfolio performance metrics
    """
    # Initialize optimizer
    optimizer = QuantumPortfolioOptimizer(expected_returns, cov_matrix, asset_names)
    
    # Formulate QUBO
    bqm = optimizer.formulate_qubo(
        risk_aversion=risk_aversion,
        budget=budget,
        penalty_strength=15.0
    )
    
    # Solve on quantum annealer
    results = optimizer.solve_quantum(bqm, num_reads=num_reads)
    
    # Calculate metrics
    metrics = optimizer.calculate_portfolio_metrics(results['selected_assets'])
    
    return results, metrics


if __name__ == "__main__":
    print("This module requires inputs from Section 1 (Portfolio Selection)")
    print("Run the integrated workflow in main.py")