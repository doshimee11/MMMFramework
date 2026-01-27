"""
Budget Optimization for MMM
Finds optimal marketing budget allocation across channels
Uses constrained optimization to maximize sales/ROI
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MARKETING_CHANNELS, OPTIMIZATION_CONFIG
from src.data_modeling import run_complete_modeling
    

class BudgetOptimizer:
    """
    Optimize marketing budget allocation
    
    Methods:
    - maximize_sales: Maximize predicted sales
    - maximize_roi: Maximize return on investment
    - minimize_cost: Minimize cost for target sales
    """
    
    def __init__(self, model, X_mean, feature_columns):
        self.model = model
        self.X_mean = X_mean  # Mean values for non-spend features
        self.feature_columns = feature_columns
        self.channel_list = list(MARKETING_CHANNELS.keys())
    
    def create_input_vector(self, spend_allocation):
        """Create feature vector from spend allocation"""
        X = self.X_mean.copy()
        
        for i, channel in enumerate(self.channel_list):
            spend_feature = f'{channel}_spend'
            if spend_feature in self.feature_columns:
                idx = self.feature_columns.index(spend_feature)
                X[idx] = spend_allocation[i]
        
        return X.reshape(1, -1)
    
    def objective_maximize_sales(self, spend_allocation):
        """Objective: Maximize predicted sales (minimize negative sales)"""
        X = self.create_input_vector(spend_allocation)
        predicted_sales = self.model.predict(X)[0]
        return -predicted_sales  # Negative because we minimize
    
    def objective_maximize_roi(self, spend_allocation):
        """Objective: Maximize ROI"""
        X = self.create_input_vector(spend_allocation)
        predicted_sales = self.model.predict(X)[0]
        
        total_spend = spend_allocation.sum()
        roi = (predicted_sales - total_spend) / total_spend if total_spend > 0 else 0
        
        return -roi  # Negative because we minimize
    
    def objective_minimize_cost(self, spend_allocation, target_sales):
        """Objective: Minimize cost to achieve target sales"""
        return spend_allocation.sum()
    
    def optimize(self, total_budget=None, objective='maximize_sales', method='scipy'):
        """
        Run optimization
        
        Args:
            total_budget: Total marketing budget constraint
            objective: 'maximize_sales', 'maximize_roi', 'minimize_cost'
            method: 'scipy' or 'evolutionary'
        
        Returns:
            Optimal allocation dictionary
        """
        
        total_budget = total_budget or OPTIMIZATION_CONFIG['total_budget']
        min_spend = OPTIMIZATION_CONFIG['min_budget_per_channel']
        max_spend = OPTIMIZATION_CONFIG['max_budget_per_channel']
        
        n_channels = len(self.channel_list)
        
        # Bounds for each channel
        bounds = [(min_spend, max_spend) for _ in range(n_channels)]
        
        # Constraints
        constraints = []
        
        # Budget constraint: sum of spend = total_budget
        constraints.append({
            'type': 'eq',
            'fun': lambda x: x.sum() - total_budget
        })
        
        # Select objective function
        if objective == 'maximize_sales':
            obj_func = self.objective_maximize_sales
        elif objective == 'maximize_roi':
            obj_func = self.objective_maximize_roi
        else:
            obj_func = self.objective_maximize_sales
        
        # Initial guess (equal allocation)
        x0 = np.ones(n_channels) * (total_budget / n_channels)
        
        # Optimize
        if method == 'scipy':
            result = minimize(
                obj_func,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            optimal_spend = result.x
        
        else:  # evolutionary algorithm
            result = differential_evolution(
                obj_func,
                bounds,
                constraints=constraints,
                seed=42,
                maxiter=100
            )
            optimal_spend = result.x
        
        # Create results
        X_optimal = self.create_input_vector(optimal_spend)
        predicted_sales = self.model.predict(X_optimal)[0]
        total_spend = optimal_spend.sum()
        roi = (predicted_sales - total_spend) / total_spend
        
        results = {
            'total_budget': total_budget,
            'predicted_sales': predicted_sales,
            'total_roi': roi,
            'allocation': {}
        }
        
        for i, channel in enumerate(self.channel_list):
            results['allocation'][channel] = {
                'spend': optimal_spend[i],
                'percentage': optimal_spend[i] / total_spend * 100
            }
        
        return results


def compare_scenarios(model, X_train, feature_columns, scenarios=None):
    """
    Compare different budget allocation scenarios
    
    Scenarios:
    - Current: Actual historical allocation
    - Optimized: Mathematical optimization
    - Equal: Equal allocation across channels
    - Digital-first: Prioritize digital channels
    """
    
    print("\n" + "="*70)
    print("SCENARIO PLANNING")
    print("="*70)
    
    # Calculate mean values for non-spend features
    X_mean = X_train.mean().values
    
    # Initialize optimizer
    optimizer = BudgetOptimizer(model, X_mean, feature_columns)
    
    results = {}
    
    # Scenario 1: Current allocation (historical average)
    print("\n1. Current Allocation...")
    current_allocation = []
    for channel in optimizer.channel_list:
        spend_feature = f'{channel}_spend'
        if spend_feature in X_train.columns:
            current_allocation.append(X_train[spend_feature].mean())
        else:
            current_allocation.append(0)
    
    current_allocation = np.array(current_allocation)
    X_current = optimizer.create_input_vector(current_allocation)
    current_sales = model.predict(X_current)[0]
    current_spend = current_allocation.sum()
    current_roi = (current_sales - current_spend) / current_spend
    
    results['current'] = {
        'total_budget': current_spend,
        'predicted_sales': current_sales,
        'total_roi': current_roi,
        'allocation': {
            channel: {
                'spend': current_allocation[i],
                'percentage': current_allocation[i] / current_spend * 100
            }
            for i, channel in enumerate(optimizer.channel_list)
        }
    }
    
    # Scenario 2: Optimized allocation
    print("2. Optimized Allocation...")
    results['optimized'] = optimizer.optimize(
        total_budget=current_spend,
        objective='maximize_sales'
    )
    
    # Scenario 3: Equal allocation
    print("3. Equal Allocation...")
    equal_allocation = np.ones(len(optimizer.channel_list)) * (current_spend / len(optimizer.channel_list))
    X_equal = optimizer.create_input_vector(equal_allocation)
    equal_sales = model.predict(X_equal)[0]
    equal_roi = (equal_sales - current_spend) / current_spend
    
    results['equal'] = {
        'total_budget': current_spend,
        'predicted_sales': equal_sales,
        'total_roi': equal_roi,
        'allocation': {
            channel: {
                'spend': equal_allocation[i],
                'percentage': equal_allocation[i] / current_spend * 100
            }
            for i, channel in enumerate(optimizer.channel_list)
        }
    }
    
    # Print comparison
    print("\n" + "="*70)
    print("SCENARIO COMPARISON")
    print("="*70)
    
    comparison = []
    for scenario_name, scenario_data in results.items():
        comparison.append({
            'Scenario': scenario_name.upper(),
            'Total Spend': f"${scenario_data['total_budget']:,.0f}",
            'Predicted Sales': f"${scenario_data['predicted_sales']:,.0f}",
            'ROI': f"{scenario_data['total_roi']:.2f}x",
            'Sales Lift vs Current': f"{(scenario_data['predicted_sales'] / results['current']['predicted_sales'] - 1) * 100:+.1f}%"
        })
    
    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False))
    
    # Detailed allocation comparison
    print("\n" + "="*70)
    print("ALLOCATION BREAKDOWN")
    print("="*70)
    
    for scenario_name, scenario_data in results.items():
        print(f"\n{scenario_name.upper()}:")
        for channel, alloc in scenario_data['allocation'].items():
            print(f"  {channel:10s}: ${alloc['spend']:8,.0f} ({alloc['percentage']:5.1f}%)")
    
    return results


def sensitivity_analysis(model, X_train, feature_columns, channel='tv', budget_range=(10000, 100000), n_points=20):
    """
    Perform sensitivity analysis for a specific channel
    Shows how sales respond to changes in channel spend
    """
    
    print("\n" + "="*70)
    print(f"SENSITIVITY ANALYSIS - {channel.upper()}")
    print("="*70)
    
    X_mean = X_train.mean().values
    spend_feature = f'{channel}_spend'
    
    if spend_feature not in feature_columns:
        print(f"Channel {channel} not found in features")
        return None
    
    # Create range of spend values
    spend_values = np.linspace(budget_range[0], budget_range[1], n_points)
    
    results = []
    for spend in spend_values:
        X = X_mean.copy()
        idx = feature_columns.index(spend_feature)
        X[idx] = spend
        
        predicted_sales = model.predict(X.reshape(1, -1))[0]
        roi = (predicted_sales - spend) / spend
        roas = predicted_sales / spend
        
        results.append({
            'spend': spend,
            'predicted_sales': predicted_sales,
            'roi': roi,
            'roas': roas,
            'incremental_sales': predicted_sales - (model.predict(np.zeros((1, len(X_mean))))[0] if spend == 0 else results[-1]['predicted_sales'])
        })
    
    results_df = pd.DataFrame(results)
    
    print(f"\nSpend range: ${budget_range[0]:,.0f} - ${budget_range[1]:,.0f}")
    print(f"Maximum ROI: {results_df['roi'].max():.2f}x at ${results_df.loc[results_df['roi'].idxmax(), 'spend']:,.0f}")
    print(f"Optimal spend (max sales): ${results_df.loc[results_df['predicted_sales'].idxmax(), 'spend']:,.0f}")
    
    return results_df


if __name__ == "__main__":
    # Run modeling
    mmm_results = run_complete_modeling()
    
    # Get best model
    best_model = mmm_results['models'][mmm_results['best_model']]
    data = mmm_results['data']
    
    # Run scenario comparison
    scenarios = compare_scenarios(
        best_model,
        data['X_train'],
        data['feature_columns']
    )
    
    # Sensitivity analysis for top channel
    top_channel = mmm_results['roi'].iloc[0]['channel']
    sensitivity = sensitivity_analysis(
        best_model,
        data['X_train'],
        data['feature_columns'],
        channel=top_channel
    )
