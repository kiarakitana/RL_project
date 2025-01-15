import numpy as np
import pandas as pd
from env import DataCenterEnv
from tabularQN import TabularQAgent
import argparse
from datetime import datetime, timedelta

class AgentEvaluator:
    def __init__(self, q_table_path, validation_data_path):
        self.agent = TabularQAgent()
        self.agent.load(q_table_path)
        self.env = DataCenterEnv(validation_data_path)
        
    def evaluate(self):
        """Run full evaluation and return detailed performance metrics"""
        state = self.env.observation()
        terminated = False
        
        # Initialize tracking variables
        transactions = []  # Store all buy/sell actions
        energy_storage = []  # Track energy storage levels
        total_cost = 0
        daily_costs = []
        daily_storage = []
        current_day = 1

        while not terminated:
            # Get initial state info
            storage_level, price, hour, day = state
            
            # Get action from agent (using greedy policy)
            action = self.agent.get_action(state, validation=True)
            
            # Take step in environment
            next_state, reward, terminated = self.env.step(action)
            
            # Record transaction
            energy_transacted = action * self.env.max_power_rate
            cost = -reward if energy_transacted > 0 else -reward/0.8  # Adjust for selling efficiency
            
            transactions.append({
                'day': day,
                'hour': hour,
                'price': price,
                'action': action,
                'energy_transacted': energy_transacted,
                'cost': cost,
                'storage_level': storage_level,
                'reward': reward
            })
            
            # Track storage
            energy_storage.append({
                'day': day,
                'hour': hour,
                'storage_level': storage_level
            })
            
            # Track daily metrics
            if day != current_day:
                daily_costs.append({
                    'day': current_day,
                    'total_cost': sum(t['cost'] for t in transactions if t['day'] == current_day)
                })
                daily_storage.append({
                    'day': current_day,
                    'final_storage': storage_level
                })
                current_day = day
            
            total_cost += cost
            state = next_state
            
        # Convert to DataFrames
        transactions_df = pd.DataFrame(transactions)
        storage_df = pd.DataFrame(energy_storage)
        daily_costs_df = pd.DataFrame(daily_costs)
        daily_storage_df = pd.DataFrame(daily_storage)
        
        # Calculate summary statistics
        summary_stats = {
            'total_cost': total_cost,
            'average_daily_cost': total_cost / len(daily_costs_df),
            'total_energy_bought': transactions_df[transactions_df['energy_transacted'] > 0]['energy_transacted'].sum(),
            'total_energy_sold': -transactions_df[transactions_df['energy_transacted'] < 0]['energy_transacted'].sum(),
            'average_storage_level': storage_df['storage_level'].mean(),
            'peak_storage': storage_df['storage_level'].max(),
            'min_storage': storage_df['storage_level'].min(),
            'days_simulated': len(daily_costs_df)
        }
        
        return {
            'transactions': transactions_df,
            'storage_history': storage_df,
            'daily_costs': daily_costs_df,
            'daily_storage': daily_storage_df,
            'summary': summary_stats
        }
    
    def save_results(self, results, output_prefix='evaluation'):
        """Save all results to CSV files"""
        # Save transactions
        results['transactions'].to_csv(f'{output_prefix}_transactions.csv', index=False)
        
        # Save storage history
        results['storage_history'].to_csv(f'{output_prefix}_storage.csv', index=False)
        
        # Save daily metrics
        results['daily_costs'].to_csv(f'{output_prefix}_daily_costs.csv', index=False)
        results['daily_storage'].to_csv(f'{output_prefix}_daily_storage.csv', index=False)
        
        # Save summary stats
        pd.Series(results['summary']).to_csv(f'{output_prefix}_summary.csv')
        
        # Print summary
        print("\nEvaluation Summary:")
        print("-" * 50)
        print(f"Total Cost: ${results['summary']['total_cost']:,.2f}")
        print(f"Average Daily Cost: ${results['summary']['average_daily_cost']:,.2f}")
        print(f"Total Energy Bought: {results['summary']['total_energy_bought']:,.2f} MWh")
        print(f"Total Energy Sold: {results['summary']['total_energy_sold']:,.2f} MWh")
        print(f"Average Storage Level: {results['summary']['average_storage_level']:.2f} MWh")
        print(f"Peak Storage: {results['summary']['peak_storage']:.2f} MWh")
        print(f"Minimum Storage: {results['summary']['min_storage']:.2f} MWh")
        print(f"Days Simulated: {results['summary']['days_simulated']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_table', type=str, default='q_table.npy', help='Path to saved Q-table')
    parser.add_argument('--val_data', type=str, default='validate.xlsx', help='Path to validation data')
    parser.add_argument('--output', type=str, default='evaluation', help='Prefix for output files')
    args = parser.parse_args()
    
    evaluator = AgentEvaluator(args.q_table, args.val_data)
    results = evaluator.evaluate()
    evaluator.save_results(results, args.output)