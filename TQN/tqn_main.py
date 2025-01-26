import argparse
import pickle
import pandas as pd
import sys
import os
from collections import deque

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DataCenterEnv
from tqn_utility import *
from tqn import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx', 
                        help='Path to the Excel data file.')
    parser.add_argument('--epochs', type=int, default=151,
                        help='Number of passes over the entire dataset.')
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor.')
    parser.add_argument('--epsilon_decay', type=float, default=0.98,
                        help="Multiply epsilon by this factor each epoch.")
    parser.add_argument('--epsilon_min', type=float, default=0.1,
                        help="Minimum epsilon value to avoid overfitting.")
    args = parser.parse_args()

    path_to_dataset = args.path
    
    # Create the agent
    agent = QLearningAgent(alpha=args.alpha,
                          gamma=args.gamma,
                          epsilon=1)
    
    # Use deques to store reward history
    actual_rewards_history = deque(maxlen=5)
    
    # For multiple training epochs over the entire dataset
    for epoch in range(args.epochs):
        # Reset state
        terminated = False
        total_reward = 0.0
        total_actual_reward = 0.0
        # daily_reward = 0.0

        # Keep track of action counts
        action_counts = np.zeros(agent.n_actions, dtype=int)

        # Epsilon Strategy
        if epoch == 0:
            agent.epsilon = 1.0  # pure random exploration
        elif epoch > args.epochs - 1:
            agent.epsilon = 0  # agent test run
        else:
            # Decay epsilon but don't go below epsilon_min
            agent.epsilon = max(args.epsilon_min,
                              agent.epsilon * args.epsilon_decay)
        
        # Reset environment and price tracker
        env = DataCenterEnv(path_to_dataset)
        global price_tracker
        price_tracker = PriceTracker()  # Reset price history
        
        state = env.observation()  # [storage_level, price, hour, day]
        
        while not terminated:
            storage_level, price, hour, day = state

            # Update price tracker
            price_tracker.update(price, day)
            daily_avg = price_tracker.daily_avg
            weekly_avg = price_tracker.biweekly_avg
            
            # Convert storage to bin
            bin_idx = small_storage_bin(storage_level)
            hour_idx = hour_bins(int(hour) - 1)
            daily_r_idx = daily_avg_diff_bins(price, daily_avg)
            weekly_r_idx = weekly_avg_diff_bins(price, weekly_avg)

            # Get action from agent
            action = agent.get_action(bin_idx, daily_r_idx, weekly_r_idx, hour_idx, price, weekly_avg)
            
            # Step in the environment
            next_state, reward, terminated = env.step(action)

            # Get the true action
            true_action = 0 if reward == 0 else int(reward / reward)

            # Count actions
            if action == -1:
                action_counts[2] += 1
            else:
                action_counts[true_action] += 1
            
            # Parse next state
            next_storage, next_price, next_hour, next_day = next_state

            price_tracker.update(next_price, next_day)
            next_daily_avg = price_tracker.daily_avg
            next_weekly_avg = price_tracker.biweekly_avg

            next_bin_idx = small_storage_bin(next_storage)
            next_daily_r_idx = daily_avg_diff_bins(price, next_daily_avg)
            next_weekly_r_idx = weekly_avg_diff_bins(price, next_weekly_avg)
            next_hour_idx = hour_bins(int(next_hour) - 1)
            
            # Calculate shaped reward using price history
            actual_reward = reward_function(bin_idx, true_action, price,  price / daily_avg, 
                                            price / weekly_avg, weekly_avg)
            # print(f'reward: {round(actual_reward, 2)} price: {price} daily_avg: {round(daily_avg, 2)} weekly_avg: {round(weekly_avg, 2)} action: {action_cont}') 

            # Update agent
            agent.update(
                bin_idx, daily_r_idx, weekly_r_idx, hour_idx,
                true_action,
                actual_reward,
                next_bin_idx, next_daily_r_idx, next_weekly_r_idx, next_hour_idx,
                terminated
            )
            
            # Update metrics
            total_reward += reward
            total_actual_reward += actual_reward
            
            state = next_state

        # Epoch complete - print stats
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Epsilon: {agent.epsilon:.2f}")
        print(f"Total Env Reward: {total_reward:.2f}")
        print(f"Total Shaped Reward: {total_actual_reward:.2f}")
        print(f"Action counts - Hold: {action_counts[0]}, Buy: {action_counts[1]}, Sell: {action_counts[2]}\n")

        # Add the actual reward to history
        actual_rewards_history.append(total_actual_reward)

        # Convergence check with 5 epochs
        if len(actual_rewards_history) == 5:
            last_five = list(actual_rewards_history)
            avg_5 = np.mean(last_five)
            range_5 = max(last_five) - min(last_five)
            
            if avg_5 != 0 and range_5 < 0.01 * abs(avg_5) and agent.epsilon == args.epsilon_min:
                print("Convergence criterion met (last 5 rewards differ by < 1%). Stopping early.")
                break

    print("\nTraining Summary:")
    print(f"Final action distribution:")
    print(f"  Hold (0): {action_counts[0]}")
    print(f"  Buy  (1): {action_counts[1]}")
    print(f"  Sell (2): {action_counts[2]}")

    q_table_size = agent.Q.size
    print(f"Q-table shape = {agent.Q.shape}, total elements = {q_table_size}")

    # Fill unvisited states with defaults
    for b in range(agent.n_bins):
        for d in range(agent.n_daily_ratio):
            for w in range(agent.n_weekly_ratio):
                for h in range(agent.n_hours):
                        qvals = agent.Q[b, d, w, h, :]
                        if np.allclose(qvals, 0.0):
                            if d <= 1 and w <= 1 and b < 4:  # Low price & not full => buy
                                qvals[1] = 1  # prefer buy
                            elif d == 6 or w == 6 and b > 0:  # High price => sell
                                qvals[2] = 1  # prefer sell
                            elif b == 5:
                                qvals[2] = 1  # prefer sell
                            else:
                                qvals[0] = 1  # prefer hold
                            agent.Q[b, d, w, h, :] = qvals

    # Save Q-table to CSV
    rows = []
    for b in range(agent.n_bins):
        for d in range(agent.n_daily_ratio):
            for w in range(agent.n_weekly_ratio):
                for h in range(agent.n_hours):
                        qvals = agent.Q[b, d, w, h, :]
                        rows.append([b, d, w, h] + list(qvals))

    df_q = pd.DataFrame(
        rows, 
        columns=["bin", "daily_ratio", "weekly_ratio", "hour", "Q_hold", "Q_buy", "Q_sell"]
    )
    df_q.to_csv("q_table.csv", index=False)
    print("\nSaved Q-table to q_table.csv")

    # Save trained agent
    with open('trained_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    print("Saved trained agent to trained_agent.pkl")

if __name__ == "__main__":
    main()