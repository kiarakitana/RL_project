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

def run_training(args):
    # Create the agent
    agent = QLearningAgent(alpha=args.alpha, gamma=args.gamma, epsilon=1)
    
    # Reward history
    actual_rewards_history = deque(maxlen=5)
    
    # Training loop
    for epoch in range(args.epochs):
        # Reset state
        terminated = False
        total_reward = 0.0
        total_actual_reward = 0.0

        # Keep track of action counts
        action_counts = np.zeros(agent.n_actions, dtype=int)

        # Epsilon
        if epoch == 0:
            agent.epsilon = 1.0  # pure random exploration
        elif epoch > args.epochs - 1:
            agent.epsilon = 0  # agent test run
        else:
            # Decay epsilon
            agent.epsilon = max(args.epsilon_min,
                              agent.epsilon * args.epsilon_decay)
        
        # Reset environment and price tracker
        env = DataCenterEnv(args.path)
        global price_tracker
        price_tracker = PriceTracker()  # Reset price history
        
        state = env.observation()  # [storage_level, price, hour, day]
        
        while not terminated:
            storage_level, price, hour, day = state

            # Update price tracker if no outlier 
            price_tracker.update(price)
            daily_avg = price_tracker.daily_avg
            weekly_avg = price_tracker.biweekly_avg
            
            # Convert storage to bin
            bin_idx = small_storage_bin(storage_level)
            hour_idx = hour_bins(int(hour) - 1)
            daily_r_idx = daily_avg_diff_bins(price, daily_avg)
            weekly_r_idx = weekly_avg_diff_bins(price, weekly_avg)

            # Get action index from agent
            action_idx = agent.get_action(bin_idx, daily_r_idx, 
                                          weekly_r_idx, hour_idx)
            
            # Change to real action value
            action = -1 if action_idx == 2 else action_idx

            # Step in the environment
            next_state, reward, terminated = env.step(action)

            # Get the true action based on reward
            if reward > 0:
                true_action = -1
            elif reward < 0:
                true_action = 1
            else:
                true_action = 0

            # Get action index based on true action
            action_idx = 2 if true_action == -1 else true_action

            # Count actions
            action_counts[action_idx] += 1
            
            # Parse next state
            next_storage, next_price, next_hour, next_day = next_state

            price_tracker.update(next_price)
            next_daily_avg = price_tracker.daily_avg
            next_weekly_avg = price_tracker.biweekly_avg

            next_bin_idx = small_storage_bin(next_storage)
            next_daily_r_idx = daily_avg_diff_bins(price, next_daily_avg)
            next_weekly_r_idx = weekly_avg_diff_bins(price, next_weekly_avg)
            next_hour_idx = hour_bins(int(next_hour) - 1)
            
            # Calculate shaped reward
            actual_reward = reward_function(bin_idx, true_action, price/daily_avg, 
                                            price/weekly_avg, hour_idx, args.alfa_reward) 

            # Update agent
            agent.update(
                bin_idx, daily_r_idx, weekly_r_idx, hour_idx,
                action_idx,
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
    if args.tuning: 
        return agent

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
                                qvals[1] = 1  # buy
                            elif d == 6 or w == 6 and b > 0:  # High price => sell
                                qvals[2] = 1  # sell
                            elif b == 5:
                                qvals[2] = 1  # sell
                            else:
                                qvals[0] = 1  # hold
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
        columns=["Bin", "Daily_Ratio", "Weekly_Ratio", "Hour", "Hold", "Buy", "Sell"]
    )
    df_q.to_csv("q_table.csv", index=False)
    print("\nSaved Q-table in q_table.csv")

    # Save trained agent
    with open('trained_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    print("Agent saved in trained_agent.pkl")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx')
    parser.add_argument('--epochs', type=int, default=121)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--epsilon_decay', type=float, default=0.98)
    parser.add_argument('--epsilon_min', type=float, default=0.1)
    parser.add_argument('--alfa_reward', type=float, default=0.8)
    parser.add_argument('--tuning', type=bool, default=False)
    args = parser.parse_args()
    run_training(args)

if __name__ == "__main__":
    main()