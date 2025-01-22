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

    # Build day_of_week and month lookups from the same dataset
    dow_map, month_map = build_day_maps(path_to_dataset)
    
    # Create the agent
    agent = QLearningAgent(alpha=args.alpha,
                          gamma=args.gamma,
                          epsilon=1)
    
    # Use deques to store reward history
    actual_rewards_history = deque(maxlen=5)
    daily_rewards_history = deque(maxlen=24)  # Track rewards per hour
    
    # For multiple training epochs over the entire dataset
    for epoch in range(args.epochs):
        # Reset state
        terminated = False
        total_reward = 0.0
        total_actual_reward = 0.0
        daily_reward = 0.0

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
        
        # Initialize price tracker with first 336 hours (2 weeks) if available
        df = pd.read_excel(path_to_dataset)
        initial_prices = df.iloc[0:336, 1:25].values.flatten()  # Get first 2 weeks
        for init_price in initial_prices:
            price_tracker.update(init_price)
        
        while not terminated:
            storage_level, price, hour, day = state

            # Convert day -> day_of_week, month_of_year
            day_1_based = int(day)
            day_of_week = dow_map[day_1_based - 1]     # 0..6
            month_of_year = month_map[day_1_based - 1]  # 0..11
            
            # Convert storage to bin
            bin_idx = storage_to_bin(storage_level)
            price_idx = price_bins(price)
            hour_idx = int(hour) - 1
            dow_idx = int(day_of_week)
            month_idx = int(month_of_year)

            # Get action from agent
            action_idx = agent.get_action(bin_idx, price_idx, hour_idx, dow_idx, month_idx)
            action_counts[action_idx] += 1
            
            # Convert discrete action -> continuous
            action_cont = discrete_action_to_continuous(action_idx)
            
            # Step in the environment
            next_state, reward, terminated = env.step(action_cont)
            
            # Parse next state
            next_storage, next_price, next_hour, next_day = next_state
            next_day_1_based = int(next_day)
            next_dow = dow_map[next_day_1_based - 1]
            next_month = month_map[next_day_1_based - 1]

            next_bin_idx = storage_to_bin(next_storage)
            next_price_idx = price_bins(next_price)
            next_hour_idx = int(next_hour) - 1
            next_dow_idx = int(next_dow)
            next_month_idx = int(next_month)
            
            # Calculate shaped reward using price history
            actual_reward = reward_function(reward, bin_idx, action_cont, price)
            
            # Update agent
            agent.update(
                bin_idx, price_idx, hour_idx, dow_idx, month_idx,
                action_idx,
                actual_reward,
                next_bin_idx, next_price_idx, next_hour_idx, next_dow_idx, next_month_idx,
                terminated
            )
            
            # Update metrics
            total_reward += reward
            total_actual_reward += actual_reward
            daily_reward += actual_reward
            
            # Track daily performance
            if int(next_hour) == 1 and int(hour) == 24:  # Day completed
                daily_rewards_history.append(daily_reward)
                daily_reward = 0.0
                
                # Print daily stats every 7 days
                if len(daily_rewards_history) % 7 == 0:
                    avg_daily = np.mean(list(daily_rewards_history)[-7:])
                    print(f"Day {len(daily_rewards_history)} - Avg Daily Reward: {avg_daily:.2f}")
            
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
        for p in range(agent.n_prices):
            for h in range(agent.n_hours):
                for d in range(agent.n_dow):
                    for m in range(agent.n_month):
                        qvals = agent.Q[b, p, h, d, m, :]
                        if np.allclose(qvals, 0.0):
                            if p <= 9 and b < 17:  # Low price & not full => buy
                                qvals[1] = 1  # prefer buy
                            elif p >= agent.n_prices - 2:  # High price => sell
                                qvals[2] = 1  # prefer sell
                            else:
                                qvals[0] = 1  # prefer hold
                            agent.Q[b, p, h, d, m, :] = qvals

    # Save Q-table to CSV
    rows = []
    for b in range(agent.n_bins):
        for p in range(agent.n_prices):
            for h in range(agent.n_hours):
                for d in range(agent.n_dow):
                    for m in range(agent.n_month):
                        qvals = agent.Q[b, p, h, d, m, :]
                        rows.append([b, p, h, d, m] + list(qvals))

    df_q = pd.DataFrame(
        rows, 
        columns=["bin", "price", "hour", "dow", "month", "Q_hold", "Q_buy", "Q_sell"]
    )
    df_q.to_csv("q_table.csv", index=False)
    print("\nSaved Q-table to q_table.csv")

    # Save trained agent
    with open('trained_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    print("Saved trained agent to trained_agent.pkl")

if __name__ == "__main__":
    main()