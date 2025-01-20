import argparse
import pickle
import pandas as pd
from collections import deque
from env import DataCenterEnv
from TQN.tqn_utility import *
from TQN.tqn import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='train.xlsx', help='Path to the Excel data file.')
    parser.add_argument('--epochs', type=int, default=151, help='Number of passes over the entire dataset.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
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
    
    # Use a deque to store only the last 5 actual rewards
    actual_rewards_history = deque(maxlen=5)
    
    # For multiple training epochs over the entire dataset
    for epoch in range(args.epochs):
        # We will step through the entire data until termination
        terminated = False
        total_reward = 0.0
        total_actual_reward = 0.0

        # Keep track of how many times each action is chosen per epoch
        action_counts = np.zeros(agent.n_actions, dtype=int)

        # Epsilon Strategy
        if epoch == 0:
            agent.epsilon = 1.0  # pure random exploration
        if epoch > args.epochs - 1:
            agent.epsilon = 0 # agent test run on test set
        else:
            # Decay epsilon but don't go below epsilon_min
            agent.epsilon = max(args.epsilon_min,
                                agent.epsilon * args.epsilon_decay)
        
        # Reset environment
        env = DataCenterEnv(path_to_dataset)
        state = env.observation()  # [storage_level, price, hour, day]
        
        while not terminated:
            storage_level, price, hour, day = state

            # Convert day -> day_of_week, month_of_year
            day_1_based = int(day)  # 1..N
            day_of_week = dow_map[day_1_based - 1]     # 0..6
            month_of_year = month_map[day_1_based - 1] # 0..11
            
            # Convert storage to bin
            bin_idx = storage_to_bin(storage_level)
            price_idx = price_bins(price)
            hour_idx = int(hour) - 1  # 0..23
            dow_idx = int(day_of_week)
            month_idx = int(month_of_year)

            # Epsilon-greedy action
            action_idx = agent.get_action(bin_idx, price_idx, hour_idx, dow_idx, month_idx)

            # Count how many times each action is used
            action_counts[action_idx] += 1
            
            # Convert discrete action -> continuous
            action_cont = discrete_action_to_continuous(action_idx)
            
            # Step in the environment
            next_state, reward, terminated = env.step(action_cont)
            
            # next_state => [next_storage, next_price, next_hour, next_day]
            next_storage, next_price, next_hour, next_day = next_state
            
            # next day => day_of_week, month_of_year
            next_day_1_based = int(next_day)
            next_dow = dow_map[next_day_1_based - 1]     # 0..6
            next_month = month_map[next_day_1_based - 1] # 0..11

            next_bin_idx = storage_to_bin(next_storage)
            next_price_idx = price_bins(next_price)
            next_hour_idx = int(next_hour) - 1
            next_dow_idx = int(next_dow)
            next_month_idx = int(next_month)
            
            # Calculate the shaped reward
            actual_reward = reward_function(reward, bin_idx, action_cont, price)
            
            # Q-update
            agent.update(
                bin_idx, price_idx, hour_idx, dow_idx, month_idx,
                action_idx,
                actual_reward,
                next_bin_idx, next_price_idx, next_hour_idx, next_dow_idx, next_month_idx,
                terminated
            )
            
            total_reward += reward
            total_actual_reward += actual_reward
            state = next_state
        
        print(f"Epoch {epoch+1}/{args.epochs} - Epsilon: {agent.epsilon:.2f} - Total Reward: {total_reward:.2f} - Actual Reward: {total_actual_reward:.2f}")

        # Add the actual reward to the deque
        actual_rewards_history.append(total_actual_reward)

        # Convergence check: only if we have exactly 5 entries
        if len(actual_rewards_history) == 5:
            last_five = list(actual_rewards_history)
            avg_5 = np.mean(last_five)
            range_5 = max(last_five) - min(last_five)
            
            # If average is not zero and the range is below 2% of it
            if avg_5 != 0 and range_5 < 0.01 * avg_5 and agent.epsilon == 0.1:
                print("Convergence criterion met (last 5 rewards differ by < 2%). Stopping early.")
                break


    print("Training finished!")

    print(f"Actions used:")
    print(f"  Hold (0): {action_counts[0]}")
    print(f"  Buy  (1): {action_counts[1]}")
    print(f"  Sell (2): {action_counts[2]}")

    q_table_size = agent.Q.size  # product of all dimensions
    print(f"Q-table shape = {agent.Q.shape}, total elements = {q_table_size}")

    # ------------------------------------------------------------------
    # Identify & filling all states that have Q = [0, 0, 0].
    # ------------------------------------------------------------------
    for b in range(agent.n_bins):
        for p in range(agent.n_prices):
            for h in range(agent.n_hours):
                for d in range(agent.n_dow):
                    for m in range(agent.n_month):
                        qvals = agent.Q[b, p, h, d, m, :]
                        # Check if all actions are 0 => never visited
                        if np.allclose(qvals, 0.0):
                            # Trick the agent with default Q-values
                            if p <= 9 and b < 17:  
                                # Price price_bin <= 9 & storage < 17  => buy
                                qvals[0] = 0  # hold
                                qvals[1] = 1  # buy
                                qvals[2] = 0  # sell
                            elif p == agent.n_prices - 1:
                                # Last price bin => sell
                                qvals[0] = 0
                                qvals[1] = 0
                                qvals[2] = 1
                            else:
                                # Otherwise => hold
                                qvals[0] = 1
                                qvals[1] = 0
                                qvals[2] = 0
                            agent.Q[b, p, h, d, m, :] = qvals

    # ------------------------------------------------------------------
    # Build the rows for the CSV
    # ------------------------------------------------------------------
    rows = []
    for b in range(agent.n_bins):
        for p in range(agent.n_prices):
            for h in range(agent.n_hours):
                for d in range(agent.n_dow):
                    for m in range(agent.n_month):
                        qvals = agent.Q[b, p, h, d, m, :]
                        rows.append([b, p, h, d, m] + list(qvals))

    # Create a DataFrame
    df_q = pd.DataFrame(
        rows, 
        columns=["bin", "price", "hour", "dow", "month", "Q_hold", "Q_buy", "Q_sell"]
    )

    # Write to CSV
    df_q.to_csv("q_table.csv", index=False)
    print("Saved Q-table to q_table.csv.")


    with open('trained_agent.pkl', 'wb') as f:
        pickle.dump(agent, f)
    print("Trained agent saved to 'trained_agent.pkl'.")


if __name__ == "__main__":
    main()
