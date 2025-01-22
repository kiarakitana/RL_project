import argparse
import pickle
import sys
import os
from collections import deque

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DataCenterEnv
from tqn_utility import *
from tqn import *

def validate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='validate.xlsx',
                       help='Path to the validation Excel data file.')
    args = parser.parse_args()

    # Load the trained agent
    try:
        with open('trained_agent.pkl', 'rb') as f:
            agent = pickle.load(f)
    except FileNotFoundError:
        print("Error: trained_agent.pkl not found. Please run training first.")
        return

    # Set epsilon=0 for purely greedy evaluation
    agent.epsilon = 0.0

    # Build day_of_week and month lookups for the validation data
    dow_map, month_map = build_day_maps(args.path)

    # Reset price tracker for validation
    global price_tracker
    price_tracker = PriceTracker()

    # Create the environment with the validation set
    env = DataCenterEnv(args.path)
    state = env.observation()
    terminated = False

    total_reward = 0.0
    total_actual_reward = 0.0
    daily_rewards = []
    current_day_reward = 0.0
    last_hour = 1

    while not terminated:
        storage_level, price, hour, day = state
        
        # Track daily rewards
        if hour == 1 and last_hour == 24:
            daily_rewards.append(current_day_reward)
            current_day_reward = 0.0
        last_hour = hour

        # Convert day -> day_of_week, month_of_year
        day_1_based = int(day)
        day_of_week = dow_map[day_1_based - 1]
        month_of_year = month_map[day_1_based - 1]

        # Update price tracker
        price_tracker.update(price)
        
        bin_idx = storage_to_bin(storage_level)
        price_idx = price_bins(price)
        hour_idx = int(hour) - 1
        dow_idx = int(day_of_week)
        month_idx = int(month_of_year)

        # Choose best action (purely greedy)
        action_idx = agent.get_action(bin_idx, price_idx, hour_idx, dow_idx, month_idx)
        action_cont = discrete_action_to_continuous(action_idx)

        next_state, reward, terminated = env.step(action_cont)

        # Compute shaped reward
        actual_reward = reward_function(reward, bin_idx, action_cont, price)

        total_reward += reward
        total_actual_reward += actual_reward
        current_day_reward += actual_reward

        state = next_state

    # Add last day's reward if needed
    if current_day_reward != 0:
        daily_rewards.append(current_day_reward)

    print("\nValidation Results:")
    print(f"Total Environment Reward: {total_reward:.2f}")
    print(f"Total Shaped Reward: {total_actual_reward:.2f}")
    print(f"Average Daily Reward: {np.mean(daily_rewards):.2f}")
    print(f"Best Daily Reward: {max(daily_rewards):.2f}")
    print(f"Worst Daily Reward: {min(daily_rewards):.2f}")
    print(f"Number of Days: {len(daily_rewards)}")

if __name__ == "__main__":
    validate()