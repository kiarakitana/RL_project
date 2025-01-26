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

    # Reset price tracker for validation
    global price_tracker
    price_tracker = PriceTracker()

    # Create the environment with the validation set
    env = DataCenterEnv(args.path)
    state = env.observation()
    terminated = False

    total_reward = 0.0
    total_actual_reward = 0.0
    show = 360
    h = 0

    while not terminated:
        storage_level, price, hour, day = state

        # Update price tracker
        price_tracker.update(price, day)
        daily_avg = price_tracker.daily_avg
        weekly_avg = price_tracker.biweekly_avg
        
        bin_idx = small_storage_bin(storage_level)
        hour_idx = hour_bins(int(hour) - 1)
        daily_r_idx = daily_avg_diff_bins(price, daily_avg)
        weekly_r_idx = weekly_avg_diff_bins(price, weekly_avg)

        action = agent.get_action(
            bin_idx, daily_r_idx, weekly_r_idx, hour_idx, price, weekly_avg
            )

        next_state, reward, terminated = env.step(action)

        # Get the true action
        true_action = 0 if reward == 0 else reward / reward

        # Compute shaped reward
        actual_reward = reward_function(
            bin_idx, true_action, price,  price / daily_avg, price / weekly_avg, weekly_avg
            )

        total_reward += reward
        total_actual_reward += actual_reward

        state = next_state

        if h <= show:
            if true_action == 0:
                a = 'hold'
            if true_action == 1:
                a = 'buy'
            if true_action == -1:
                a = 'sell'
            print(f'day: {day}  |  hour: {hour}  |  storage: {storage_level}  |  price: {price}  |  daily avg: {round(daily_avg, 2)}  |  biweekly avg: {round(weekly_avg, 2)}  action: {a}  | reward: {round(actual_reward, 2)}')
            h += 1


    print("\nValidation Results:")
    print(f"Total Environment Reward: {total_reward:.2f}")
    print(f"Total Shaped Reward: {total_actual_reward:.2f}")

if __name__ == "__main__":
    validate()