import argparse
import pickle
from env import DataCenterEnv
from TQN.tqn_utility import *


def validate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='validate.xlsx', 
                        help='Path to the validation Excel data file.')
    args = parser.parse_args()

    # Load the trained agent
    with open('trained_agent.pkl', 'rb') as f:
        agent = pickle.load(f)
    
    # Set epsilon=0 for purely greedy evaluation (or a small value if desired)
    agent.epsilon = 0.0

    # Build day_of_week and month lookups for the validation data
    dow_map, month_map = build_day_maps(args.path)

    # Create the environment with the validation set
    env = DataCenterEnv(args.path)
    state = env.observation()
    terminated = False

    total_reward = 0.0
    total_actual_reward = 0.0

    while not terminated:
        storage_level, price, hour, day = state
        
        # Convert day -> day_of_week, month_of_year
        day_1_based = int(day)
        day_of_week = dow_map[day_1_based - 1]     
        month_of_year = month_map[day_1_based - 1]
        
        bin_idx = storage_to_bin(storage_level)
        price_idx = price_bins(price)
        hour_idx = int(hour) - 1
        dow_idx = int(day_of_week)
        month_idx = int(month_of_year)

        # Choose best action (since epsilon=0, this is purely greedy)
        action_idx = agent.get_action(bin_idx, price_idx, hour_idx, dow_idx, month_idx)
        action_cont = discrete_action_to_continuous(action_idx)
        
        next_state, reward, terminated = env.step(action_cont)

        # Compute any shaped reward if you want to track it
        actual_reward = reward_function(reward, bin_idx, action_cont, price)
        
        total_reward += reward
        total_actual_reward += actual_reward
        
        state = next_state

    print(f"Validation run completed.")
    print(f"Total Env Reward: {total_reward:.2f}")
    print(f"Total Shaped Reward: {total_actual_reward:.2f}")

if __name__ == "__main__":
    validate()
