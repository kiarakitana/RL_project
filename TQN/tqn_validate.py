import pickle
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DataCenterEnv
from tqn_utility import *
from tqn import *

def validate(path='validate.xlsx', show=False, agent=False):
    # Load the trained agent
    if not agent:
        try:
            with open('trained_agent_best.pkl', 'rb') as f:
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
    env = DataCenterEnv(path)
    state = env.observation()
    terminated = False

    total_reward = 0.0
    total_actual_reward = 0.0
    h = 366

    while not terminated:
        storage_level, price, hour, day = state

        # Update price tracker
        price_tracker.update(price)
        daily_avg = price_tracker.daily_avg
        weekly_avg = price_tracker.biweekly_avg
        
        bin_idx = small_storage_bin(storage_level)
        hour_idx = hour_bins(int(hour) - 1)
        daily_r_idx = daily_avg_diff_bins(price, daily_avg)
        weekly_r_idx = weekly_avg_diff_bins(price, weekly_avg)

        action_idx = agent.get_action(bin_idx, daily_r_idx, weekly_r_idx, hour_idx)

        action = -1 if action_idx == 2 else action_idx

        next_state, reward, terminated = env.step(action)

        reward = round(reward)

        # Get the true action
        if reward > 0:
            true_action = -1
        elif reward < 0:
            true_action = 1
        else:
            true_action = 0

        action_idx = 2 if true_action == -1 else true_action

        # Compute shaped reward
        actual_reward = reward_function(
            bin_idx, true_action, price / daily_avg, price / weekly_avg, hour_idx
            )

        total_reward += reward
        total_actual_reward += actual_reward
        state = next_state

        if show:
            if h != 0:
                if action_idx == 0:
                    a = 'hold'
                if action_idx == 1:
                    a = 'buy'
                if action_idx == 2:
                    a = 'sell'
                print(f'day: {day} | hour: {hour} | storage: {storage_level} | price: {price} | daily avg: {round(daily_avg, 2)} | biweekly avg: {round(weekly_avg, 2)} action: {a} | reward: {round(actual_reward, 2)}')
                h -= 1


    print("\nValidation Results:")
    print(f"Total Environment Reward: {total_reward:.2f}")
    print(f"Total Shaped Reward: {total_actual_reward:.2f}")
    return total_reward

if __name__ == "__main__":
    validate()