import numpy as np
import random


class QLearningAgent:
    def __init__(self,
                 n_bins=6,
                 n_daily_ratio = 7,
                 n_weekly_ratio = 7,
                 n_hours=3,
                 n_actions=3,
                 alpha=0.3,
                 gamma=0.99,
                 epsilon=0.1):

        self.n_bins = n_bins
        self.n_daily_ratio = n_daily_ratio
        self.n_weekly_ratio = n_weekly_ratio
        self.n_hours = n_hours
        self.n_actions = n_actions
        
        # Learning rate
        self.alpha = alpha
        # Discount factor
        self.gamma = gamma
        # Epsilon for epsilon-greedy
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.Q = np.zeros((n_bins, n_daily_ratio, n_weekly_ratio, n_hours, n_actions), dtype=float)

    def get_action(self, bin_idx, daily_r_idx, weekly_r_idx, hour_idx):
        """
        Epsilon-greedy or take max
        """
        if random.random() < self.epsilon:
            if bin_idx > 0 and weekly_r_idx == 2 or daily_r_idx == 2:
                return 2
            if bin_idx == 0:
                return np.random.randint(self.n_actions - 1)
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[bin_idx, daily_r_idx, weekly_r_idx, hour_idx, :])
        
    def update(self,
               bin_idx, daily_r_idx, weekly_r_idx, hour_idx,
               action_idx,
               reward,
               next_bin_idx, next_daily_r_idx, next_weekly_r_idx, next_hour_idx,
               done):
        """
        Tabular Q-learning update
        """
        old_value = self.Q[bin_idx, daily_r_idx, weekly_r_idx, hour_idx, action_idx]

        if not done:
            future = np.max(self.Q[next_bin_idx, next_daily_r_idx, next_weekly_r_idx, next_hour_idx, :])
        else:
            future = 0.0

        new_value = old_value + self.alpha * (reward + self.gamma * future - old_value)
        self.Q[bin_idx, daily_r_idx, weekly_r_idx, hour_idx, action_idx] = new_value
