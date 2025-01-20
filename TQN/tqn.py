import numpy as np
import random


class QLearningAgent:
    def __init__(self,
                 n_bins=19,
                 n_prices = 14,
                 n_hours=24,
                 n_dow=7,
                 n_month=12,
                 n_actions=3,
                 alpha=0.3,
                 gamma=0.99,
                 epsilon=0.1):
        """
        A simple tabular Q-learning agent with:
          Q[storage_bin, hour, action]
        """
        self.n_bins = n_bins
        self.n_prices = n_prices
        self.n_hours = n_hours
        self.n_dow = n_dow
        self.n_month = n_month
        self.n_actions = n_actions
        
        # Learning rate
        self.alpha = alpha
        # Discount factor
        self.gamma = gamma
        # Epsilon for epsilon-greedy
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.Q = np.zeros((n_bins, n_prices, n_hours, n_dow, n_month, n_actions), dtype=float)

    def get_action(self, bin_idx, price_idx, hour_idx, dow_idx, month_idx):
        """
        Epsilon-greedy over the 5D state.
        """
        if random.random() < self.epsilon:
            if price_idx == 13:
                return 2

            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[bin_idx, price_idx, hour_idx, dow_idx, month_idx, :])
        
    def update(self,
               bin_idx, price_idx, hour_idx, dow_idx, month_idx,
               action_idx,
               reward,
               next_bin_idx, next_price_idx, next_hour_idx, next_dow_idx, next_month_idx,
               done):
        """
        Tabular Q-learning update:
          Q(s,a) = (1-alpha)*Q(s,a) + alpha*(r + gamma*max_a' Q(s', a'))
        """
        old_value = self.Q[bin_idx, price_idx, hour_idx, dow_idx, month_idx, action_idx]

        if not done:
            future = np.max(self.Q[next_bin_idx, next_price_idx, next_hour_idx, next_dow_idx, next_month_idx, :])
        else:
            future = 0.0

        new_value = old_value + self.alpha * (reward + self.gamma * future - old_value)
        self.Q[bin_idx, price_idx, hour_idx, dow_idx, month_idx, action_idx] = new_value
