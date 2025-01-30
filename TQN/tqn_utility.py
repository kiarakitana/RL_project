import numpy as np
import pandas as pd
from collections import deque

class PriceTracker:
    def __init__(self, lookback_daily=24, lookback_2weeks=336, default=47.0):
        """
        Initialize price tracking with specified lookback periods and the default avg
        """
        self.daily_prices = deque(maxlen=lookback_daily)
        self.biweekly_prices = deque(maxlen=lookback_2weeks)

        self.daily_prices.append(default)
        self.biweekly_prices.append(default)
        
    def update(self, price):
        """Update averages if it's not an outlier"""
        if price/self.daily_avg >= 0.2 and price/self.daily_avg <= 2:
            self.daily_prices.append(price)
            self.biweekly_prices.append(price)
        
    @property
    def daily_avg(self):
        """Get daily moving average"""
        if len(self.daily_prices) > 0:
            return np.mean(self.daily_prices)
        return 47.0
        
    @property
    def biweekly_avg(self):
        """Get 2-week moving average"""
        if len(self.biweekly_prices) > 0:
            return np.mean(self.biweekly_prices)
        return 47.0

def small_storage_bin(storage_level, max_storage=170.0, req_storage=120.0):
    if storage_level <= 0:
        return 0
    if storage_level < req_storage:
        return 1
    if storage_level == req_storage:
        return 2
    if storage_level < max_storage:
        return 3
    if storage_level == max_storage:
        return 4
    else:
        return 5

def daily_avg_diff_bins(price, daily_avg):
    """Convert price to daily avg ratio to discrete bins"""
    bins = np.array([0, 0.5, 0.75, 1, 1.25, 1.5, 2])
    ratio = price / daily_avg
    return np.searchsorted(bins, ratio, side='right') - 1

def weekly_avg_diff_bins(price, weekly_avg):
    """Convert price to weekly avg ratio to discrete bins"""
    bins = np.array([0, 0.5, 0.75, 1, 1.25, 1.5, 2])
    ratio = price / weekly_avg
    return np.searchsorted(bins, ratio, side='right') - 1

def hour_bins(hour):
    """Convert hours to discrete bins"""
    bins = np.array([0, 8, 16])
    return np.searchsorted(bins, hour, side='right') - 1

def reward_function(storage, action, daily_r, weekly_r, hour, beta=0.8):
    price_advantage = beta * daily_r + (1 - beta) * weekly_r
    price_advantage = np.clip(price_advantage, 0, 2)

    reduce_selling = 2/3 if action == -1 else 1

    buy_early = 0.2 if hour == 0 and action == 1 else 0

    storage_bonus = action

    if storage >= 4 and action > 0:  # buying above the capacity
        storage_bonus = -5
    if storage == 5 and action == 0: # waiting above capacity
        storage_bonus = -5

    # Extra bonus for selling
    selling_high = 0
    if price_advantage >= 1.75 and action < 0:
        selling_high = 5

    # Extra bonus for buying low
    buying_low = 0
    if storage < 4 and action > 0 and price_advantage <= 0.75:
        buying_low = 2

    return -1 * action * reduce_selling * price_advantage + storage_bonus + selling_high + buying_low + buy_early
