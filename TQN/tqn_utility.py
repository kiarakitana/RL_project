import numpy as np
import pandas as pd
from collections import deque

class PriceTracker:
    def __init__(self, lookback_daily=24, lookback_2weeks=336):
        """
        Initialize price tracking with specified lookback periods
        24 hours = 1 day
        336 hours = 14 days (2 weeks)
        """
        self.daily_prices = deque(maxlen=lookback_daily)
        self.biweekly_prices = deque(maxlen=lookback_2weeks)
        
    def update(self, price):
        """Add new price and update averages"""
        self.daily_prices.append(price)
        self.biweekly_prices.append(price)
        
    @property
    def daily_avg(self):
        """Get daily moving average"""
        if len(self.daily_prices) > 0:
            return np.mean(self.daily_prices)
        return 50.0  # Default value if no history
        
    @property
    def biweekly_avg(self):
        """Get 2-week moving average"""
        if len(self.biweekly_prices) > 0:
            return np.mean(self.biweekly_prices)
        return 50.0  # Default value if no history

# Global price tracker instance
price_tracker = PriceTracker()

def build_day_maps(path_to_dataset):
    """
    Reads the same spreadsheet as the environment,
    then builds arrays mapping day -> day_of_week, day -> month_of_year.
    day_of_week in [0..6], month_of_year in [0..11].
    """
    df = pd.read_excel(path_to_dataset)  # same as in env
    timestamps = df['PRICES']           # day time stamps dd/mm/yyyy

    dow_map = []
    month_map = []

    for i in range(len(timestamps)):
        date = pd.to_datetime(timestamps[i], dayfirst=True)
        dow_map.append(date.weekday())     # Monday=0, Sunday=6
        month_map.append(date.month - 1)   # 0..11

    return np.array(dow_map), np.array(month_map)

def storage_to_bin(storage_level, bin_size=10.0, max_storage=170.0):
    """Convert continuous storage_level to integer bin index"""
    if storage_level < 0:
        return 0
    if storage_level >= max_storage:
        return 18
    bin_index = int(storage_level // bin_size)
    return min(bin_index, 18)

def discrete_action_to_continuous(action_idx):
    """Map discrete action space to continuous values"""
    if action_idx == 0:
        return 0.0    # hold
    elif action_idx == 1:
        return 1.0    # buy
    elif action_idx == 2:
        return -1.0   # sell

def price_bins(price):
    """Convert continuous price to discrete bins"""
    bins = np.array([0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 75, 100, 200])
    return np.searchsorted(bins, price, side='right') - 1

def reward_function(env_reward, storage, action, price):
    """
    Enhanced reward function incorporating price averages
    Args:
        env_reward: original environment reward
        storage: current storage bin index
        action: continuous action value (-1, 0, 1)
        price: current price
    """
    # Update price tracker
    price_tracker.update(price)
    daily_avg = price_tracker.daily_avg
    biweekly_avg = price_tracker.biweekly_avg
    
    # Base reward components
    new_reward = 0.0
    
    # Buying strategy (action > 0)
    if action > 0:
        # Penalize buying when storage is high
        if storage >= 17:
            new_reward = -10
        else:
            # Buy reward based on price comparison with averages
            price_diff_daily = (daily_avg - price) / daily_avg
            price_diff_biweekly = (biweekly_avg - price) / biweekly_avg
            
            # Weighted combination of price differences
            price_advantage = 0.7 * price_diff_daily + 0.3 * price_diff_biweekly
            
            if storage < 12:  # Low storage: more urgent to buy
                new_reward = 5 * price_advantage + 4
            else:  # Normal buying
                new_reward = 5 * price_advantage
    
    # Selling strategy (action < 0)
    elif action < 0:
        # Strong sell signals
        if price >= 4 * biweekly_avg:
            new_reward = 10
        elif storage == 18:  # Must sell when storage full
            new_reward = 10
        elif storage < 12:  # Penalize selling at low storage
            new_reward = -5
        else:
            # Sell reward based on price comparison with averages
            price_diff_daily = (price - daily_avg) / daily_avg
            price_diff_biweekly = (price - biweekly_avg) / biweekly_avg
            
            # Weighted combination of price differences
            price_advantage = 0.7 * price_diff_daily + 0.3 * price_diff_biweekly
            new_reward = 5 * price_advantage - 2
    
    # Holding strategy (action == 0)
    else:
        if storage == 17:
            new_reward = 5  # Good to hold near capacity
        elif storage == 18:
            new_reward = -10  # Penalize holding at full capacity
        else:
            # Hold reward based on price trends
            price_diff_daily = (daily_avg - price) / daily_avg
            price_diff_biweekly = (biweekly_avg - price) / biweekly_avg
            
            # If current price is significantly below averages, penalize holding
            price_advantage = 0.7 * price_diff_daily + 0.3 * price_diff_biweekly
            if storage >= 12:
                new_reward = -3 * price_advantage + 2
            else:
                new_reward = -3 * price_advantage
    
    # Clip final reward
    return np.clip(new_reward, -10, 10)