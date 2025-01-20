import numpy as np
import pandas as pd

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
    """
    Convert a continuous storage_level to an integer bin index in [0..18].
    Everything above max_storage (170 MWh) is considered bin 18.
    """
    if storage_level < 0:
        # Just in case numerical issues or environment mistakes,
        # but environment code prevents going below 0 anyway.
        return 0
    
    if storage_level >= max_storage:
        # consider everything >= 170 in the top bin
        return 18
    
    # e.g. if storage=0..9.999 => bin 0, 10..19.999 => bin 1, ...
    bin_index = int(storage_level // bin_size)
    return min(bin_index, 18)

def bin_to_storage(bin_index, bin_size=10.0):
    """
    Optional: If you want to interpret the bin center, you could do:
    storage = bin_index * bin_size + bin_size/2
    But usually we only do the forward mapping (storage->bin), not needed here.
    """
    return bin_index * bin_size

def discrete_action_to_continuous(action_idx):
    """
    Map our discrete action space {0,1,2} to environment's continuous [-1,1].
      0 -> hold (0)
      1 -> buy  ( +1 )  => +10MW
      2 -> sell ( -1 )  => -10MW
    """
    if action_idx == 0:
        return 0.0
    elif action_idx == 1:
        return 1.0
    elif action_idx == 2:
        return -1.0
    
def price_bins(price):
    bins = np.array([0, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 75, 100, 200])
    return np.searchsorted(bins, price, side='right') - 1

def reward_function(reward, storage, action, price):
    """
    Do it with rolling avg of 2 weeks maybe
    """
    avg_price = 50
    # buying
    if action > 0:
        if storage >= 17:
            new_reward = -10
        if storage < 12:
            new_reward = (avg_price - price) / 10 + 4
        else:
            new_reward = (avg_price - price) / 10
    # selling
    if action < 0:
        if price >= 4 * avg_price:
            new_reward = 10
        if storage == 18:
            new_reward = 10
        if storage < 12:
            new_reward = -5
        else:
            new_reward = reward / 10 - 10
    # waiting
    if action == 0:
        if storage == 17:
            new_reward = 5
        if storage == 18:
            new_reward = -10
        if storage >= 12:
            new_reward = -(avg_price - price) / 10 + 2
        if storage < 12:
            new_reward = -(avg_price - price) / 10
    return np.clip(new_reward, -10, 10)
