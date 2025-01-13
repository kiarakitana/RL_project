import numpy as np
from collections import defaultdict

class TabularQAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon_init=1.0, 
                 epsilon_decay=0.9995, epsilon_min=0.05):
        # Q-table: Use defaultdict to handle state-action pairs lazily
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Discretization parameters
        self.storage_bins = np.linspace(0, 170, 18)  # 0 to max (120 + 50) in 10 MWh steps
        self.price_bins = np.array([0, 25, 50, 75, 100, 150, 200, 300])  # Price brackets
        self.action_bins = np.linspace(-1, 1, 11)  # 11 discrete actions including -1, 0, 1
        
    def discretize_state(self, state):
        """Convert continuous state to discrete state"""
        storage, price, hour, day = state
        
        # Discretize each component
        storage_idx = np.digitize(storage, self.storage_bins)
        price_idx = np.digitize(price, self.price_bins)
        
        # Hour and day are already discrete
        # Return tuple for hashable state
        return (storage_idx, price_idx, int(hour), int(day))
    
    def discretize_action(self, action):
        """Convert continuous action to closest discrete action"""
        return self.action_bins[np.abs(self.action_bins - action).argmin()]
    
    def get_action(self, state, validation=False):
        """Epsilon-greedy action selection, with greedy selection during validation"""
        discrete_state = self.discretize_state(state)
        
        if not validation and np.random.random() < self.epsilon:
            # Random action (only during training)
            return np.random.choice(self.action_bins)
        
        # Greedy action
        q_values = self.q_table[discrete_state]
        if not q_values:  # If state never seen before
            return np.random.choice(self.action_bins)
            
        return max(q_values.items(), key=lambda x: x[1])[0]
    
    def update(self, state, action, reward, next_state):
        """Update Q-value using Q-learning update rule"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        discrete_action = self.discretize_action(action)
        
        self.update_q_table(discrete_state, discrete_action, reward, discrete_next_state)
        
    def update_q_table(self, discrete_state, discrete_action, reward, discrete_next_state):
        """Update Q-table with pre-discretized states and actions"""
        # Get best next action value
        next_q_values = self.q_table[discrete_next_state]
        next_max_q = max(next_q_values.values()) if next_q_values else 0
        
        # Update rule
        current_q = self.q_table[discrete_state][discrete_action]
        new_q = current_q + self.lr * (
            reward + self.gamma * next_max_q - current_q
        )
        self.q_table[discrete_state][discrete_action] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        
    def save(self, filepath):
        """Save Q-table to file"""
        # Convert defaultdict to regular dict for saving
        q_dict = {str(k): dict(v) for k, v in self.q_table.items()}
        np.save(filepath, q_dict)
        
    def load(self, filepath):
        """Load Q-table from file"""
        q_dict = np.load(filepath, allow_pickle=True).item()
        # Convert back to defaultdict
        self.q_table = defaultdict(lambda: defaultdict(float))
        for k, v in q_dict.items():
            state = eval(k)  # Convert string tuple back to tuple
            self.q_table[state] = defaultdict(float, v)