from itertools import product
import numpy as np
from tqn_main import run_training
from tqn_validate import validate

def main():
    hyperparams = {
        'gamma': [0.99, 0.95, 0.8],
        'alpha': [0.1, 0.2, 0.3, 0.5],
        'alfa_reward': [0.6, 0.75, 0.8],
        'reduce_selling_value': [0.5, 2/3, 0.8]
    }

    best_reward = -np.inf
    best_hyperparams = {}

    for gamma, alpha, alfa, reduce_sell in product(*hyperparams.values()):
        print(f"Testing gamma={gamma}, alpha={alpha}, alfa={alfa}, reduce_sell={reduce_sell}")
        
        class Args:
            def __init__(self, gamma, alpha, alfa_reward, reduce_selling_value):
                self.path = 'train.xlsx'
                self.epochs = 120  # Reduced for quicker tuning; adjust as needed
                self.alpha = alpha
                self.gamma = gamma
                self.epsilon_decay = 0.98
                self.epsilon_min = 0
                self.alfa_reward = alfa_reward
                self.reduce_selling_value = reduce_selling_value
                self.tuning = True

        args = Args(gamma, alpha, alfa, reduce_sell)
        agent = run_training(args)
        reward = validate(agent=agent)
        
        if reward < best_reward:
            best_reward = reward
            best_hyperparams = {
                'gamma': gamma,
                'alpha': alpha,
                'alfa_reward': alfa,
                'reduce_selling_value': reduce_sell
            }
            print(f"New best reward: {best_reward}")

    print("\nBest Hyperparameters:")
    for key, value in best_hyperparams.items():
        print(f"{key}: {value}")
    print(f"Best Reward: {best_reward}")

if __name__ == "__main__":
    main()
