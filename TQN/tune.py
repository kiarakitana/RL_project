from itertools import product
import numpy as np
import pickle
from tqn_main import run_training
from tqn_validate import validate

def main():
    hyperparams = {
        'gamma': [0.99, 0.95],
        'alpha': [0.1, 0.3, 0.5],
        'alfa_reward': [0.6, 0.8],
    }

    best_reward = np.inf
    best_hyperparams = {}

    for gamma, alpha, alfa in product(*hyperparams.values()):
        print(f"Testing: gamma={gamma}, alpha={alpha}, alfa={alfa}")
        
        class Args:
            def __init__(self, gamma, alpha, alfa_reward):
                self.path = 'train.xlsx'
                self.epochs = 120  # Reduced for quicker tuning; adjust as needed
                self.alpha = alpha
                self.gamma = gamma
                self.epsilon_decay = 0.98
                self.epsilon_min = 0
                self.alfa_reward = alfa_reward
                self.tuning = True

        args = Args(gamma, alpha, alfa)
        agent = run_training(args)
        reward = validate(agent=agent)
        
        if reward < best_reward:
            best_reward = reward
            best_hyperparams = {
                'gamma': gamma,
                'alpha': alpha,
                'alfa_reward': alfa,
            }
            print(f"New best reward: {best_reward}")
            try:
                with open('best_params.txt', 'w', encoding='utf-8') as file:
                    for key, value in best_hyperparams.items():
                        file.write(f"{key}: {value}\n")
                print(f"Dictionary successfully saved to best_params.txt")
            except Exception as e:
                print(f"An error occurred: {e}")
            # Save trained agent
            with open('best_tuned_agent.pkl', 'wb') as f:
                pickle.dump(agent, f)
            print("Agent saved in trained_agent.pkl")

    print("\nBest Hyperparameters:")
    for key, value in best_hyperparams.items():
        print(f"{key}: {value}")
    print(f"Best Reward: {best_reward}")

if __name__ == "__main__":
    main()
