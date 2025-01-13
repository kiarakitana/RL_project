import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import gym
from env import DataCenterEnv
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import random
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:

    def __init__(self,hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']   # size of replay memory
        self.mini_batch_size = hyperparameters['mini_batch_size']         # size of the training data set sampled from the replay memory
        self.epsilon_init = hyperparameters['epsilon_init']               # 1 = 100% random actions
        self.epsilon_decay = hyperparameters['epsilon_decay']             # epsilon decay rate
        self.epsilon_min = hyperparameters['epsilon_min']                 # minimum epsilon value
                
    
    def run(self, is_training=True):
        #env = DataCenterEnv(path_to_test_data='/Users/shivanikandhai/Documents/School/Artificial_Intelligence/Reinforcement Learning/train.xlsx')

        args = argparse.ArgumentParser()
        args.add_argument('--path', type=str, default='train.xlsx')
        args = args.parse_args()

        np.set_printoptions(suppress=True, precision=2)
        path_to_dataset = args.path

        env = DataCenterEnv(path_to_dataset)

        num_actions = 1 # continuous action
        num_states = 4 # 4 elements in a state 
        
        rewards_per_episode = []
        epsilon_history = []

        policy_dqn = DQN(num_states,num_actions).to(device)

        if is_training:
            memory = ReplayMemory(self.replay_memory_size)

            epsilon = self.epsilon_init

        
        for episode in itertools.count():
            state = env.observation()
            print("Current state:", state)
            state = torch.tensor(state, dtype=torch.float, device=device)

            terminated = False
            episode_reward = 0.0

            while not terminated:
                
                # next action: epsilon greedy
                if is_training and random.random() < epsilon:
                    action = env.continuous_action_space.sample()
                    action = torch.tensor(action, dtype=torch.float, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
                        #action = torch.tensor(action, dtype=torch.float, device=device)

                # processing: action is passed into step function to execute action
                new_state, reward, terminated = env.step(action.item())  # gives back: observation for what the next state, what the reward was, what happened to the agent (f/t), additional info for debugging

                
                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))
                
                # move to new state
                state = new_state

            rewards_per_episode.append(episode_reward)
            
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

if __name__ == '__main__':
    agent = Agent('datacenter')
    agent.run(is_training=True)