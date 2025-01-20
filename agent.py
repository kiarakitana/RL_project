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

'''
## PABLO: REWARD FUNCTION


if price is high sell

if price low buy anyway, but not more than 170

if you dont have your req storage buy but try to buy low


## SEM

2 Reward functions
1 for buying
1 for selling

We normalise all the prices of the energy to 0-20

For buying let's say the base reward is 10
If the normalised price is 10 the reward will be:
10-10 = 0
If the normalised price is 20 the reward will be:
10-20=-10
Etc.

For selling we can use this, but inverted.
The base reward will be -10
If the normalised price is 10 the reward will be:
-10+10=0
If the normalised price is 20 the reward will be:
-10+20=10
'''

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
        self.network_sync_rate = hyperparameters['network_sync_rate']        
        self.learning_rate_a = hyperparameters['learning_rate_a']         # learning rate for the actor
        self.discount_factor_g = hyperparameters['discount_factor_g']     # discount factor
        self.loss_fn = nn.MSELoss()                                       # MSE loss function
        self.optimizer = None



    def run(self, is_training=True):
        
        # initializing environment
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

            target_dqn = DQN(num_states,num_actions).to(device) # target network
            target_dqn.load_state_dict(policy_dqn.state_dict()) # copies all weights and biases of the policy network to the target network

            # track number of steps taken to sync policy and target network
            step_count = 0

            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a) # adam optimizer


        for episode in itertools.count():
        #for episode in range(15):
            state = env.observation()
            print("Current state:", state)
            state = torch.tensor(state, dtype=torch.float, device=device)
            
            terminated = False
            episode_reward = 0.0
            

            
            if not terminated:
                
                # choose next action: epsilon greedy
                if is_training and random.random() < epsilon:
                    action = env.continuous_action_space.sample()
                    action = torch.tensor(action, dtype=torch.float, device=device)    # use this for reward shaping
                    print(f'Random action chosen:{action}')
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()     # use this for reward shaping
                        action = torch.tensor(action, dtype=torch.float, device=device)
                        print(f'Nonrandom action chosen: {action}')


                # execute action, receive new state, reward, terminated or not
                new_state, reward, terminated = env.step(action.item())  
                

                '''
                implement reward shaping
                '''



                episode_reward += reward # this is the part where we can do reward shaping ??

                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device) # this is the part where we can do reward shaping ?

                if is_training:
                    memory.append((state, action, new_state, reward, terminated))

                    step_count += 1
                
                # move to new state
                state = new_state

                rewards_per_episode.append(episode_reward)
            
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                epsilon_history.append(epsilon)

                # if enough experience collected: sync policy and target networks
                if len(memory)>self.mini_batch_size:

                    # sample mini batch from replay memory
                    mini_batch = memory.sample(self.mini_batch_size)

                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # update target network after certain number of steps: copy policy network to target network
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

                #print("Action:", action)
                #print("Next state:", new_state)
                print("Reward:", reward)
                print()
        

    # optimizing policy network 
    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                target_q = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * target_dqn(new_state).max()
            
            current_q = policy_dqn(state)

            loss = self.loss_fn(current_q, target_q)

            self.optimizer.zero_grad()                  # clear gradients
            loss.backward()                             # backprop
            self.optimizer.step()                       # update network parameters

        #print(f'rewards per episode:{rewards_per_episode}')
        # TOTAL REWARD OUTPUT

if __name__ == '__main__':
    agent = Agent('datacenter')
    agent.run(is_training=True)