import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import gym
from env import DataCenterEnv

# build the model wrapped inside of a function

env = DataCenterEnv(path_to_test_data='/Users/shivanikandhai/Documents/School/Artificial_Intelligence/Reinforcement Learning/train.xlsx')
states = env.observation().shape[0]
actions = env.continuous_action_space.shape[0]
print(f'{states} states, {actions} actions')

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(states, actions)

model.summary()