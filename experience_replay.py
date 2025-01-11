from collections import deque
import random

class ReplayMemory():

    def __init__(self, maxlen, seed=None):
        self.memory = deque([], maxlen=maxlen)

        if seed is not None:
            random.seed(seed)
    
    def append(self, transition):
        self.memory.append(transition) # appends experience to memory

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size) # randomly sample memory and return whatever batch size selected
    
    def __len__(self):
        return len(self.memory) # length of the memory