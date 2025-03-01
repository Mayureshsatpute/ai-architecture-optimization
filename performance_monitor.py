import gym
import numpy as np
import random

class SystemPerformanceEnv(gym.Env):
    def __init__(self):
        super(SystemPerformanceEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # 0: Low, 1: Medium, 2: High resource allocation
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)  # CPU, Memory, Latency
        self.state = np.array([50, 50, 50], dtype=np.float32)  # Initial workload
        self.done = False
    
    def step(self, action):
        if action == 0:  # Low allocation
            self.state[0] = max(0, self.state[0] - random.randint(5, 15))  # Prevent negative CPU
            self.state[1] = max(0, self.state[1] - random.randint(5, 15))  # Prevent negative Memory
        elif action == 1:  # Medium allocation
            self.state[0] = max(0, self.state[0] - random.randint(10, 20))
            self.state[1] = max(0, self.state[1] - random.randint(10, 20))
        else:  # High allocation
            self.state[0] = max(0, self.state[0] - random.randint(15, 25))
            self.state[1] = max(0, self.state[1] - random.randint(15, 25))
        
        self.state[2] = max(0, self.state[2] - random.randint(5, 10))  # Prevent negative latency
        reward = -abs(self.state[2] - 20)  # Reward closer to 20ms latency
        self.done = any(s <= 0 for s in self.state)
        return self.state, reward, self.done, {}
    
    def reset(self):
        self.state = np.array([50, 50, 50], dtype=np.float32)
        self.done = False
        return self.state
