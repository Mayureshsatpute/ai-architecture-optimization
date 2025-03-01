from stable_baselines3 import PPO
from performance_monitor import SystemPerformanceEnv
import numpy as np

# Load trained AI model
model = PPO.load("ai_optimizer")
env = SystemPerformanceEnv()
obs = env.reset()

print("Optimizing software architecture...")
for _ in range(10):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    print(f"Action: {action}, State: {obs}, Reward: {reward}")
    if done:
        break
