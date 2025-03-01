from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from performance_monitor import SystemPerformanceEnv

env = DummyVecEnv([lambda: SystemPerformanceEnv()])
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

model.save("ai_optimizer")
print("✅ AI model trained and saved.")
