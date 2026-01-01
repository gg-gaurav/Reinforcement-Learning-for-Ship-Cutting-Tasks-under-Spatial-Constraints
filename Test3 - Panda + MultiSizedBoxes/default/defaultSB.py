import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG

env = gym.make("PandaReach-v3")
i = 1
model_name = f"basicRun_{i}"
model = DDPG(policy= "MultiInputPolicy", env=env)
model.learn(30000, progress_bar= True)

model.save(model_name)