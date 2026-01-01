# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 13:08:29 2025

@author: Gaurav
"""

import gymnasium as gym
from stable_baselines3 import DDPG,PPO
import os
import numpy as np
import time
from tqdm.rich import tqdm

from pandaOrientTaskEnv import pandaOrientTaskEnv

env = pandaOrientTaskEnv(render_mode= "human")
env.reset()

### ------------------------------------- 
# Path to the model we want to load 
timestep = 1753955580
checkpoint = 10000
model_dir = os.path.join("models_v2", f"DDPG-{timestep}",f"episode_{checkpoint}.zip")
#model_dir = os.path.join("models_v2", f"PPO-{timestep}",f"episode_{checkpoint}.zip")
print(f"model dir: {model_dir}")
### --------------------------------------

model = DDPG.load(model_dir)
#model = PPO.load(model_dir)
episodes = 10

for ep in range(1, episodes+1):

    obs, _ = env.reset()
    done = False

    for j in tqdm(range(201)):
        """We run for a maximum of 200 steps for each episode"""

        action, _states = model.predict(obs)
        #print(f"Predicted Action : {action}")

        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.07)
        
        done = terminated or truncated
        if done:
            break
    
# Reward Store
print(f"Waypoint Reward: {env.task.waypointR}")
#print(f"Velocity Reward: {env.task.velocityR}")
#print(f"Bounds Reward: {env.task.boundsR}")
#print(f"Orientation Reward: {env.task.orientationR}")
#print(f"Singularity Reward: {env.task.singularityR}")
#print(f"Success Reward: {env.task.successR}")
print(f"Dwell Reward: {env.task.dwellR}")

env.close()
