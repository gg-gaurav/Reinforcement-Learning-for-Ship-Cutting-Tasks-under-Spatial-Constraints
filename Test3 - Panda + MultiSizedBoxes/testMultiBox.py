# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 13:08:29 2025

@author: Gaurav
"""

import gymnasium as gym
from stable_baselines3 import DDPG
import os
import numpy as np
import time
from tqdm.rich import tqdm

from pandaMultiSizedBoxesTaskEnv import pandaMultiSizedBoxesTaskEnv

env = pandaMultiSizedBoxesTaskEnv(render_mode= "human")
env.reset()

### ------------------------------------- 
# Path to the model we want to load 
timestep = 1745968640
checkpoint = 4000.0
model_dir = os.path.join("models", f"DDPG-{timestep} (OG)",f"episode_{checkpoint}")
print(f"model dir: {model_dir}")
### --------------------------------------

model = DDPG.load(model_dir)
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

env.close()
