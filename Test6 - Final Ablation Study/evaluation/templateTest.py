import os
import time
import random
import numpy as np
from tqdm.rich import tqdm
import gymnasium as gym

import torch
from stable_baselines3 import DDPG,PPO
from stable_baselines3.common.utils import set_random_seed

from templateTaskEnv import templateTaskEnv

# Move back one folder
import sys
import os
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------
# Set Seed for Reproducibility
# ---------------------------------------------------------

SEED= 42
np.random.seed(SEED)                # Seed for Numpy
random.seed(SEED)                   # Seed for python
set_random_seed(SEED)               # Seed for stable_baselines3

# Seed for PyTorch
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------------------------------------

env = templateTaskEnv(render_mode= "human")
env.reset(SEED)

### ------------------------------------- 
# Path to the model we want to load 
timestep =  1756458739
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
#print(f"Waypoint Reward: {env.task.waypointR}")
#print(f"Dwell Reward: {env.task.dwellR}")
#print(f"Velocity Reward: {env.task.velocityR}")
#print(f"Orientation Reward: {env.task.orientationR}")
#print(f"Bounds Reward: {env.task.boundsR}")
#print(f"Singularity Reward: {env.task.singularityR}")
#print(f"Success Reward: {env.task.successR}")


env.close()