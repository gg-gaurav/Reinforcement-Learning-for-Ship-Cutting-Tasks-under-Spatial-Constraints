import os
import time
import random
import numpy as np
from tqdm.rich import tqdm
import gymnasium as gym

import torch
from stable_baselines3 import DDPG,PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.utils import set_random_seed

from templateTaskEnv import templateTaskEnv

# Move back one folder -----------------------------------
import sys
import os
# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir('..')
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

# ---------------------------
# Model and normalization paths
# ---------------------------
timestep = 1755790144
checkpoint = 20000
base_dir = f"models_v2\DDPG-{timestep}"
#base_dir = f"models_v2/PPO-{timestep}
model_path = os.path.join(base_dir,f"episode_{checkpoint}.zip")
vecnorm_path = os.path.join(base_dir,"norm_stats","vecnormalize.pkl")  # <- Update if different

print(f"Model path     : {model_path}")
print(f"VecNormalize   : {vecnorm_path}")

# ---------------------------
# Create vectorized test env and load normalization stats
# ---------------------------
def make_test_env():
    env = templateTaskEnv(render_mode="human")
    env.reset(SEED)
    return env

dummy_env = DummyVecEnv([make_test_env])

if os.path.exists(vecnorm_path):
    vec_env= VecNormalize.load(vecnorm_path, dummy_env)
    vec_env.training = False                        # Don't update stats at test time
    vec_env.norm_reward = False                     # Optional: see true rewards
else:
    print("[WARNING] VecNormalize not found. Proceeding without normalization")
    vec_env = dummy_env

# ---------------------------
# Load model with normalized env
# ---------------------------
model = DDPG.load(model_path, env=vec_env)
#model = PPO.load(model_path, env=vec_env)

# ---------------------------
# Load model with normalized env
# ---------------------------
model = DDPG.load(model_path, env=vec_env)

# ---------------------------
# Run test episodes
# ---------------------------
episodes = 10

for ep in range(1, episodes + 1):
    obs = vec_env.reset()
    done = False

    print(f"Starting Episode {ep}")
    for _ in tqdm(range(201)):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated = vec_env.step(action)
        #print(obs)
        #print(reward)
        #print(terminated)
        #print(truncated)
        time.sleep(0.07)

        done = terminated
        #print(f"done: {done}")
        if done:
            break

vec_env.close()