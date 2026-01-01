import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG
import numpy as np
import time
from tqdm.rich import tqdm


env = gym.make("PandaReach-v3",render_mode= "human")
env.reset()

### ------------------------------------- 
# Path to the model we want to load 
i = 1
model_dir = f"basicRun_{i}"
print(f"model dir: {model_dir}")
### --------------------------------------

model = DDPG.load(model_dir)
episodes = 10

for ep in range(1, episodes+1):

    obs, _ = env.reset()
    done = False

    for j in tqdm(range(200)):
        """We run for a maximum of 200 steps for each episode"""

        action = [1,1,1]
        # action, _states = model.predict(obs)
        #print(f"Predicted Action : {action}")

        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(1)
        
        done = terminated or truncated
        if done:
            time.sleep(5)
            break

env.close()