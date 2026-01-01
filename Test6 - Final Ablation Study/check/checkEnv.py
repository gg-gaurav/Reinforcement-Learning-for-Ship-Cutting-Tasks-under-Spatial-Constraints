import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from pandaTaskEnv import pandaTaskEnv
#from templateTaskEnv import templateTaskEnv
import time


#env = pandaOrientTaskEnv(render_mode="human")
env = pandaTaskEnv(render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()                                      # random action
    print(f"Action : {action}")
    observation, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.075)
    #print(f"Observation : {observation}")
    #print(f"Reward : {reward}")
    #print(f"Terminated : {terminated}")
    #print(f"Truncated : {truncated}")
    if terminated or truncated:
        observation, info = env.reset()