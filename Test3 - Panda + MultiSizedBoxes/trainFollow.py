import numpy as np
import os
import time
import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from pandaFollowTaskEnv import pandaFollowTaskEnv
from customCallback import customCallback

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor


## ----------------------------------------------
# Directories
models_dir = f"models/DDPG-{int(time.time())}"
logs_dir = f"logs/DDPG-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Model directory created {models_dir}")

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    print(f"Logs directory created {logs_dir}")

## ----------------------------------------------
# Load Environment
env = pandaFollowTaskEnv()
max_steps = env.task.max_steps     
print(f"Max Steps per epsiode: {max_steps}")               

env = Monitor(env, filename= logs_dir)
env.reset()

## ----------------------------------------------
if torch.cuda.is_available():
    device= "cuda"
    print(f"CUDA available")
else:
    device= "cpu" 
    print(f"CUDA not available. Implementing on cpu")

## ----------------------------------------------
# Hyperparameters

policy_kwargs = {'net_arch': [256, 256, 256]}
tau = 0.05                                          # (Tau = 1 - PolyackAveraging)
learning_rate = 1e-3
batch_size = 256
buffer_size = int(1e6)

n_actions = env.action_space.shape[-1]
gaussianNoise = NormalActionNoise(mean= np.zeros(n_actions), sigma= 0.2 * np.ones(n_actions))

# --------------------------------------------------------------
model = DDPG("MultiInputPolicy", 
             env, 
             action_noise= gaussianNoise,
             policy_kwargs= policy_kwargs,
             tau= tau,
             learning_rate= learning_rate,
             batch_size= batch_size,
             buffer_size= buffer_size,
             verbose= 1,
             seed = 42, 
             device = device,
             tensorboard_log= logs_dir)

log_callback = customCallback(update_freq= max_steps)               # graph updates every 10 episodes

TIMESTEPS = 10000                                                   # Partial Training: total steps = 10000 * 20 = 2e5
                                                                    # Full Training   : total steps = 50000 * 20 = 1e6
print(f"Training ....") 
for i in range(1,20):
    model.learn(total_timesteps = TIMESTEPS, 
                log_interval= 10, 
                tb_log_name= "DDPG",
                reset_num_timesteps= False, 
                progress_bar= True,
                callback= log_callback)
    
    ep = i * TIMESTEPS / 200                                        # Partial Training: total episodes = 200000/200 = 1000
    model.save(f"{models_dir}/episode_{ep}")                        # Full Training   : total episodes = 1e6/200    = 5000
    print(f"Model save checkpoint {i}: {models_dir}/episode_{ep}")

del model
plt.ioff()
plt.show()

rewards = env.get_episode_rewards()
episode_lengths = env.get_episode_lengths()

actor_losses = log_callback.actor_losses                # Actor and Critic Losses for each step (not each episode)
critic_losses = log_callback.critic_losses              

env.close()


# ----- Plots ----------------------
plt.plot(rewards, label= "Rewards")
plt.xlabel("Episode")
plt.title("Reward vs Episode")
plt.grid(True)
plt.show()


plt.plot(episode_lengths, label= "Episode Length")
plt.xlabel("Episode")
plt.title("Episode_Length vs Episode")
plt.grid(True)
plt.show()


