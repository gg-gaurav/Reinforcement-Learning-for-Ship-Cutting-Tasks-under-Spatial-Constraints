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


# WE USE THIS SCRIPT TO CONTINUE TRAINING FROM A SPECIFIC POINT INSTEAD OF 
# TRAINING THE WHOLE MODEL AGAIN
# -----------------------------------------------------------
# Directories
timestep = 1749051964
checkpoint = 4750.0

checkpoint_path = os.path.join("models", f"DDPG-{timestep}",f"episode_{checkpoint}")

models_dir = os.path.join("models", f"DDPG-{timestep}-continued")
logs_dir = os.path.join("logs",f"DDPG-continued-{timestep}")


if not os.path.exists(checkpoint_path):
    print(f"Checkpoints path {checkpoint_path} doesn't exist.")

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Model directory created {models_dir}")

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    print(f"Logs directory created {logs_dir}")

# -----------------------------------------------------------
# Load Environment
env = pandaFollowTaskEnv()
max_steps = env.task.max_steps     
print(f"Max Steps per epsiode: {max_steps}")               

env = Monitor(env, filename= logs_dir)
env.reset()

## ---------------------------------------------------------
# Device Setup

if torch.cuda.is_available():
    device= "cuda"
    print(f"CUDA available")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    device= "cpu" 
    print(f"CUDA not available. Implementing on cpu")

## --------------------------------------------------------
# Load model from checkpoint

n_actions = env.action_space.shape[-1]
gaussianNoise = NormalActionNoise(mean=np.zeros(n_actions), 
                                sigma=0.2 * np.ones(n_actions))

model = DDPG.load(checkpoint_path, env=env, device=device)
model.action_noise = gaussianNoise  # re-attach action noise
print(f"Model loaded from checkpoint: {checkpoint_path}")

## -------------------------------------------------------
# Continue Training

log_callback = customCallback(update_freq= max_steps)               # graph updates every 10 episodesstart_time0 = time.time()

start_time0 = time.time()
train_time = []
TIMESTEPS = 50000                                                   # Partial Training: total steps = 10000 * 20 = 2e5
                                                                    # Full Training   : total steps = 50000 * 20 = 1e6
print(f"Continuing Training ....") 
for i in range(1,20):
    start_time1 = time.time()
    model.learn(total_timesteps = TIMESTEPS, 
                log_interval= 10, 
                tb_log_name= "DDPG-continued",
                reset_num_timesteps= False, 
                progress_bar= True,
                callback= log_callback)
    
    ep = (i * TIMESTEPS / 200) + checkpoint                         # Partial Training: total episodes = 200000/200 = 1000
    model.save(f"{models_dir}/episode_{ep}")                        # Full Training   : total episodes = 1e6/200    = 5000
    print(f"Model save checkpoint {i}: {models_dir}/episode_{ep}")

    end_time = time.time()
    train_time.append(end_time - start_time1)

print(f"Total time: {end_time - start_time0}")


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


plt.plot(train_time, label= "Train Times")
plt.xlabel("Set")
plt.ylabel("Time")
plt.title("Train Times")
plt.grid(True)
plt.show()