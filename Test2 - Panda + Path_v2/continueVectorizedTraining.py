if __name__ == "__main__":

    import numpy as np
    import os 
    import time
    import gymnasium as gym
    import torch
    import matplotlib.pyplot as plt

    from pandaFollowTaskEnv import pandaFollowTaskEnv
    from vectorizedCustomCallback import vectorizedCustomCallback

    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv

    ## ----------------------------------------------
    # Check for CUDA

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"CUDA available")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print(f"CUDA not available. Implementing on cpu")

    # -----------------------------------------------------------
    # Directories
    timestep = 1749635610
    checkpoint = 4750.0

    checkpoint_path = os.path.join("models", f"DDPG-{timestep}",f"episode_{checkpoint}")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoints path {checkpoint_path} doesn't exist.")


    models_dir = os.path.join("models", f"DDPG-{timestep}-continued")
    logs_dir = os.path.join("logs",f"DDPG-continued-{timestep}")

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Model directory created {models_dir}")

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Logs directory created {logs_dir}")

    ## ----------------------------------------------
    # Function to create Dummy envs
    def make_env(log_file_name):
        def _init():
            env = pandaFollowTaskEnv()
            filename = os.path.join(logs_dir, log_file_name)

            os.makedirs(logs_dir, exist_ok=True)
            env = Monitor(env, filename=filename)
            return env
        return _init    

    # Creating the vectorized Envs: 4 env running parallely
    vec_env = SubprocVecEnv([make_env(log_file_name=f"env-{i}") for i in range(4)])

    ## ----------------------------------------------
    # Dummy Env for max_steps
    dummy_env = pandaFollowTaskEnv()
    max_steps = dummy_env.task.max_steps
    del dummy_env

    ## ----------------------------------------------
    # Hyperparameters

    policy_kwargs = {'net_arch': [256, 256, 256]}
    tau = 0.05                                          # (Tau = 1 - PolyackAveraging)
    learning_rate = 1e-3
    batch_size = 256
    buffer_size = int(1e6)

    n_actions = vec_env.action_space.shape[-1]
    gaussianNoise = NormalActionNoise(mean= np.zeros(n_actions), sigma= 0.2 * np.ones(n_actions))

    ## ------------------------------------------------------------
    # Load Model from Checkpoint
    
    model = DDPG.load(checkpoint_path, env=vec_env, device=device)
    model.action_noise = gaussianNoise
    print(f"Loaded model from checkpoint: {checkpoint_path}")

    log_callback = vectorizedCustomCallback(update_freq= max_steps)               # graph updates every 10 episodes

    # -------------------------------------------------------------
    TIMESTEPS = 50000                                                   # Partial Training: total steps = 10000 * 20 = 2e5
                                                                        # Full Training   : total steps = 50000 * 20 = 1e6

    print(f"Continue Training ....") 
    
    start_time0 = time.time()
    train_time = []
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

    rewards = log_callback.episodic_rewards
    actor_losses = log_callback.actor_losses                # Actor and Critic Losses for each step (not each episode)
    critic_losses = log_callback.critic_losses              

    vec_env.close()


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