if __name__ == "__main__":

    import numpy as np
    import os 
    import random
    import time
    import gymnasium as gym
    import torch
    import matplotlib.pyplot as plt
    import mlflow

    from base_options import parse_args
    from pandaTaskEnv import pandaTaskEnv
    from vectorizedCallbackMLFlow import vectorizedCallbackMLFlow

    from stable_baselines3 import DDPG, PPO
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.utils import set_random_seed


    ## ---------------------------------------------
    # Setting Seed

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

    ## ----------------------------------------------
    # Check for CUDA
    if torch.cuda.is_available():
        device= "cuda"
        print(f"CUDA available")
    else:
        device= "cpu" 
        print(f"CUDA not available. Implementing on cpu")

    ## ----------------------------------------------
    # Directories
    opt = parse_args()
    algo = opt.algo

    timestamp = int(time.time())
    models_dir = f"models_v2/{algo}-{timestamp}"
    logs_dir = f"logs_v2/{algo}-{timestamp}"

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
            env = pandaTaskEnv()
            
            filename = os.path.join(logs_dir, log_file_name)
            os.makedirs(logs_dir, exist_ok=True)
            env = Monitor(env, filename=filename)

            env.reset()
            return env
        return _init    

    
    
    # Creating the vectorized Envs: 32 env running parallely
    n_envs = 32
    raw_env = SubprocVecEnv([make_env(log_file_name=f"env-{i}") for i in range(n_envs)])
    
    # Wrap with VecNormalize
    vec_env = VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs= 10.0)


    ## ----------------------------------------------
    # Dummy Env
    dummy_env = pandaTaskEnv()
    max_steps = dummy_env.task.max_steps
    del dummy_env

    ## ----------------------------------------------
    # Hyperparameters

    policy_kwargs = {
        'net_arch': [512, 512, 256],                        # For both PPO and DDPG
        'activation_fn': torch.nn.ReLU                      # Needed for PPO
        }
    tau = opt.tau                                           # (Tau = 1 - PolyackAveraging)
    learning_rate = opt.learning_rate
    batch_size = opt.batch_size
    buffer_size = opt.buffer_size

    n_actions = vec_env.action_space.shape[-1]
    gaussianNoise = NormalActionNoise(mean= np.zeros(n_actions), sigma= opt.noise_sigma * np.ones(n_actions))

    ## --------------------------------------------------------------
    # MLFlow Logging

    print(f"MLFlow experiment name: {opt.experiment_name}")
    mlflow.set_experiment(opt.experiment_name)
    
    print(f"Started MLFlow logging")
    opt.run_name =f"{algo}-{timestamp}"

    with mlflow.start_run(run_name = opt.run_name):

        tag = input("Info about this run: ")
        mlflow.set_tag("training_info", tag)
        
        # Reward Characteristics --------------------------------------------------------------- MAKE CHANGES HERE BEFORE EVERY RUN 
        mlflow.log_params({"Navigation Reward":"Sparse + Shaped Reward",
                           "Velocity Reward": True,
                           "Bounded Navigation Reward": "Dense + Shaped Reward",
                           "Orientation Reward": False})
        

        if algo == "DDPG": 
            # Log hyperparameters
            mlflow.log_params({
            "algorithm": "DDPG",
            "tau": tau,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "net_arch": policy_kwargs["net_arch"],
            "n_envs": n_envs,
            "action_noise_sigma": 0.2,
            "device": device,
            "obs_norm": True,
            "reward_norm": True,
            "clip_obs": 10.0,
            "episodes": opt.timesteps * opt.sets / 200
            })

            # DDPG Model
            model = DDPG("MultiInputPolicy", 
                        vec_env, 
                        action_noise= gaussianNoise,
                        policy_kwargs= policy_kwargs,
                        tau= tau,
                        learning_rate= learning_rate,
                        batch_size= batch_size,
                        buffer_size= buffer_size,
                        verbose= 1,
                        seed = 42, 
                        device = device,
                        tensorboard_log= logs_dir
                        )
            
        elif algo == "PPO":
            
            mlflow.log_params({
            "algorithm": "PPO",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "net_arch": policy_kwargs["net_arch"],
            "n_envs": n_envs,
            "device": device,
            "obs_norm": True,
            "reward_norm": True,
            "clip_obs": 10.0,
            "episodes": opt.timesteps * opt.sets / 200
            })

            model = PPO(
                        policy="MultiInputPolicy",     
                        env=vec_env,
                        learning_rate=learning_rate,    # actor learning rate (you can set this globally)
                        n_steps=2048,                   # rollout length per env
                        batch_size=batch_size,          # minibatch size
                        n_epochs=10,                    # update the policy 10 times per rollout
                        gamma=0.99,                     # discount factor
                        gae_lambda=0.95,                # GAE parameter (default and standard)
                        clip_range=0.2,                 # PPO clipping epsilon
                        ent_coef=0.0,                   # entropy coefficient (encourages exploration)
                        vf_coef=0.5,                    # value loss weight
                        max_grad_norm=0.5,              # gradient clipping
                        policy_kwargs=policy_kwargs,
                        verbose=1,
                        seed=42,
                        device=device,
                        tensorboard_log=logs_dir
                        )
           

        log_callback = vectorizedCallbackMLFlow(steps_per_ep= max_steps,
                                                episodic_window = opt.episodic_window,
                                                algo = opt.algo)        


        TIMESTEPS = opt.timesteps
        mlflow.log_param("Timestep", TIMESTEPS)                                 # Partial Training: total steps = 10000 * 20 = 2e5
                                                                                # Full Training   : total steps = 50000 * 20 = 1e6
        start_time0 = time.time()
        train_time = []

        print(f"Training Started....") 
        for i in range(opt.sets):

            start_time1 = time.time()
            model.learn(total_timesteps = TIMESTEPS, 
                        log_interval= 10, 
                        tb_log_name= f"{algo}",
                        reset_num_timesteps= False, 
                        progress_bar= True,
                        callback= log_callback)
                                                                                # Partial Training: total episodes = 200000/200 = 1000
                                                                                # Full Training   : total episodes = 1e6/200    = 5000
            # Save the model
            ep = (i+1) * TIMESTEPS / 200                                            
            model_path= f"{models_dir}/episode_{int(ep)}"                                                     
            model.save(model_path)                        
            print(f"Model save checkpoint {i+1}: {model_path}")

            # Log model as an artifact
            mlflow.log_artifact(model_path + ".zip", artifact_path= "models")
            print(f"Model logged as artifact on MLFlow!")

            # Log Metrics to MLFlow
            elapsed_time = time.time() - start_time1
            train_time.append(elapsed_time)
            mlflow.log_metric("train_time", elapsed_time, step= i)

        total_elapsed_time = time.time() - start_time0
        mlflow.log_metric("total_train_time", total_elapsed_time)
        print(f"Total training time: {total_elapsed_time:.2f} seconds")
    
        # ----------------------------------------------------------
        # Save metrics and plots as artifcats
        rewards = log_callback.episodic_rewards
        actor_losses = log_callback.actor_losses
        critic_losses = log_callback.critic_losses

        # Saving metrics and grapsh locally
        save_path = f"{models_dir}/details"
        os.makedirs(save_path, exist_ok= True)

        np.save(os.path.join(save_path, "rewards.npy"), rewards)
        np.save(os.path.join(save_path, "actor_losses.npy"), actor_losses)
        np.save(os.path.join(save_path, "critic_losses.npy"), critic_losses)
        np.save(os.path.join(save_path, "train_times.npy"), train_time)

        mlflow.log_artifact(os.path.join(save_path, "rewards.npy"))
        mlflow.log_artifact(os.path.join(save_path, "actor_losses.npy"))
        mlflow.log_artifact(os.path.join(save_path, "critic_losses.npy"))
        mlflow.log_artifact(os.path.join(save_path, "train_times.npy"))

        # Plot and save training time
        plt.figure()
        plt.plot(train_time, label= "Train Times")
        plt.xlabel("Set")
        plt.ylabel("Time (s)")
        plt.title("Training Time per Set")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, "train_times.png"))
        mlflow.log_artifact(os.path.join(save_path,"train_times.png"))
        plt.close()

        if rewards:
            plt.figure()
            plt.plot(rewards, label="Episodic Reward")
            plt.xlabel("Episode")
            plt.ylabel("Reward")
            plt.title("Episodic Rewards")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(save_path, "episodic_reward.png"))
            mlflow.log_artifact(os.path.join(save_path, "episodic_reward.png"))
            plt.close()
        
    # Save normalization statistics
    vec_norm_dir = os.path.join(models_dir, "norm_stats")

    if not os.path.exists(vec_norm_dir):
        os.makedirs(vec_norm_dir)
        print(f"Norm Stats directory created {vec_norm_dir}")
    
    vec_norm_path = os.path.join(vec_norm_dir, "vecnormalize.pkl")
    vec_env.save(vec_norm_path)
    mlflow.log_artifact(vec_norm_path)

    
    vec_env.close()
    del model
    torch.cuda.empty_cache()