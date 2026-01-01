from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np
import mlflow

class vectorizedCallbackMLFlow(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``. 

    Parameters:
    verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    
    """
    def __init__(self, verbose: int = 0, steps_per_ep: int= 200, episodic_window: int=10, algo: str = "DDPG"):
        super().__init__(verbose)

        self.steps = steps_per_ep
        self.episodic_window = episodic_window                      # Number of episodes over which we average the episodic return
        self.update_frequency = self.steps * self.episodic_window   # logs actor and critic losses after every episodic_window (10) episodes

        ## REWARD STORAGE:
        self.waypointR = []
        self.dwellR = []
        self.velocityR = []
        self.boundsR = []
        self.orientationR = []

        self.algo = algo

        self.actor_losses = []
        self.critic_losses = []
        self.episodic_rewards = []
        self.current_reward = []  
        self.episode_count = 0   
        


    def _on_training_start(self) -> None:
        
        # One reward accumulator per environment
        n_envs = self.training_env.num_envs
        self.current_reward = [0.0 for _ in range(n_envs)]



    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        Return:
        If the callback returns False, training is aborted early.
        """
        log_data = self.model.logger.name_to_value

        # Logging Losses ------------------------------------------

        if self.algo == "DDPG":
            if "train/actor_loss" in log_data:
                actor_loss = log_data["train/actor_loss"]
                self.actor_losses.append(actor_loss)
            
            if "train/critic_loss" in log_data:
                critic_loss = log_data["train/critic_loss"]
                self.critic_losses.append(critic_loss)

        if self.algo == "PPO":

            policy_gradient_loss = log_data.get("train/policy_gradient_loss",0.0)
            entropy_loss = log_data.get("train/entropy_loss",0.0)
            actor_loss = - (policy_gradient_loss + entropy_loss)
            self.actor_losses.append(actor_loss)

            value_loss = log_data.get("train/value_loss",0.0)
            self.critic_losses.append(value_loss)
            

        # Every 2000 steps or 10 episodes log the mean actor loss for the last 100 steps
        if self.num_timesteps > 0 and self.num_timesteps % self.update_frequency == 0: 
            
            if len(self.actor_losses) >= 1 and len(self.critic_losses) >= 1:
                mlflow.log_metrics({
                "actor_loss": np.mean(self.actor_losses[-100:]),
                "critic_loss": np.mean(self.critic_losses[-100:]),
                }, step=self.num_timesteps)


        # Episodic Rewards --------------------------------------       
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        # iterate through all the environments
        for i in range(len(dones)):

            # add the reward to the current_reward for each env    
            self.current_reward[i] += rewards[i]

            # for when the episode is complete and done == True       
            if dones[i]:

                self.episode_count += 1
                
                reward = float(self.current_reward[i])
                self.episodic_rewards.append(reward)
                mlflow.log_metric("episodic_reward", reward, step= self.episode_count)      # at the end of every episode log the return

                info = infos[i]
                # Cumulative rewards over that specific episode
                self.waypointR.append(float(info.get("waypointR", 0.0)))
                self.dwellR.append(float(info.get("dwellR", 0.0)))
                self.velocityR.append(float(info.get("velocityR", 0.0)))
                self.boundsR.append(float(info.get("boundsR", 0.0)))
                self.orientationR.append(float(info.get("orientationR", 0.0)))

                # Log the avg episodic reward of every 10 episodes
                if len(self.episodic_rewards) >= self.episodic_window and len(self.episodic_rewards) % self.episodic_window == 0:
                    avg_reward = np.mean(self.episodic_rewards[-self.episodic_window:])
                    mlflow.log_metric("avg_episodic_reward", avg_reward, step = self.episode_count)

                    # Log the sum of each reward at the end of each episode
                    mlflow.log_metrics({
                        "reward/waypointR": np.mean(self.waypointR[-self.episodic_window:]),
                        "reward/dwellR": np.mean(self.dwellR[-self.episodic_window:]),
                        "reward/velocityR":np.mean(self.velocityR[-self.episodic_window]),
                        "reward/boundsR":np.mean(self.boundsR[-self.episodic_window]),
                        "reward/orientationR":np.mean(self.orientationR[-self.episodic_window])
                        }, 
                        step = self.episode_count)
                
                mlflow.log_metric("episode_number", self.episode_count, step = self.episode_count)
                self.current_reward[i] = 0.0
        
        return True
    

    def _on_training_end(self) -> None:
        # Example: save reward plot
        plt.figure()
        plt.plot(self.episodic_rewards)
        plt.title("Episodic Reward Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig("reward_curve.png")
        mlflow.log_artifact("reward_curve.png")
        plt.close()
