from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import numpy as np

class customCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``. 

    Parameters:
    verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    update_freq (int): frequency at which the live plot is updated
    """
    def __init__(self, verbose: int = 0, update_freq: int= 10):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.window_size = 2000                 # calculates the average loss for a particular number of episodes
        self.actor_loss_window = []
        self.critic_loss_window = []


        self.actor_losses = []
        self.critic_losses = []
        self.episodic_rewards = []
        self.current_reward = 0.0
        self.update_freq = update_freq          # frequency at which the graph is updated

        # Live Plot
        plt.ion()
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize= (8,9))

        # Actor Loss Plot
        self.actor_line, = self.ax1.plot([], [], label= "Actor Loss", color= "blue")
        self.ax1.set_xlabel("Training Steps (approx)")
        self.ax1.set_ylabel("Loss")
        self.ax1.set_title("Actor Loss")
        self.ax1.grid(True)
        self.ax1.legend()

        # Critic Loss Plot
        self.critic_line, = self.ax2.plot([], [], label= "Critic Loss", color= "red")
        self.ax2.set_xlabel("Training Steps (approx)")
        self.ax2.set_ylabel("Loss")
        self.ax2.set_title("Critic Loss")
        self.ax2.grid(True)
        self.ax2.legend()
        
        # Reward Plot
        self.reward_line, = self.ax3.plot([], [], label= "Episodic Reward", color= "green")
        self.ax3.set_xlabel("Episodes")
        self.ax3.set_ylabel("Reward")
        self.ax3.set_title("Episodic Reward")
        self.ax3.grid(True)
        self.ax3.legend()

        self.fig.tight_layout()
        self.fig.show()
        self.fig.canvas.draw()



    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        Return:
        If the callback returns False, training is aborted early.
        """
        log_data = self.model.logger.name_to_value
        
        # Losses
        if "train/actor_loss" in log_data:
            self.actor_losses.append(log_data["train/actor_loss"])
        if "train/critic_loss" in log_data:
            self.critic_losses.append(log_data["train/critic_loss"])

        # Rewards
        self.current_reward += self.locals["rewards"][0]

        if self.locals["dones"]:
            self.episodic_rewards.append(self.current_reward)
            self.current_reward = 0.0

        # Plot Update
        if len(self.actor_losses) % self.update_freq == 0 :
            
            
            # axis.plot.set_data(x[:i], y[:i])
            # self.actor_line.set_data(range(len(self.actor_losses)), self.actor_losses)
            # self.critic_line.set_data(range(len(self.critic_losses)), self.critic_losses)
            if len(self.actor_losses) >= self.window_size:
                smoothed_actor = self.running_mean(self.actor_losses, self.window_size)
                smoothed_critic = self.running_mean(self.critic_losses, self.window_size)
               

                self.actor_line.set_data(range(len(smoothed_actor)), smoothed_actor)
                self.critic_line.set_data(range(len(smoothed_critic)), smoothed_critic)
                
            else:
                # Not enough data, just plot raw
                self.actor_line.set_data(range(len(self.actor_losses)), self.actor_losses)
                self.critic_line.set_data(range(len(self.critic_losses)), self.critic_losses)
                
            self.reward_line.set_data(range(len(self.episodic_rewards)), self.episodic_rewards)
            
            self.ax1.relim(); self.ax1.autoscale_view()
            self.ax2.relim(); self.ax2.autoscale_view()
            self.ax3.relim(); self.ax3.autoscale_view()

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        return True


    def running_mean(self, arr, window_size):
        return np.convolve(arr, np.ones(window_size)/ window_size, "valid")