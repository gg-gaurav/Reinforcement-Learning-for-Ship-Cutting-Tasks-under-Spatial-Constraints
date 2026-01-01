import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Tests on the pandaTaskEnv")

    # MLFLow Requirements
    parser.add_argument("--experiment_name", type= str, default=f"Test 16 - Final Ablation Study")
    parser.add_argument("--run_name", type=str, default=f"default run")

    # Hyperparameters
    parser.add_argument("--algo", type=str, default= "DDPG", help = "DDPG | PPO")
    parser.add_argument("--tau", type=float, default=0.05, help="Polyak averaging factor")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate ---> 1e-3 for DDPG | 1e-4 for PPO")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size ---> 256 for DDPG | 128 for PPO")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Replay buffer size")
    parser.add_argument("--noise_sigma", type=float, default=0.2, help="Standard deviation for the Gaussian Noise added to the actions")

    parser.add_argument("--timesteps", type=int, default=100000, help="Timesteps per training segment")   
    parser.add_argument("--sets", type=int, default=20, help="Number of training segments")              
    parser.add_argument("--episodic_window", type= int, default=10, help=f"Number of episodes over which we take an avergae while plotting the graphs")
    
    return parser.parse_args()