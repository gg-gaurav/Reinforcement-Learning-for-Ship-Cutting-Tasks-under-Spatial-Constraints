from stable_baselines3.common.noise import ActionNoise
import numpy as np

class DecayingOUNoise(ActionNoise):
    
    def __init__(
            self,
            mean: np.ndarray,
            sigma: float = 0.2,
            theta: float = 0.15,
            dt: float = 1e-2,
            x0 : np.ndarray = None,
            decay_rate: float = 0.99,
            min_sigma: float = 0.01):
        
        self.mean = mean
        self.initial_sigma = sigma
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x_prev = x0 if x0 is not None else np.zeros_like(self.mean)
        self.decay_rate = decay_rate
        self.min_sigma = min_sigma
        self.reset_count = 0

    def __call__(self) -> np.ndarray:
        noise = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = noise
        return noise

    def reset(self) -> None:
        self.x_prev = np.zeros_like(self.mean)
        # Decay sigma after each episode or rollout
        self.sigma = max(self.sigma * self.decay_rate, self.min_sigma)    