import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    def __init__(self):
       # super().__init__()

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2 * 3,), dtype=np.float32, seed=None)

        # bandwidths : 0% fluctuation, 10% fluctuation, 20% fluctuation,..., 90% fluctuation.
        # observation_spec = [10] * (self.iotDeviceNum + self.edgeDeviceNum)
        # self.observation_space = spaces.MultiDiscrete(observation_spec)
        self.observation_space = spaces.Box(low=0.0, high=20,
                                            shape=(7,),
                                            dtype=np.float32, seed=None)

    def rewardFun(self, action):
        pass

    def step(self, action):
        pass

    def reset(self, seed=None, options=None):
        pass

    def render(self):
        pass
