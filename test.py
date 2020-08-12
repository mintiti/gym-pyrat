from pyrat.envs import Pyrat_Env
from pyrat.wrappers import AlphaZero, AlphaZeroMatricizePositions, FinalReward
import gym
import numpy as np
import time

if __name__ == '__main__':
    print(tf.test.is_gpu_available())
    env = gym.make("PyRatEnv-v0")
    env = AlphaZero(env)
    env.reset()
    unwr = env.unwrapped
    done = False

