from pyrat_env.envs import Pyrat_Env
from pyrat_env.wrappers import AlphaZero, AlphaZeroMatricizePositions, FinalReward
import gym
import numpy as np
import time
import tensorflow as tf
if __name__ == '__main__':
    print(tf.test.is_gpu_available())
    env = gym.make("PyRatEnv-v0")
    env = AlphaZero(env)
    env.reset()
    unwr = env.unwrapped
    done = False

