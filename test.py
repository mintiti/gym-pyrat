from pyrat_env.envs import Pyrat_Env
from pyrat_env.wrappers import AlphaZero, MatricizePositions, FinalReward
import gym
import numpy as np
import time

if __name__ == '__main__':
    env = gym.make("PyRatEnv-v0")
    env = AlphaZero(env)
    env.reset()
    unwr = env.unwrapped
    done = False
    while not done:
        obs, rew, done, _ = env.step(np.random.randint(4, size=(2,)))
        print(obs.shape)
        print(obs[5])
        print(obs[6])
        print(rew, done)

        time.sleep(0.2)
