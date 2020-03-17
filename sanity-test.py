import gym
import pyrat_env

if __name__ == '__main__':
    env = gym.make("PyRatEnv-v0")
    print(env.observation_space)