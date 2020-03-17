import gym
import pyrat_env

if __name__ == '__main__':
    env = gym.make("PyRatEnv-v0")
    while True :
        env.render()
    print(env.observation_space)