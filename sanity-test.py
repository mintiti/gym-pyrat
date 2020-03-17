import gym
import pyrat_env
#from stable_baselines.common.env_checker import check_env
if __name__ == '__main__':
    env = gym.make("PyRatEnv-v0")
    obs = env.reset()
    print(obs)
    check_env(env)

    while True :
        env.render()
        env.step((0,0))
    print(env.observation_space)