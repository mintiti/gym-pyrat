from pyrat_env.envs import Pyrat_Env
from pyrat_env.wrappers import AlphaZero, MatricizePositions, FinalReward
import gym
if __name__ == '__main__':
    env = gym.make("PyRatEnv-v0")

    env = AlphaZero(env)

    print(isinstance(env,MatricizePositions))
    print(isinstance(env,FinalReward))
