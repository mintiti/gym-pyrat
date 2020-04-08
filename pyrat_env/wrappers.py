from gym.core import ObservationWrapper
import gym.spaces as spaces
import numpy as np
from gym.spaces import Box, Dict,Discrete, Tuple,MultiBinary,MultiDiscrete


def flatdim2(space):
    if isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Discrete):
        return 1
    elif isinstance(space, Tuple):
        return int(sum([flatdim2(s) for s in space.spaces]))
    elif isinstance(space, Dict):
        return int(sum([flatdim2(s) for s in space.spaces.values()]))
    elif isinstance(space, MultiBinary):
        return int(space.n)
    elif isinstance(space, MultiDiscrete):
        return int(np.prod(space.shape))
    else:
        raise NotImplementedError

def flatten2(space, x):
    if isinstance(space, Box):
        return np.asarray(x, dtype=np.float32).flatten()
    elif isinstance(space, Discrete):
        return [x]
    elif isinstance(space, Tuple):
        return np.concatenate([flatten2(s, x_part) for x_part, s in zip(x, space.spaces)])
    elif isinstance(space, Dict):
        return np.concatenate([flatten2(s, x[key]) for key, s in space.spaces.items()])
    elif isinstance(space, MultiBinary):
        return np.asarray(x).flatten()
    elif isinstance(space, MultiDiscrete):
        return np.asarray(x).flatten()
    else:
        raise NotImplementedError

class FlattenObservationMatrices(ObservationWrapper):
    def __init__(self,env):
        super(FlattenObservationMatrices, self).__init__(env)

        flatdim = flatdim2(env.observation_space)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=(flatdim,), dtype=np.float32)

    def observation(self, observation):
        return flatten2(self.env.observation_space, observation)

class MatricizePositions(ObservationWrapper):
    def __init__(self,env):
        super(MatricizePositions, self).__init__(env)

        obs_dim = (7,21,15)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=obs_dim, dtype=np.float32)

    def observation(self, observation):
        ret = []
        ret.append(observation['Maze_up'])
        ret.append(observation['Maze_right'])
        ret.append(observation['Maze_left'])
        ret.append(observation['Maze_down'])
        ret.append(observation['pieces_of_cheese'])


        # player location matrices
        player1 = np.zeros(ret[0].shape)
        player2 =  np.zeros(ret[0].shape)

        player1[observation['player1_location']] = 1
        player2[observation['player2_location']] = 1

        ret.append(player1)
        ret.append(player2)

        return np.array(ret)


