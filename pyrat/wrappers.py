from gym.core import ObservationWrapper, RewardWrapper, ActionWrapper
import gym.spaces as spaces
import numpy as np
from gym.spaces import Box, Dict, Discrete, Tuple, MultiBinary, MultiDiscrete




# TODO : Wrapper to play vs greedy
class VsGreedyNoMud(ActionWrapper):
    def __init__(self,env):
        super(VsGreedyNoMud, self).__init__(env)
        self.action_space = Discrete(4)
        self.greedy_player = 1 # Either 0 or 1 (first or second player)

    def action(self, action):
        greedy_action = self._calculate_greedy_action()
        actions = [None,None]

        actions[not self.greedy_player] = action
        actions[self.greedy_player] = greedy_action

        return tuple(actions)


    def reverse_action(self, action):
        return action[not self.greedy_player]

    # Greedy algorithmic
    def _distance(self,la, lb):
        ax, ay = la
        bx, by = lb
        return abs(bx - ax) + abs(by - ay)

    def _calculate_greedy_action(self):
        unwrapped_env = self.unwrapped
        game_state = unwrapped_env.state

        playerLocation = game_state.player1_location if self.greedy_player == 0\
            else game_state.player2_location
        closest_poc = (-1, -1)
        best_distance = game_state.width + game_state.height
        for poc in game_state.pieces_of_cheese:
            dist = self._distance(poc, playerLocation)
            if dist < best_distance:
                best_distance = dist
                closest_poc = poc
        ax, ay = playerLocation
        bx, by = closest_poc
        if bx > ax:
            return 2
        if bx < ax:
            return 0
        if by > ay:
            return 1
        if by < ay:
            return 3
        pass




class AlphaZeroMatricizePositions(ObservationWrapper):
    def __init__(self, env):
        super(AlphaZeroMatricizePositions, self).__init__(env)

        obs_dim = (9, 21, 15)
        self.observation_space = Box(low=-float('inf'), high=float('inf'), shape=obs_dim, dtype=np.float16)

    def observation(self, observation):
        ret = []
        ret.append(observation['Maze_left'])
        ret.append(observation['Maze_up'])
        ret.append(observation['Maze_right'])
        ret.append(observation['Maze_down'])
        ret.append(observation['pieces_of_cheese'])
        player1_score_matrix = np.full_like(observation['pieces_of_cheese'], observation['player1_score'], dtype= np.float16)
        player2_score_matrix = np.full_like(observation['pieces_of_cheese'], observation['player2_score'], dtype= np.float16)
        ret.append(player1_score_matrix)
        ret.append(player2_score_matrix)

        # player location matrices
        player1 = np.zeros(ret[0].shape, dtype= np.float16)
        player2 = np.zeros(ret[0].shape, dtype= np.float16)

        player1[observation['player1_location']] = 1
        player2[observation['player2_location']] = 1

        ret.append(player1)
        ret.append(player2)

        return np.array(ret, dtype= np.float16)

class MatricizeObservation(ObservationWrapper):
    def __init__(self,env):
        super(MatricizeObservation,self).__init__(env)

        n_channels = len(self.observation_space.spaces)
        width,height = self.observation_space['Maze_down'].shape

        self.observation_space = Box(low = 0, high = np.iinfo(np.int64).max,shape = (n_channels,width,height), dtype= np.float16)

    def observation(self, observation):
        ret = []
        ret.append(observation['Maze_left'])
        ret.append(observation['Maze_up'])
        ret.append(observation['Maze_right'])
        ret.append(observation['Maze_down'])
        ret.append(observation['pieces_of_cheese'])
        # turns
        #max turns
        #player1 score
        #p1 location
        # p1 moves
        # p1 misses
        # p1 mud

        # p2 score
        # p2 location
        # p2 moves
        # p2 misses
        # p2 mud
class FinalReward(RewardWrapper):
    def __init__(self, env):
        super(FinalReward, self).__init__(env)
        self.cumulated_reward = 0

    def reward(self, reward):
        self.cumulated_reward += reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.reward(reward)
        if done:
            reward = self.cumulated_reward / abs(self.cumulated_reward)
        else:
            reward = 0

        return observation, reward, done, info


def AlphaZero(env):
    return FinalReward(AlphaZeroMatricizePositions(env))
