# Imports

import gym
from gym import spaces
import numpy as np
from pickle import load, dump

# CONSTANTS
from pyrat_env.envs.core import GameGenerator

DECISION_FROM_ACTION_DICT = {
    0: 'L',
    1: 'U',
    2: "R",
    3: 'D'
}


class PyratEnv(gym.Env):
    # TODO : add mud
    # TODO : add redo rendering
    # TODO : code a replay system and a replayer
    """
    Description:
        2 agents compete in a maze for rewards randomly dispersed in the maze. The goal is to collect the most.
    Observation:
        Type: Dict({
            'Maze_up':
                Type : Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix [M_ij] where M_ij = 1 if you can go up from case ij

            'Maze_down':
                Type : Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix [M_ij] where M_ij = 1 if you can go down from case ij

            'Maze_right':
                Type : Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix [M_ij] where M_ij = 1 if you can go right from case ij

            'Maze_left':
                Type : Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix [M_ij] where M_ij = 1 if you can go left from case ij

            'pieces_of _cheese' :
                Type :Box( low= 0 , high =1, shape = (maze_width, maze_height), dtype = np.int8)
                    A matrix where m_ij = 0 if there is no cheese on this bloc and 1 if there is

            'turns' :
                Type : Box(low=0, high= max_turns, shape=(1,), dtype=np.int)
                    1 dimensional array containing the number of turns played

            'max_turns' :
                Type : Box(low=0, high= + inf, shape=(1,), dtype=np.int)
                    Maximum number of turns allowed to be played

            'player1_location' :
                Type : Tuple ( Discrete(maze_width), Discrete(maze_height))
                    The location of player 1

            'player2_location' :
                Type : Tuple ( Discrete(maze_width), Discrete(maze_height))
                    The location of player 2
        })

    Actions:
        Type: Tuple( Discrete(4), Discrete(4))
        For each agent :
        Num	Action
        0	Agent to the left
        1	Agent up
        2   Agent to the right
        3   Agent down

    Reward:
        Reward is 1 if player 1 eats a cheese and player 2 doesn't, 0 if both take a cheese and -1 if player 2 takes a cheese and player1 doesn't
    Starting State:
        A random (connected) maze with random cheese disposition
        For now no mud is included
        Each player start respectively on the lower left and higher right corner
    Episode Termination:
        Max number of turns is reached
        One player takes more than half of the cheeses available
    """
    # Gym API methods
    metadata = {'render.modes': ['human', 'none', 'text']}
    reward_range = (-1, 1)

    def __init__(self, width=21, height=15, nb_pieces_of_cheese=41, max_turns=2000, target_density=0.7, connected=True,
                 symmetry=True, mud_density=0.7, mud_range=10, maze_file="", start_random=False):
        self.game_generator = GameGenerator(width=width, height=height, nb_pieces_of_cheese=nb_pieces_of_cheese,
                                            max_turns=max_turns, target_density=target_density, connected=connected,
                                            symmetry=symmetry, mud_density=mud_density, mud_range=mud_range,
                                            maze_file=maze_file, start_random=start_random)

        self.state = self.game_generator()

        # Gym Interface
        self.action_space = spaces.Tuple([spaces.Discrete(4),
                                          spaces.Discrete(4)
                                          ])

        # Define the observation space
        self._set_obs_space()

    @classmethod
    def fromPickle(cls, p="./maze_files/maze.p"):
        """
        Lets you load a given maze from a previously pickled object
        Sample can be found under ./maze_files/maze.p
        :param p: the path to the maze
        :return: the object instance
        """
        return load(open(p, 'rb'))

    def step(self, action):
        reward, done = self.state.step(action)
        # Calculate the return variables
        observation = self._observation()
        info = dict()

        return observation, reward, done, info

    def reset(self):
        self.state = self.game_generator()
        # Set the observation space
        self._set_obs_space()

        return self._observation()

    # TODO : Rendering : maybe switch it to something better than pygame
    # TODO : Sound
    def render(self, mode='human'):
        if mode == 'human':
            (window_width, window_height) = cfg['resolution']
            scale, offset_x, offset_y, image_background, image_cheese, image_corner, image_moving_python, image_moving_rat, image_python, image_rat, image_wall, image_mud, image_portrait_python, image_portrait_rat, tiles, image_tile = init_coords_and_images(
                self.width, self.height, True, True, window_width, window_height)
            if self.bg is None:
                pygame.init()
                screen = pygame.display.set_mode(cfg['resolution'])
                self.bg = build_background(screen, self.maze, tiles, image_background, image_tile, image_wall,
                                           image_corner,
                                           image_mud,
                                           offset_x, offset_y, self.width, self.height, window_width, window_height,
                                           image_portrait_rat,
                                           image_portrait_python, scale, True, True)
            screen = pygame.display.get_surface()
            screen.blit(self.bg, (0, 0))
            draw_pieces_of_cheese(self.pieces_of_cheese, image_cheese, offset_x, offset_y, scale, self.width,
                                  self.height, screen,
                                  window_height)
            draw_players(self.player1_location, self.player2_location, image_rat, image_python, offset_x, offset_y,
                         scale, self.width,
                         self.height, screen, window_height)
            draw_scores("Rat", self.player1_score, image_portrait_rat, "Python", self.player2_score,
                        image_portrait_python, window_width,
                        window_height, screen, True, True, self.player1_last_move, self.player1_misses,
                        self.player2_last_move, self.player2_misses, 0,
                        0)
            pygame.display.update()
            pygame.event.get()

        elif mode == "none":
            pass

        elif mode == "text":
            print(self.state)

    # Utils methods

    def save_pickle(self, path="./maze_files/maze_save.p"):
        """
        pickles the maze to the given path
        :param path: the path to save to
        :return:
        """
        dump(self, open(path, "wb"))

    def _observation(self):
        """Creates the observation"""
        # get the unformatted observation from the state
        raw_obs = self.state.get_obs()
        # Create the maze matrices
        maze_matrix_L, maze_matrix_U, maze_matrix_R, maze_matrix_D = self.state.get_maze_matrix()
        # Create the cheese matrix
        cheese_matrix = self.state.get_cheese_matrix()

        return dict({
            # Global infos
            'Maze_up': maze_matrix_U,
            'Maze_right': maze_matrix_R,
            'Maze_left': maze_matrix_L,
            'Maze_down': maze_matrix_D,
            'pieces_of_cheese': cheese_matrix,

            'turns': np.array([raw_obs["turns"]], dtype=np.int),
            'max_turns': np.array([raw_obs['max_turns']], dtype=np.int),

            # Player 1 variables
            'player1_score': np.array([raw_obs["player1_score"]]),
            'player1_location': np.array(raw_obs["player1_location"], dtype=np.int),
            'player1_moves': np.array([raw_obs["player1_moves"]], dtype=np.int),
            'player1_misses': np.array([raw_obs["player1_misses"]], dtype=np.int),
            'player1_mud': np.array([raw_obs["player1_mud"]], dtype=np.int),

            # Player 2 variables
            'player2_score': np.array([raw_obs["player2_score"]]),
            'player2_location': np.array(raw_obs["player2_location"], dtype=np.int),
            'player2_moves': np.array([raw_obs["player2_moves"]], dtype=np.int),
            'player2_misses': np.array([raw_obs["player2_misses"]], dtype=np.int),
            'player2_mud': np.array([raw_obs["player2_mud"]], dtype=np.int),
        })

    def _set_obs_space(self):
        self.observation_space = spaces.Dict({
            # Global infos
            'Maze_up': spaces.Box(low=0, high=1, shape=(self.state.width, self.state.height), dtype=np.int8),
            'Maze_down': spaces.Box(low=0, high=1, shape=(self.state.width, self.state.height), dtype=np.int8),
            'Maze_right': spaces.Box(low=0, high=1, shape=(self.state.width, self.state.height), dtype=np.int8),
            'Maze_left': spaces.Box(low=0, high=1, shape=(self.state.width, self.state.height), dtype=np.int8),
            'pieces_of_cheese': spaces.Box(low=0, high=1, shape=(self.state.width, self.state.height), dtype=np.int8),

            'turns': spaces.Box(low=0, high=self.state.max_turns, shape=(1,), dtype=np.int),
            'max_turns': spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(1,), dtype=np.int),

            # Player 1 variables
            'player1_score': spaces.Box(low=0, high=self.state.original_nb_cheeses, shape=(1,)),
            'player1_location': spaces.Box(low=np.array([0, 0]),
                                           high=np.array([self.state.width - 1, self.state.height - 1]), shape=(2,),
                                           dtype=np.int),
            'player1_moves': spaces.Box(low=0, high=self.state.max_turns, shape=(1,), dtype=np.int),
            'player1_misses': spaces.Box(low=0, high=self.state.max_turns, shape=(1,), dtype=np.int),
            'player1_mud': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int),

            # Player 2 variables
            'player2_score': spaces.Box(low=0, high=self.state.original_nb_cheeses, shape=(1,)),
            'player2_location': spaces.Box(low=np.array([0, 0]),
                                           high=np.array([self.state.width - 1, self.state.height - 1]), shape=(2,),
                                           dtype=np.int),
            'player2_moves': spaces.Box(low=0, high=self.state.max_turns, shape=(1,), dtype=np.int),
            'player2_misses': spaces.Box(low=0, high=self.state.max_turns, shape=(1,), dtype=np.int),
            'player2_mud': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int),

        })
