# Imports

import gym
from gym import spaces
from ..pyrat import move
import numpy as np
from ..config import cfg
from pickle import load, dump
from ..imports.maze import *
from ..imports.display import *
from ..imports.parameters import *
from abc import ABC

# CONSTANTS
DECISION_FROM_ACTION_DICT = {
    0: 'L',
    1: 'U',
    2: "R",
    3: 'D'
}


class GameGenerator:
    """Maze Generator class
    Handles Pyrat Maze generation.
    Most functions are ported from https://github.com/vgripon/PyRat/blob/master/imports/maze.py

    To get a maze :
        >> maze_generator = GameGenerator()
        >> width, height, maze, pieces_of_cheese,  player1_location, player2_location = maze_generator()"""

    def __init__(self, width=21, height=15, nb_pieces_of_cheese=41, target_density=0.7, mud_density=0, connected=True,
                 symmetry=True, mud_range=10, start_random=False, maze_file="", seed="random", max_turns = 2000):
        """
        Initializes the maze generator
        Most arguments can take either a fixed value or can be sampled uniformly at random from a distribution.
        Only height and width
        For INT parameters :
            - scalar value : parameter is always that value
            - list [a,b] : integer sampled uniformly at random from integer interval  [a,b]
        For FLOAT parameters :
            - scalar value : parameter is always that value
            - list [a,b] : float sampled uniformly at random from integer interval  [a,b]
        For BOOL parameters :
            - True : parameter is always True
            - False : parameter is always False
            - scalar value p within [0,1]: parameter follows a Bernoulli Law of parameter p, i.e. has probability p to be True

        ---------------------
        Arguments:
            width : int
                NOT RANDOMIZABLE
                Width of the maze. Default : 21
            height : int
                NOT RANDOMIZABLE
                Height of the maze. Default : 15
            nb_pieces_of_cheese : int
                RANDOMIZABLE
                Number of initial pieces of cheese put into the maze. Default : 41
            target_density : float
                RANDOMIZABLE
                Probability of having a wall between two given maze cells. Needs to be in [0,1]. Default : 0.7
            mud_density : float
                RANDOMIZABLE
                Probability of having mud between two given maze cells. Needs to be in [0,1]. Default : 0
            connected : boolean
                RANDOMIZABLE
                Whether the maze should be in one connected component or not. Default : True
            symmetry : boolean
                RANDOMIZABLE
                Whether the maze should be symmetric. Default : True
            mud_range : int
                RANDOMIZABLE
                Upper bound on the mud values i.e. the number of turns it can take to cross from one cell to another.
            start_random : boolean
                RANDOMIZABLE
                Whether the players should be started in fixed positions (upper right corner and lower left) or if they can be started from anywhere in the maze.
                Default : False
            maze_file : string
                Path to a maze file, if you want the maze that's generated to always be the same
            seed : int or "random"
                Seed for torch, numpy and random.
                Can be either chose randomly if "random" or be a fixed int
        ---------------------
        Use examples :
        >> GameGenerator() : Default values.
                15x21 maze, 41 cheeses, 0.7 wall density, no mud, one connected component, a symmetric maze and players starting in the lower left and upper right corners respectively.
        >> GameGenerator(nb_pieces_of_cheese = [21,41]) :
                Same as previously, but with a number of pieces of cheese chosen uniformly in [21,41].
        >> GameGenerator(target_density = [0.2,0.7]) :
                Same as the first one, but with a wall density chosen uniformly in [0.2,0.7].
        >> GameGenerator(symmetry = 0.8)
                Same as the first example, but with a maze having 0.8 probability to be symmetric.
        """
        # TODO : Implement seed everywhere
        self.width = width
        self.height = height
        self.max_turns = max_turns
        self.nb_pieces_of_cheese = IntParameter(nb_pieces_of_cheese)
        self.target_density = FloatParameter(target_density)
        self.mud_density = FloatParameter(mud_density)
        self.connected = BooleanParameter(connected)
        self.symmetry = BooleanParameter(symmetry)
        self.mud_range = IntParameter(mud_range)
        self.start_random = BooleanParameter(start_random)
        self.maze_file = maze_file

        assert isinstance(seed, str) or isinstance(seed,
                                                   int), "seed parameter needs to be either and integer or 'random'"
        if isinstance(seed, str):
            assert seed == "random", f"unrecognized value for seed '{seed}'. Needs to be 'random'"
        self.seed = seed

        self.maze_gen = MazeGenerator()

    def _sample_parameters(self):
        (width,
         height,
         nb_pieces_of_cheese,
         target_density,
         mud_density,
         connected,
         symmetry,
         mud_range,
         start_random,
         maze_file) = (self.width, self.height,
                       self.nb_pieces_of_cheese.sample(),
                       self.target_density.sample(),
                       self.mud_density.sample(),
                       self.connected.sample(),
                       self.symmetry.sample(),
                       self.mud_range.sample(),
                       self.start_random.sample(),
                       self.maze_file)
        if self.seed == "random":
            seed = random.randint(0, sys.maxsize)
        else:
            seed = self.seed

        return width, height, nb_pieces_of_cheese, target_density, mud_density, connected, symmetry, mud_range, start_random, maze_file, seed

    def __call__(self):
        width, height, nb_pieces_of_cheese, target_density, mud_density, connected, symmetry, mud_range, start_random, maze_file, seed = self._sample_parameters()
        width, height, maze, pieces_of_cheese,  player1_location, player2_location = self.maze_gen(width, height,
                                                                                                  nb_pieces_of_cheese,
                                                                                                  target_density,
                                                                                                  mud_density,
                                                                                                  connected, symmetry,
                                                                                                  mud_range,
                                                                                                  start_random,
                                                                                                  maze_file, seed)
        game_state = PyratState(width=width, height = height,maze = maze,pieces= pieces_of_cheese, player1_location = player1_location, player2_location= player2_location )
        return game_state


class PyratState:
    """
    Class representing a game state.
    Mud support is to be tested
    """
    # Dictionary for converting a direction to an offset and vice-versa
    DIRECTION_DICT = {
        "L": (-1, 0),
        "U": (0, 1),
        "R": (1, 0),
        "D": (0, -1),
        (-1, 0): "L",
        (0, 1): "U",
        (1, 0): "R",
        (0, -1): "D"}

    def __init__(self, width=21, height=15, max_turns=2000, turns = 0,maze = {},
                 pieces = [], player1_location = None,player1_score = 0, player1_moves = 0, player1_misses = 0,player1_mud = 0,
                 player2_location = None, player2_score = 0, player2_moves = 0,player2_misses = 0,player2_mud = 0):
        # Maze configs
        self.width = width
        self.height = height

        # Global variables
        self.turns = turns
        self.max_turns = max_turns

        self.maze = maze
        self.pieces_of_cheese = pieces

        # Player 1 variables
        self.player1_location = player1_location
        self.player1_score = player1_score
        self.player1_moves = player1_moves
        self.player1_misses = player1_misses
        self.player1_mud = player1_mud

        # Player 2 variables
        self.player2_location = player2_location
        self.player2_score = player2_score
        self.player2_moves = player2_moves
        self.player2_misses = player2_misses
        self.player2_mud = player2_mud

    @property
    def original_nb_cheeses(self):
        return self.player1_score + self.player2_score + len(self.pieces_of_cheese)

    def step(self, decisions):
        """Makes a step for both players at the same time
        """
        # Move the players
        self._move(decisions)

        # Check for cheese captures and calculate the new scores
        reward = self._capture_cheeses_and_update_scores()

        return reward

    def set_state(self, state : dict):
        assert self._valid_state_dict(state), f"Invalid state : state needs to contain keys {self.__dict__.keys()}"
        for k in self.__dict__.keys():
            self.__setattr__(k,state[k])

    def get_obs(self):
        return self.__dict__

    def _move(self, decisions):
        p1_action, p2_action = decisions
        next_cell_p1, next_cell_p2 = np.add(self.player1_location, self.DIRECTION_DICT[p1_action]), np.add(
            self.player2_location, self.DIRECTION_DICT[p2_action])

        # Player 1
        # If the player is stuck in mud
        if self.player1_mud > 0:
            # One turn towards being able to move
            self.player1_mud -= 1
        # Elif next move is valid
        elif next_cell_p1 in self.maze[self.player1_location]:
            # Number of turns after this one the player is going to be stuck
            self.player1_mud = self.maze[self.player1_location][next_cell_p1] - 1
            self.player1_location = next_cell_p1
            self.player1_moves += 1
        else:
            self.player1_misses += 1

        # Player 2
        # If the player is stuck in mud
        if self.player2_mud > 0:
            # One turn towards being able to move
            self.player2_mud -= 1
        # Elif next move is valid
        elif next_cell_p2 in self.maze[self.player2_location]:
            # Number of turns after this one the player is going to be stuck
            self.player2_mud = self.maze[self.player2_location][next_cell_p2] - 1
            self.player2_location = next_cell_p2
            self.player2_moves += 1
        else:
            self.player2_misses += 1

        self.turns += 1

    def _capture_cheeses_and_update_scores(self):
        reward = 0
        if self.player1_location in self.pieces_of_cheese and self.player1_mud <= 0:
            self.pieces_of_cheese.remove(self.player1_location)
            if self.player2_location == self.player1_location and self.player2_mud <= 0:
                self.player1_score += 0.5
                self.player2_score += 0.5
            else:
                self.player1_score += 1
                reward = 1
        if self.player2_location in self.pieces_of_cheese and self.player2_mud <= 0:
            self.pieces_of_cheese.remove(self.player2_location)
            self.player2_score += 1
            reward = -1

        return reward

    def _check_done(self):
        return (self.turns >= self.max_turns) or (
                self.player1_score > self.original_nb_cheeses /2) or (
                self.player2_score > self.original_nb_cheeses /2)


    # Utility methods

    def _valid_state_dict(self,state : dict):
        valid = True
        for key in self.__dict__.keys():
            valid = valid and (key in state.keys())

        return valid

    def __str__(self):
        return f"""Game State : {self.turns}/{self.max_turns} turns
- Maze : {self.maze}
- Pieces of cheese : ({len(self.pieces_of_cheese)} remaining)
                     {self.pieces_of_cheese}
---------------- Player 1 ----------------
- Score : {self.player1_score}
- Location : {self.player1_location}
- Moves : {self.player1_moves}
- Misses : {self.player1_misses}
- Turns remaining stuck in mud : {self.player1_mud}

---------------- Player 2 ----------------
- Score : {self.player2_score}
- Location : {self.player2_location}
- Moves : {self.player2_moves}
- Misses : {self.player2_misses}
- Turns remaining stuck in mud : {self.player2_mud}\n"""

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class MazeParameter(ABC):
    def sample(self):
        pass

    def __str__(self):
        print(f"<{self.__class__} object with value {self.value}>")

    def __repr__(self):
        return f"<{self.__class__} object with value {self.value}>"


class BooleanParameter(MazeParameter):
    def __init__(self, value):
        assert isinstance(value, bool) or isinstance(value, float) or isinstance(value,
                                                                                 int), "value must be either a boolean or a float"

        self.value = float(value)

    @property
    def sampler(self):
        return lambda: bool(np.random.binomial(1, self.value))

    def sample(self):
        return self.sampler()

    def __str__(self):
        print(f"<{self.__class__} object with value {self.value}>")


class FloatParameter(MazeParameter):
    def __init__(self, value):
        assert isinstance(value, float) or isinstance(value,
                                                      list) or isinstance(value,
                                                                          int), "value must be either a float or a list representing an interval"
        if isinstance(value, list):
            assert len(value) == 2, f"length of value must be 2 to represent an interval, currently length {len(value)}"
            a, b = value
            assert a <= b, "Your interval [a,b] is invalid because a>b. You must have a<=b for it to be a valid interval"
            assert a >= 0 and b <= 1, "Your interval must be contained in [0,1]"

        if isinstance(value, float) or isinstance(value, int):
            self.value = [value, value]

        else:
            self.value = value

    @property
    def sampler(self):
        a, b = self.value
        return lambda: random.uniform(a, b)

    def sample(self):
        return self.sampler()


class IntParameter(MazeParameter):
    def __init__(self, value):
        assert isinstance(value, int) or isinstance(value,
                                                    list), "value must be either an int or a list representing an interval"
        if isinstance(value, list):
            assert len(value) == 2, f"length of value must be 2 to represent an interval, currently length {len(value)}"
            a, b = value
            a, b = int(a), int(b)
            value = [a, b]
            assert a <= b, "Your interval [a,b] is invalid because a>b. You must have a<=b for it to be a valid interval"

        if isinstance(value, float) or isinstance(value, int):
            self.value = [int(value), int(value)]
        else:
            self.value = value

    @property
    def sampler(self):
        a, b = self.value
        return lambda: random.randint(a, b)

    def sample(self):
        return self.sampler()


class MazeGenerator:
    """Class responsible for returning a maze.
    Maze generation returns the width, the height, the list of pieces of cheeses and the maze dictionary.
    Most of the code is directly ported from https://github.com/vgripon/PyRat/blob/master/imports/maze.py"""

    def __init__(self):
        self.width = None
        self.height = None
        self.maze = {}
        self.pieces_of_cheese = []

    def connected_region(self, cell, connected, possible_border):
        for (i, j) in self.maze[cell]:
            if connected[i][j] == 0:
                connected[i][j] = 1
                possible_border.append((i, j))
                self.connected_region((i, j), connected, possible_border)

    def gen_mud(self, mud_density, mud_range):
        """Generates a value for the distance between 2 cells"""
        if random.uniform(0, 1) < mud_density:
            return random.randrange(2, mud_range + 1)
        else:
            return 1

    def generate_pieces_of_cheese(self, nb_pieces, symmetry, start_random):
        """Generates the list of pieces of cheese, given maze configs

        ----
        Returns:
            (pieces_of_cheese : list, player1_location : (x,y) coordinates, player2_location :(x,y) coordinates)
            """
        player1_location = (-1, -1)
        player2_location = (-1, -1)
        if start_random:
            remaining = nb_pieces + 2
        else:
            remaining = nb_pieces
            player1_location = (0, 0)
            player2_location = (self.width - 1, self.height - 1)
        pieces = []
        candidates = []
        considered = []
        if symmetry:
            if nb_pieces % 2 == 1 and (self.width % 2 == 0 or self.height % 2 == 0):
                sys.exit(
                    "The maze has even width or even height and thus cannot contain an odd number of pieces of cheese if symmetric.")
            if nb_pieces % 2 == 1:
                pieces.append((self.width // 2, self.height // 2))
                considered.append((self.width // 2, self.height // 2))
                remaining = remaining - 1
        for i in range(self.width):
            for j in range(self.height):
                if (not (symmetry) or not ((i, j) in considered)) and (i, j) != player1_location and (
                        i, j) != player2_location:
                    candidates.append((i, j))
                    if symmetry:
                        considered.append((i, j))
                        considered.append((self.width - 1 - i, self.height - 1 - j))
        while remaining > 0:
            if len(candidates) == 0:
                sys.exit("Too many pieces of cheese for that dimension of maze")
            chosen = candidates[random.randrange(len(candidates))]
            pieces.append(chosen)
            if symmetry:
                a, b = chosen
                pieces.append((self.width - a - 1, self.height - 1 - b))
                symmetric = (self.width - a - 1, self.height - 1 - b)
                candidates = [i for i in candidates if i != symmetric]
                remaining = remaining - 1
            candidates = [i for i in candidates if i != chosen]
            remaining = remaining - 1
        if not (start_random):
            pieces.append(player1_location)
            pieces.append(player2_location)
        return pieces[:-2], pieces[-2], pieces[-1]

    def generate_maze(self, target_density, connected, symmetry, mud_density, mud_range, maze_file, seed):
        """Generates the maze and
        ------
        Returns :
            width (int): the maze's width
            height (int): the maze's height
            pieces_of_cheese (list): the list of cheeses in the maze. Is [] if the maze was not loaded from a file.
            maze (dict) : the dictionary representing the maze"""
        width = self.width
        height = self.height
        if maze_file != "":
            with open(maze_file, 'r') as content_file:
                content = content_file.read()
            lines = content.split("\n")
            width = int(lines[0])
            height = int(lines[1])
            maze = {}
            for i in range(width):
                for j in range(height):
                    maze[(i, j)] = {}
                    line = lines[i + j * width + 2].split(" ")
                    if line[0] != "0":
                        maze[(i, j)][(i, j + 1)] = int(line[0])
                    if line[1] != "0":
                        maze[(i, j)][(i, j - 1)] = int(line[1])
                    if line[2] != "0":
                        maze[(i, j)][(i - 1, j)] = int(line[2])
                    if line[3] != "0":
                        maze[(i, j)][(i + 1, j)] = int(line[3])
            line = lines[height * width + 2].split(" ")
            pieces_of_cheese = []
            for i in range(len(line)):
                l = int(line[i])
                pieces_of_cheese.append((l % width, l // width))
        else:
            random.seed(seed)
            # Start with purely random maze
            maze = {};
            not_considered = {};
            for i in range(width):
                for j in range(height):
                    maze[(i, j)] = {}
                    not_considered[(i, j)] = True
            for i in range(width):
                for j in range(height):
                    if not (symmetry) or not_considered[(i, j)]:
                        if random.uniform(0, 1) > target_density and i + 1 < width:
                            m = self.gen_mud(mud_density, mud_range)
                            maze[(i, j)][(i + 1, j)] = m
                            maze[(i + 1, j)][(i, j)] = m
                            if symmetry:
                                maze[(width - 1 - i, height - 1 - j)][(width - 2 - i, height - 1 - j)] = m
                                maze[(width - 2 - i, height - 1 - j)][(width - 1 - i, height - 1 - j)] = m
                        if random.uniform(0, 1) > target_density and j + 1 < height:
                            m = self.gen_mud(mud_density, mud_range)
                            maze[(i, j)][(i, j + 1)] = m
                            maze[(i, j + 1)][(i, j)] = m
                            if symmetry:
                                maze[(width - 1 - i, height - 2 - j)][(width - 1 - i, height - 1 - j)] = m
                                maze[(width - 1 - i, height - 1 - j)][(width - 1 - i, height - 2 - j)] = m
                        if symmetry:
                            not_considered[(i, j)] = False
                            not_considered[(width - 1 - i, height - 1 - j)] = False
            for i in range(width):
                for j in range(height):
                    if len(maze[(i, j)]) == 0 and (i == 0 or j == 0 or i == width - 1 or j == height - 1):
                        m = self.gen_mud(mud_density, mud_range)
                        possibilities = []
                        if i + 1 < width:
                            possibilities.append((i + 1, j))
                        if j + 1 < height:
                            possibilities.append((i, j + 1))
                        if i - 1 >= 0:
                            possibilities.append((i - 1, j))
                        if j - 1 >= 0:
                            possibilities.append((i, j - 1))
                        chosen = possibilities[random.randrange(len(possibilities))]
                        maze[(i, j)][chosen] = m
                        maze[chosen][(i, j)] = m
                        if symmetry:
                            ii, jj = chosen
                            maze[(width - 1 - i, height - 1 - j)][(width - 1 - ii, height - 1 - jj)] = m
                            maze[(width - 1 - ii, height - 1 - jj)][(width - 1 - i, height - 1 - j)] = m

            # Then connect it
            if connected:
                connected = [[0 for x in range(height)] for y in range(width)]
                possible_border = [(0, height - 1)]
                connected[0][height - 1] = 1
                connected_region(maze, (0, height - 1), connected, possible_border)
                while 1:
                    border = []
                    new_possible_border = []
                    for (i, j) in possible_border:
                        is_candidate = False
                        if not ((i + 1, j) in maze[(i, j)]) and i + 1 < width:
                            if connected[i + 1][j] == 0:
                                border.append(((i, j), (i + 1, j)))
                                is_candidate = True
                        if not ((i - 1, j) in maze[(i, j)]) and i > 0:
                            if connected[i - 1][j] == 0:
                                border.append(((i, j), (i - 1, j)))
                                is_candidate = True
                        if not ((i, j + 1) in maze[(i, j)]) and j + 1 < height:
                            if connected[i][j + 1] == 0:
                                border.append(((i, j), (i, j + 1)))
                                is_candidate = True
                        if not ((i, j - 1) in maze[(i, j)]) and j > 0:
                            if connected[i][j - 1] == 0:
                                border.append(((i, j), (i, j - 1)))
                                is_candidate = True
                        if is_candidate:
                            new_possible_border.append((i, j))
                    possible_border = new_possible_border
                    if not border:
                        break
                    a, b = border[random.randrange(len(border))]
                    m = self.gen_mud(mud_density, mud_range)
                    maze[a][b] = m
                    maze[b][a] = m
                    ai, aj = a
                    bi, bj = b
                    if symmetry:
                        bsym = (width - 1 - bi, height - 1 - bj)
                        asym = (width - 1 - ai, height - 1 - aj)
                        maze[asym][bsym] = m
                        maze[bsym][asym] = m
                    connected[bi][bj] = 1
                    connected_region(maze, b, connected, possible_border)
                    possible_border.append(b)
                    if symmetry:
                        if connected[width - 1 - bi][height - 1 - bj] == 0 and connected[width - 1 - ai][
                            height - 1 - aj] == 1:
                            connected[width - 1 - bi][height - 1 - bj] = 1
                            connected_region(maze, bsym, connected, possible_border)
                            possible_border.append(bsym)
            pieces_of_cheese = []

        return width, height, pieces_of_cheese, maze

    def __call__(self, width, height, nb_pieces_of_cheese, target_density, mud_density, connected, symmetry, mud_range,
                 start_random, maze_file, seed):
        self.width = width
        self.height = height
        self.maze = {}
        self.pieces_of_cheese = []

        self.width, self.height, self.pieces_of_cheese, self.maze = self.generate_maze(target_density, connected,
                                                                                       symmetry, mud_density, mud_range,
                                                                                       maze_file, seed)

        if self.pieces_of_cheese == []:
            self.pieces_of_cheese, p1_location, p2_location = self.generate_pieces_of_cheese(nb_pieces_of_cheese,
                                                                                             symmetry, start_random)

        return self.width, self.height, self.maze, self.pieces_of_cheese, p1_location, p2_location


class PyratEnv(gym.Env):
    # TODO : add mud
    # TODO : code a replay system and a replayer
    # TODO : recode the maze matrix to be 4 matrices indicating whether you can go up, down, to the right and to the left respectively
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
    metadata = {'render.modes': ['human', 'none']}
    reward_range = (-1, 1)

    def __init__(self, width=21, height=15, nb_pieces_of_cheese=41, max_turns=2000, target_density=0.7, connected=True,
                 symmetry=True, mud_density=0.7, mud_range=10, maze_file="", start_random=False):
        self.max_turns = max_turns
        self.target_density = target_density
        self.connected = connected
        self.symmetry = symmetry
        self.mud_density = mud_density
        self.mud_range = mud_range
        self.maze_file = maze_file
        self.start_random = start_random
        self.turn = 0
        self.maze = None
        self.maze_dimension = (width, height)
        self.pieces_of_cheese = []
        self.nb_pieces_of_cheese = nb_pieces_of_cheese
        self.player1_location = None
        self.player2_location = None
        self.player1_score = 0
        self.player2_score = 0
        self.player1_misses = 0
        self.player2_misses = 0
        self.player1_moves = 0
        self.player2_moves = 0
        self.random_seed = random.randint(0, sys.maxsize)
        self.width, self.height, self.pieces_of_cheese, self.maze = generate_maze(self.maze_dimension[0],
                                                                                  self.maze_dimension[1],
                                                                                  self.target_density,
                                                                                  self.connected, self.symmetry,
                                                                                  self.mud_density, self.mud_range,
                                                                                  self.maze_file,
                                                                                  self.random_seed)
        self.pieces_of_cheese, self.player1_location, self.player2_location = generate_pieces_of_cheese(
            self.nb_pieces_of_cheese, self.maze_dimension[0], self.maze_dimension[1],
            self.symmetry,
            self.player1_location,
            self.player2_location,
            self.start_random)

        # Wrapper attributes
        # Create the maze matrix
        self.maze_matrix_U = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_D = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_R = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_L = np.zeros((self.width, self.height), dtype=np.int8)
        self._maze_matrix_from_dict()
        # Create the cheese matrix
        self.cheese_matrix = np.zeros((self.width, self.height), dtype=np.int8)
        self._cheese_matrix_from_list()
        # create the player score matrices
        self.action_space = spaces.Tuple([spaces.Discrete(4),
                                          spaces.Discrete(4)
                                          ])

        # Define the observation space
        self.observation_space = spaces.Dict({
            'Maze_up': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'Maze_down': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'Maze_right': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'Maze_left': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'pieces_of_cheese': spaces.Box(low=0, high=1, shape=(self.width, self.height), dtype=np.int8),
            'player1_score': spaces.Box(low=0, high=nb_pieces_of_cheese, shape=(1,)),
            'player2_score': spaces.Box(low=0, high=nb_pieces_of_cheese, shape=(1,)),
            'player1_location': spaces.Tuple([spaces.Discrete(self.width), spaces.Discrete(self.height)]),
            'player2_location': spaces.Tuple([spaces.Discrete(self.width), spaces.Discrete(self.height)]),

        })

        # Follow the play :
        self.player1_last_move = None
        self.player2_last_move = None
        self.bg = None

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
        self.turn += 1
        # Perform both player's actions on the maze variables
        decision1, decision2 = DECISION_FROM_ACTION_DICT[action[0]], DECISION_FROM_ACTION_DICT[action[1]]
        self.player1_last_move = decision1
        self.player2_last_move = decision2
        self._move((decision1, decision2))

        reward = self._calculate_reward()

        # Calculate the return variables
        observation = self._observation()
        done = self._check_done()
        info = dict()

        return observation, reward, done, info

    def reset(self):
        # reset the maze randomly
        self.random_seed = random.randint(0, sys.maxsize)
        self.turn = 0
        self.pieces_of_cheese = []
        self.width, self.height, self.pieces_of_cheese, self.maze = generate_maze(self.maze_dimension[0],
                                                                                  self.maze_dimension[1],
                                                                                  self.target_density,
                                                                                  self.connected, self.symmetry,
                                                                                  self.mud_density, self.mud_range,
                                                                                  self.maze_file,
                                                                                  self.random_seed)
        self.pieces_of_cheese, self.player1_location, self.player2_location = generate_pieces_of_cheese(
            self.nb_pieces_of_cheese,
            self.maze_dimension[
                0],
            self.maze_dimension[
                1],
            self.symmetry,
            self.player1_location,
            self.player2_location,
            self.start_random)
        # Reset player turns, score, misses
        self.player1_score, self.player2_score, self.player1_misses, self.player2_misses, self.player1_moves, self.player2_moves = 0, 0, 0, 0, 0, 0
        self.player1_last_move = None
        self.player2_last_move = None
        self.bg = None
        # Reset wrapper attributes
        # Create the maze matrix
        self.maze_matrix_U = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_D = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_R = np.zeros((self.width, self.height), dtype=np.int8)
        self.maze_matrix_L = np.zeros((self.width, self.height), dtype=np.int8)
        self._maze_matrix_from_dict()
        # Create the cheese matrix
        self.cheese_matrix = np.zeros((self.width, self.height), dtype=np.int8)
        self._cheese_matrix_from_list()

        for start in self.maze:
            for end in self.maze[start]:
                self.maze[start][end] = 1

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

    # Utils methods

    def save_pickle(self, path="./maze_files/maze_save.p"):
        """
        pickles the maze to the given path
        :param path: the path to save to
        :return:
        """
        dump(self, open(path, "wb"))

    def _observation(self):
        return dict({
            'Maze_up': self.maze_matrix_U,
            'Maze_right': self.maze_matrix_R,
            'Maze_left': self.maze_matrix_L,
            'Maze_down': self.maze_matrix_D,
            'pieces_of_cheese': self.cheese_matrix,
            'player1_score': self.player1_score,
            'player2_score': self.player2_score,
            'player1_location': self.player1_location,
            'player2_location': self.player2_location,
        })

    def matrix_index_to_pos(self, index):
        # noinspection PyRedundantParentheses
        return (index % self.width, index // self.width)

    def pos_to_matrix_index(self, pos):
        return pos[1] * self.width + pos[0]

    def _maze_matrix_from_dict(self):
        """
        Generates the maze matrix
        :return:
        """
        maze_dict = self.maze
        for position in maze_dict:
            for destination in maze_dict[position]:
                direction = self._calculate_direction(position, destination)
                if direction == 'U':
                    self.maze_matrix_U[position[0], position[1]] = 1
                elif direction == 'D':
                    self.maze_matrix_D[position[0], position[1]] = 1
                elif direction == 'R':
                    self.maze_matrix_R[position[0], position[1]] = 1
                elif direction == 'L':
                    self.maze_matrix_L[position[0], position[1]] = 1

    def _cheese_matrix_from_list(self):
        for cheese in self.pieces_of_cheese:
            self.cheese_matrix[cheese] = 1

    def _calculate_reward(self):
        """
        Returns the reward for the current turn and removes the potential cheeses that have been eaten
        reward is 1 if the player 1 eats a piece of cheese and player 2 doesnt, -1 if player 2 does and player 1 doesnt and 0 in all other cases
        :return: reward
        """
        reward = 0
        if self.player1_location in self.pieces_of_cheese:
            self.pieces_of_cheese.remove(self.player1_location)
            self.cheese_matrix[self.player1_location] = 0
            if self.player2_location == self.player1_location:
                self.player2_score += 0.5
                self.player1_score += 0.5
                # reward = 0
            else:
                self.player1_score += 1
                reward = 1
        if self.player2_location in self.pieces_of_cheese:
            self.pieces_of_cheese.remove(self.player2_location)
            self.cheese_matrix[self.player2_location] = 0
            self.player2_score += 1
            reward = -1
        return reward

    def _move(self, action):
        """
        imports.maze.move function wrapper
        :param action: (decision1,decision2) of both players
        """
        (decision1, decision2) = action
        self.player1_location, self.player2_location, stuck1, stuck2, self.player1_moves, self.player2_moves, self.player1_misses, self.player2_misses = move(
            decision1, decision2, self.maze, self.player1_location, self.player2_location, 0, 0, self.player1_moves,
            self.player2_moves, self.player1_misses, self.player2_misses)

    def _check_done(self):
        # noinspection PyRedundantParentheses,PyRedundantParentheses
        return (self.turn >= self.max_turns) or (self.player1_score > (self.nb_pieces_of_cheese) / 2) or (
                self.player2_score > (self.nb_pieces_of_cheese) / 2)

    def _calculate_direction(self, source, destination):
        direction = None
        delta = (destination[0] - source[0], destination[1] - source[1])
        if delta == (0, 1):
            direction = 'U'
        elif delta == (0, -1):
            direction = 'D'
        elif delta == (1, 0):
            direction = 'R'
        elif delta == (-1, 0):
            direction = 'L'
        return direction
