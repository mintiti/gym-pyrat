import random

from pyrat_env.envs.Pyrat_Env import IntParameter, FloatParameter, BooleanParameter, GameGenerator, MazeGenerator, PyratEnv
from stable_baselines3.common.env_checker import check_env

import time

def test_int():
    print(f"starting test_int")
    # single scalar
    test_passed = True
    integer = IntParameter(4)
    for i in range(100):
        samp = integer.sample()
        passed = (samp == 4)
        print(passed)
        test_passed = test_passed and passed
    # interval
    integer = IntParameter([20,41])
    for i in range(100):
        samp = integer.sample()
        passed = (20<= samp <= 41) and isinstance(samp, int)
        print(passed)

        test_passed = test_passed and passed

    # float_interval:
    integer = IntParameter([20.0,41.0])
    for i in range(100):
        samp = integer.sample()
        passed = (20<= samp <= 41) and isinstance(samp, int)
        print(passed)
        test_passed = test_passed and passed


    print(f"test_int result : {test_passed}\n")

def test_float():
    print(f"starting test_float")
    test_passed = True
    # Single scalar value
    fl = FloatParameter(0.2)
    for i in range(100):
        samp = fl.sample()
        passed = (samp == 0.2)
        test_passed = test_passed and passed
    print(test_passed)
    # Interval
    fl = FloatParameter([0.2,0.7])
    for i in range(100):
        samp = fl.sample()
        passed = (0.2<= samp <= 0.7)
        test_passed = test_passed and passed
    print(test_passed)
    fl = FloatParameter([0.2,1])
    for i in range(100):
        samp = fl.sample()
        passed = (0.2<= samp <= 1)
        test_passed = test_passed and passed
    print(f"test_float result : {test_passed}\n")

def test_boolean():
    print(f"starting test_bool")
    test_passed = True
    b = BooleanParameter(True)
    for i in range(100):
        samp = b.sample()
        passed = (samp == True)
        test_passed = test_passed and passed
    print(test_passed)
    b = BooleanParameter(False)
    for i in range(100):
        samp = b.sample()
        passed = (samp == False)
        test_passed = test_passed and passed
    print(test_passed)

    b = BooleanParameter(0)
    for i in range(100):
        samp = b.sample()
        passed = (samp == False)
        test_passed = test_passed and passed
    print(test_passed)

    b = BooleanParameter(1)
    for i in range(100):
        samp = b.sample()
        passed = (samp == True)
        test_passed = test_passed and passed
    print(test_passed)

    b = BooleanParameter(0.8)
    a = 0
    for i in range(400):
        samp = b.sample()
        a += samp
    print("number of trues", a/400)
    passed = (0.75<= (a/400) <= 0.85)
    test_passed = test_passed and passed
    print(f"test_bool result : {test_passed}\n")

def test_game_generator():
    args = dict(width=21, height=15, nb_pieces_of_cheese=[20,41], target_density=0.7, mud_density=0, connected=True,
                 symmetry=True, mud_range=10, start_random=True)
    g = GameGenerator(**args)
    for i in range(3):
        game_state = g()
        print(game_state)

    # obs = game_state.get_obs()
    # print(obs.keys())

    game_state2 = g()
    print(game_state2)
    obs2 = game_state2.get_obs()

    print(game_state == game_state2)

    game_state.set_state(obs2)

    print(game_state)
    print(game_state == game_state2)


def test_maze_gen():
    gen = MazeGenerator()
    gen.width = 15
    gen.height = 21
    pieces, p1_pos, p2_pos = gen.generate_pieces_of_cheese(12,False,True)
    print(len(pieces))
    print(p1_pos,p2_pos)
    width, height, maze, pieces, p1_pos, p2_pos = gen(15,21,21,0.7,0.3,True,True,10,True,"", 142)

    print(width, height)
    print(maze)
    print(len(pieces))
    print(p1_pos,p2_pos)

def test_state():
    args = dict(width=21, height=15, nb_pieces_of_cheese=[20, 41], target_density=0, mud_density=0, connected=True,
                symmetry=True, mud_range=10, start_random=True)
    g = GameGenerator(**args)
    game_state = g()
    print(game_state)
    for i in range(10):
        decision1 = random.randint(0,3)
        decision2 = random.randint(0, 3)
        print("-----------")
        print(f"decisions : p1 {decision1}; p2 {decision2}")
        game_state.step((decision1,decision2))
        print(game_state,"\n")
        print(game_state.get_maze_matrix())
        print(game_state.get_cheese_matrix())

def test_env():
    args = dict(width=21, height=15, nb_pieces_of_cheese=[20, 41], target_density=0, mud_density=0, connected=True,
                symmetry=True, mud_range=10, start_random=True)
    env = PyratEnv(**args)
    print(env.action_space)
    print(env.observation_space)
    #test reset
    obs = env.reset()
    # print(obs)
    # test the observation space
    def test_obs_space(env):
        obs = env.reset()
        ok = True
        for i in obs:
            key_ok = env.observation_space[i].contains(obs[i])
            ok = key_ok and ok
            print(f"testing key {i}, key value is contained in the specified space :",key_ok)
            # print(env.observation_space[i].sample(), "current value :", obs[i])
            # print(env.observation_space[i].low)
            # print(i,obs[i].shape,"\n")
        entire_space_ok = env.observation_space.contains(obs)
        print(f"The observation is contained in th observation space : {entire_space_ok}")

        return ok and entire_space_ok

    test_obs_ok = test_obs_space(env)
    #test step
    def test_step(nb_steps,env):
        for i in range(nb_steps):
            obs, rew,done,info = env.step(env.action_space.sample())
            print(obs)
            print(f"observation is {obs}")
            print(f"reward is {rew}")
            print(f"game state is {env.state}")
        
    #test_step(100,env)
    # test flattening
    from gym.wrappers import FlattenObservation
    env = FlattenObservation(env)
    env.reset()
    print(env.observation_space)
    
    # test mud 
    args['mud_density'] = 0.7
    env = PyratEnv(**args)
    
    #test_step(100,env)
    
    args['mud_range'] = 3
    env= PyratEnv(**args)
    #test_step(100,env)

def time_steps(env,nb_trials):
    args = dict(width=21, height=15, nb_pieces_of_cheese=[20, 41], target_density=0, mud_density=0, connected=True,
                symmetry=True, mud_range=10, start_random=True)
    env.reset()
    times = []
    for i in range(nb_trials):
        action = env.action_space.sample()
        t0 = time.time()
        _, _ , done,_ = env.step(action)
        times.append(time.time()-t0)
        if done :
            env.reset()

    return sum(times) / len(times)

def test_time():
    args = dict(width=21, height=15, nb_pieces_of_cheese=10, target_density=0.7, mud_density=0, connected=True,
                symmetry=True, mud_range=10, start_random=False)
    env = PyratEnv(**args)
    nb_trials = 10000
    mean = time_steps(env,nb_trials)
    print(f"average time for a step : {mean} over {nb_trials} trials")

if __name__ == '__main__':
    test_time()
