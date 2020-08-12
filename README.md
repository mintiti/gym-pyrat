# Pyrat game environment
Refactoring on the original game [PyRat](https://github.com/vgripon/PyRat), from the algorithmic discrete maths course taught in IMT Atlantique.  
This makes it compliant to the Gym API.

## Game
2 rats are in a maze and are competing for cheese pieces.  
Player with the most cheese pieces at the end of the game wins.


### Dependencies
You will need :  
* gym
* numpy
* pygame

## Installation 
Recommended installation :   
With gym-pyrat as the working directory :
```bash
pip install -e .
```

## Included environments  
| Environment                    | Implemented        | Number of players | Observation space     | Action space                   | Other                                                                                 |
|--------------------------------|--------------------|-------------------|-----------------------|--------------------------------|---------------------------------------------------------------------------------------|
| PyratEnv-v0                    | :heavy_check_mark: | 2                 | Dict containing Boxes | Tuple(Discrete(4),Discrete(4)) | Base Env                                                |
| PyratEnvNoMudVsGreedy-v0       | :heavy_check_mark: | 1                 | Dict containing Boxes | Discrete(4)                    | Same as PyratEnv-v0, but the agent plays against greedy, and there is no mud          |
| PyratEnvNoMudNoWallVsGreedy-v0 | :x:                | 1                 | Dict containing Boxes | Discrete(4)                    | Same as PyratEnv-v0, but the agent plays against greedy, and there is no mud or walls |

## Environment parameters 
You can change most environments' parameters, such as wall density, mud density and others.  
Most parameters can be sampled from a range.

Example : 
```python
from pyrat.envs import PyratEnv

# Default maze
env = PyratEnv()

# No mud
env = PyratEnv(mud_density=0)

# 11x15 maze
env = PyratEnv(width = 11, height = 15)

# Default maze with an initial number of cheeses uniformly sampled from [20,41]
env = PyratEnv(nb_pieces_of_cheese = [20,41])
```

For more details, please refer to the [GameGenerator class](https://github.com/mintiti/gym-pyrat/blob/092935a27e50b0238a8f57dc05242071b4ec67cc/pyrat/envs/core.py#L553) documentation

## Future improvements
- Wrapper to make the observation space a stack of planes to enable CNN usage.
- [RLlib Multiagent implementation](https://docs.ray.io/en/master/rllib-env.html#multi-agent-and-hierarchical)
- [Muzero General wrapper](https://github.com/werner-duvaud/muzero-general)