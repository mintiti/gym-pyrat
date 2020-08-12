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
while being in the gym-pyrat directory :
```bash
pip install -e .
```

## Included environments
| Environment                    | Implemented        | Number of players | Observation space     | Action space                   | Other                                                                                 |
|--------------------------------|--------------------|-------------------|-----------------------|--------------------------------|---------------------------------------------------------------------------------------|
| PyratEnv-v0                    | :heavy_check_mark: | 2                 | Dict containing Boxes | Tuple(Discrete(4),Discrete(4)) | Base Env                                                |
| PyratEnvNoMudVsGreedy-v0       | :heavy_check_mark: | 1                 | Dict containing Boxes | Discrete(4)                    | Same as PyratEnv-v0, but the agent plays against greedy, and there is no mud          |
| PyratEnvNoMudNoWallVsGreedy-v0 | :x:                | 1                 | Dict containing Boxes | Discrete(4)                    | Same as PyratEnv-v0, but the agent plays against greedy, and there is no mud or walls |