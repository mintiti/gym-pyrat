from ray.tune.utils import merge_dicts

from pyrat_env.envs import PyratMultiAgent
import ray
from ray import tune
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG, ApexTrainer
APEX_DEFAULT_CONFIG['num_workers'] = 4
APEX_DEFAULT_CONFIG['use_pytorch'] = False
APEX_DEFAULT_CONFIG['env'] = PyratMultiAgent
if __name__ == '__main__':
    env = PyratMultiAgent()
    obs = env.reset()
    ray.init()
    tune.run(ApexTrainer, config = APEX_DEFAULT_CONFIG)