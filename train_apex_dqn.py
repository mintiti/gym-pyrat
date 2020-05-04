from ray.tune.utils import merge_dicts

from pyrat_env.envs import PyratMultiAgent
import ray
from ray import tune
from ray.rllib.agents.dqn.apex import APEX_DEFAULT_CONFIG, ApexTrainer


APEX_DEFAULT_CONFIG['num_workers'] = 4
APEX_DEFAULT_CONFIG["num_envs_per_worker"] = 16
APEX_DEFAULT_CONFIG['use_pytorch'] = False
APEX_DEFAULT_CONFIG['env'] = PyratMultiAgent
APEX_DEFAULT_CONFIG['env_config'] = {'flatten' : True, 'target_density' : 0, 'symmetry' : False}
APEX_DEFAULT_CONFIG["buffer_size"] = 100000
APEX_DEFAULT_CONFIG["target_network_update_freq"] = 20000
if __name__ == '__main__':
    ray.init()
    tune.run("APEX", config = APEX_DEFAULT_CONFIG)