from gym.envs.registration import register

register(
    id='PyRatEnv-v0',
    entry_point='pyrat.envs:PyratEnv',
    nondeterministic=False,
    max_episode_steps=2000
)

register(
    id='PyratEnvNoMudVsGreedy-v0',
    entry_point='pyrat.envs:PyratEnvNoMudVsGreedy',
    nondeterministic=False,
    max_episode_steps=2000
)