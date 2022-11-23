from gym.envs.registration import register

register(
    id='gyms/TextWorld-v0',
    entry_point='gyms.envs:TextWorldEnv',
    max_episode_steps=30,
)
