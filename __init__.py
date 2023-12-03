from gymnasium.envs.registration import register

register(
    id="MasterMind-v0",
    entry_point="envs:MasterMindEnv",
    max_episode_steps=300,
)
