

current_environment: dict = {}
# current_environment = env_frozen_lake.FrozenLakeEnv()
# current_environment = env_cliff_walking.CliffWalkingEnv()


def set_environment(env: dict):
    global current_environment
    current_environment = env
