import gym
env = gym.make('CliffWalking-v0')


# Defining all the required parameters
epsilon = 0.1
total_episodes = 500
max_steps = 100
alpha = 0.5
gamma = 1