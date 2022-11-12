import gym
import numpy as np
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense


env = gym.make('FrozenLake-v0')
# print(gym.envs.toy_text.frozen_lake.generate_random_map(size=8, p=0.8))
'''
    4 x 4 map
    S F F F       (S: starting point, safe)
    F H F H       (F: frozen surface, safe)
    F F F H       (H: hole, fall to your doom)
    H F F G       (G: goal, where the frisbee is located)

    8 x 8 map
    "SFFFFFFF"
    "FFFFFFFF"
    "FFFHFFFF"
    "FFFFFHFF"
    "FFFHFFFF"
    "FHHFFFHF"
    "FHFFHFHF"
    "FFFHFFFG"
'''

discount_factor = 0.95
eps = 0.5
eps_decay_factor = 0.999
learning_rate = 0.8
num_episodes = 500

# actions = [0: left, 1: down, 2: right, 3: up]
actions = ['left', 'down', 'right', 'up']



# REGULAR Q-LEARNING
print('#######################################################################')
print('#######################################################################')
print('#######################################################################')
print('TRAINING Q-LEARNING ALGORITHM')
q_table = np.zeros([env.observation_space.n,env.action_space.n])
print(f'Q-table before training:\n  action\n{q_table}')

for i in range(num_episodes):
    print('####################################################################')
    print(f'episode #: {i}')
    state = env.reset()
    eps *= eps_decay_factor
    done = False
    while not done:
        if np.random.random() < eps or np.sum(q_table[state, :]) == 0:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(q_table[state, :])
        new_state, reward, done, _ = env.step(action)
        q_table[state, action] += reward + learning_rate * (discount_factor * np.max(q_table[new_state, :]) - q_table[state, action])
        print(f'state: {state}\taction: {actions[action]}\treward: {reward}\tnew state: {new_state}\ttarget: {q_table[state, action]}')
        state = new_state
print(f'Q-table after training:\n   action\n{q_table}\n\n\n')



# DEEP Q-LEARNING
print('#######################################################################')
print('#######################################################################')
print('#######################################################################')
print('TRAINING DEEP Q-LEARNING NEURAL NETWORK')
model = Sequential()
model.add(InputLayer(batch_input_shape=(1, env.observation_space.n)))
model.add(Dense(20, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
print(model.summary())

for i in range(num_episodes):
    print('####################################################################')
    print(f'episode #: {i}')
    state = env.reset()
    eps *= eps_decay_factor
    done = False
    while not done:
        if np.random.random() < eps:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(
              model.predict(np.identity(env.observation_space.n)[state:state + 1]))
        new_state, reward, done, _ = env.step(action)
        target = reward + discount_factor * np.max(model.predict(np.identity(env.observation_space.n)[new_state:new_state + 1]))
        target_vector = model.predict(np.identity(env.observation_space.n)[state:state + 1])[0]
        print(f"target vector: {target_vector}")
        target_vector[action] = target
        model.fit(np.identity(env.observation_space.n)[state:state + 1],
                  target_vector.reshape(-1,
                                     env.action_space.n),
                  epochs=1,
                  verbose=0)
        print(f'state: {state}\taction: {actions[action]}\treward: {reward}\tnew state: {new_state}\ttarget: {target}')
        state = new_state

print(f'Q-table after training:\n   action\n{q_table}\n\n\n')
