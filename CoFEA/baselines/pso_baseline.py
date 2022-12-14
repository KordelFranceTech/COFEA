import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import model_utils_rl_pso as mu
from util.data import data_process_rl_osi as dp
from config import ConfigRL
from environments import env_frozen_lake, env_cliff_walking, environment
import models_rl as models
import random

env = {"map": env_cliff_walking.CliffWalkingEnv(), "type": "large"}
environment.set_environment(env)
env = env["map"]
env.reset()
agent = ConfigRL(model_name='q_learn', env=env)
model = models.create(agent.model_name, env)


def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total


def compute_reward(state: int):
    reward = 48 - (((4 - int(state / 12)) ** 2) + ((12 - (state % 12)) ** 2)) ** 0.5
    if state in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
        reward = -100
    return reward

def f(states):
    rewards = 0
    totalReward = {
        'SarsaAgent': [],
        'QLearningAgent': [],
        'ExpectedSarsaAgent': []
    }
    for i in range(len(states)):
        for episode in range(total_episodes):
            # Initialize the necessary parameters before
            # the start of the episode
            t = 0
            # state1 = env.reset()
            state1 = states[i]
            action1 = agent.choose_action(state1)
            episodeReward = 0
            while t < max_steps:

                # Getting the next state, reward, and other parameters
                state2, reward, done, info = env.step(action1)
                # reward = compute_reward(state2)

                # Choosing the next action
                action2 = agent.choose_action(state2)

                # Learning the Q-value
                agent.update(state1, state2, reward, action1, action2)
                # print(f'episode: {episode}\n\tstate: {state1}\taction: {action2}\treward: {reward}\tnew state: {state2}\ttarget: {target}')

                state1 = state2
                action1 = action2

                # Updating the respective vaLues
                t += 1
                episodeReward += reward

                # If at the end of learning process
                if done:
                    break
            # Append the sum of reward at the end of the episode
            totalReward[type(agent).__name__].append(episodeReward)
        rewards += np.mean(totalReward[type(agent).__name__])
    reward_error = -float(rewards / float(len(states)) ** 2)
    return reward_error


# --- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self, x0):
        self.position_i = []  # particle position
        self.velocity_i = []  # particle velocity
        self.pos_best_i = []  # best position individual
        self.err_best_i = -1  # best error individual
        self.err_i = -1  # error individual

        for i in range(0, num_dimensions):
            self.velocity_i.append(random.uniform(-1, 1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self, costFunc):
        self.err_i = costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i == -1:
            self.pos_best_i = self.position_i
            self.err_best_i = self.err_i

    # update new particle velocity
    def update_velocity(self, pos_best_g):
        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1  # cognitive constant
        c2 = 2  # social constant

        for i in range(0, num_dimensions):
            r1 = random.random()
            r2 = random.random()

            vel_cognitive = c1 * r1 * (self.pos_best_i[i] - self.position_i[i])
            vel_social = c2 * r2 * (pos_best_g[i] - self.position_i[i])
            self.velocity_i[i] = w * self.velocity_i[i] + vel_cognitive + vel_social

    # update the particle position based off new velocity updates
    def update_position(self, bounds):
        for i in range(0, num_dimensions):
            # print(f"position pre: {self.position_i[i]}")
            # self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            self.position_i[i] = self.position_i[i] + 1
            # print(f"position post: {self.position_i[i]}")

            # adjust maximum position if necessary
            if self.position_i[i] > bounds[i][1]:
                self.position_i[i] = bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i] = bounds[i][0]


class PSO():
    def __init__(self, costFunc, x0, bounds, num_particles, maxiter):
        global num_dimensions

        num_dimensions = len(x0)
        err_best_g = -1  # best error for group
        pos_best_g = []  # best position for group

        # establish the swarm
        swarm = []
        for i in range(0, num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i = 0
        while i < maxiter:
            # print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0, num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i += 1

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)


def print_map():
    map_str: str = ""
    reward_map_str: str = ""
    for i in range(len(model.Q)):
        actions = model.Q[i]
        action = np.argmax(actions)
        if action == 0:
            map_str += "^"
        elif action == 1:
            map_str += ">"
        elif action == 2:
            map_str += "v"
        elif action == 3:
            map_str += "<"
        else:
            map_str += "-"
        if (i + 1) % 12 == 0:
            map_str += "\n"
    for i in range(0, 48):
        reward_map_str += f"{int(compute_reward(i))} "
        if (i + 1) % 12 == 0:
            reward_map_str += "\n"
    print(map_str)
    print("----")
    print(reward_map_str)


if __name__ == "__main__":
    # --- RUN ----------------------------------------------------------------------+

    initial = [0]  # initial starting location [x1,x2...]
    bounds = [(0, 47)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
    PSO(f, initial, bounds, num_particles=5, maxiter=30)
    print_map()
    #--- END ----------------------------------------------------------------------+
