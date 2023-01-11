import copy
from CoFEA import experiment as EXP
from CoFEA.benchmark import get_best_policy, get_best_policy_osi, get_benchmark_policy, print_policy_string, k
import numpy as np
import random

global AGENT
global ENV


def f(states, updates=EXP.TRAJECTORIES):
    rewards = 0
    total_episodes = EXP.TOTAL_EPISODES
    max_steps = EXP.MAX_STEPS
    totalReward = EXP.REWARDS_TRACKER

    for i in range(len(states)):
        for episode in range(total_episodes):
            # Initialize the necessary parameters before
            # the start of the episode
            t = 0
            # state1 = env.reset()
            state1 = states[i]
            action1 = AGENT.choose_action(state1)
            episodeReward = 0
            while t < max_steps:

                # Getting the next state, reward, and other parameters
                state2, reward1, done, info = ENV.step(action1)
                # reward = compute_reward(state2)

                # Choosing the next action
                action2 = AGENT.choose_action(state2)

                # Learning the Q-value
                AGENT.update(state1, state2, reward1, action1, action2)
                # print(f'episode: {episode}\n\tstate: {state1}\taction: {action2}\treward: {reward}\tnew state: {state2}\ttarget: {target}')

                rewards = 0
                state_prev_i = state2
                action_prev_i = action2
                for k in range(1, updates):
                    state_i, reward_i, done, info = ENV.step(action_prev_i)
                    action_i = AGENT.choose_action(state_i)

                    # Learning the Q-value
                    AGENT.update(state_prev_i, state_i, reward_i, action_prev_i, action_i)
                    state_prev_i = state_i
                    action_prev_i = action_i
                    rewards += reward_i

                state1 = state2
                action1 = action2

                # Updating the respective vaLues
                t += 1
                episodeReward += reward1 + rewards

                # If at the end of learning process
                if done:
                    break
            # Append the sum of reward at the end of the episode
            totalReward[type(AGENT).__name__].append(episodeReward)
        rewards += np.mean(totalReward[type(AGENT).__name__])
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
            # self.position_i[i] = self.position_i[i] + 1
            self.position_i[i], reward, done, info = ENV.step(np.argmax(AGENT.Q.tolist()[self.position_i[i]]))
            # if done: break
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
        err_best_g = -1000  # best error for group
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
                if swarm[j].err_i < err_best_g or err_best_g == -1000:
                    pos_best_g = list(swarm[j].position_i)
                    err_best_g = float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0, num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
                for k in range(0, len(swarm[j].position_i)):
                    state1 = swarm[j].position_i[k]
                    action1 = AGENT.choose_action(state1)
                    state2, reward, done, info = ENV.step(action1)
                    action2 = AGENT.choose_action(state2)
                    AGENT.update(state1, state2, reward, action1, action2)
                    if done: break
            i += 1

        # print final results
        print('FINAL:')
        print(pos_best_g)
        print(err_best_g)


def print_map(map_size: str):
    size: int = 48
    row_size: int = 12
    if map_size == "small":
        size = 48
        row_size = 12
    elif map_size == "large":
        size = 4*12*4
        row_size = 24
    elif map_size == "mega":
        size = 16*12*4
        row_size = 48
    elif map_size == "giga":
        size = 64*12*4
        row_size = 96
    elif map_size == "L":
        size = 407
        row_size = 37
    elif map_size == "R":
        size = 840
        row_size = 30
    elif map_size == "P":
        size = 900
        row_size = 30
    map_str: str = ""
    reward_map_str: str = ""
    for i in range(len(AGENT.Q)):
        actions = AGENT.Q[i]
        action = np.argmax(actions)
        if i == (int(EXP.TERMINAL_STATE[0]*row_size) + EXP.TERMINAL_STATE[1]):
            map_str += "G"
        # elif i == (int(INITIAL[0]*row_size) + INITIAL[1]):
        #     map_str += "S"
        elif action == 0:
            map_str += "^"
        elif action == 1:
            map_str += ">"
        elif action == 2:
            map_str += "v"
        elif action == 3:
            map_str += "<"
        else:
            map_str += "-"
        if (i + 1) % row_size == 0:
            map_str += "\n"
    for i in range(0, size):
        reward_map_str += f"{int(EXP.compute_reward(i))} "
        if (i + 1) % size == 0:
            reward_map_str += "\n"
    print(map_str)
    print("----")


def build_trajectories(agent, e, config):
    model = copy.deepcopy(agent)
    env = copy.deepcopy(e)
    q_table = model.Q
    trajectories: list = []
    n, max_steps = config.epochs, config.max_steps
    rewards = []
    num_steps = []
    state1 = env.reset()
    action1 = model.choose_action(state1)
    for episode in range(n):
        total_reward = 0
        for i in range(max_steps):
            state2, reward, done, info = env.step(action1)
            action2 = model.choose_action(state2)
            # trajectory.append([state1, action1, state2, action2])
            trajectories.append([state1, action1])

            state1 = state2
            action1 = action2

            total_reward += reward
            if done:
                rewards.append(total_reward)
                num_steps.append(i + 1)
                break
    env.close()
    del env
    del model
    return trajectories


def train_pso_model(model, env, env_type, config, debug=False):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training config
        criterion
    """
    global AGENT, ENV
    AGENT = model
    ENV = env
    totalReward = {
        type(model).__name__: [],
    }

    if debug:
        print(f"model name: {type(model).__name__}")
        print(f"reward: {totalReward}\n")

    env.reset()
    PSO(f, EXP.INITIAL, EXP.BOUNDS, num_particles=EXP.NUM_PARTICLES, maxiter=EXP.MAX_ITER)
    print_map(env_type)
    current_policy = get_best_policy(q_table=model.Q)
    benchmark_policy = get_best_policy(get_benchmark_policy(type(model).__name__))
    if debug:
        print(f"accuracy: {get_policy_accuracy(current_policy, benchmark_policy)}")

    trajectories: list = build_trajectories(AGENT, ENV, config)

    # print(f"model name: {type(model).__name__}")
    # print(f"reward: {totalReward}\n")
    return trajectories, current_policy, benchmark_policy


def train(model, env, env_type, config):
    #  model = models.create(config.model_name)
    #  model = nn.DataParallel(model).cuda()
    # dataloader = dp.get_dataloader(train_data, config, is_training=True)
    trajectory, current_policy, benchmark_policy = train_pso_model(model, env, env_type, config)
    #  return model
    return trajectory, current_policy, benchmark_policy


def get_policy_accuracy(current: list, benchmark: list):
    count = 0
    for i in range(0, len(current)):
        if current[i] == benchmark[i]:
            count += 1
    return count / len(current)


def predict_prob(model, env, config, device):
    q_table = model.Q
    n, max_steps = config.epochs, config.max_steps
    rewards = []
    num_steps = []
    probs = []
    for episode in range(n):
        s = env.reset()
        total_reward = 0
        for i in range(max_steps):
            a = np.argmax(q_table[s, :])
            prob = softmax(q_table[s, :])
            probs += [prob]
            s, r, done, info = env.step(a)
            total_reward += r
            if done:
                # rewards.append([total_reward])
                num_steps.append(i + 1)
                break
            rewards.append([total_reward])
    env.close()
    # print(rewards)
    # print(probs)
    # return np.concatenate(probs)
    return probs


def evaluate(model, env, config, device):
    q_table = model.Q
    n, max_steps = config.epochs, config.max_steps
    rewards = []
    num_steps = []
    for episode in range(n):
        s = env.reset()
        total_reward = 0
        for i in range(max_steps):
            a = np.argmax(q_table[s, :])
            # a = np.argmax(q_table.best_individual.predict(np.identity(env.observation_space.n)[s, :]))
            s, r, done, info = env.step(a)
            total_reward += r
            if done:
                rewards.append(total_reward)
                num_steps.append(i + 1)
                break
    env.close()
    # return 100*np.sum(rewards)/len(rewards)
    current_policy = get_best_policy(q_table=model.Q)
    benchmark_policy = get_best_policy(get_benchmark_policy(type(model).__name__))
    accuracy = get_policy_accuracy(current_policy, benchmark_policy)

    curr_policy_str: str = print_policy_string(benchmark_policy)
    print(f"\n\n\tOPTIMAL POLICY:\n{curr_policy_str}")
    policy_str: str = print_policy_string(current_policy)
    print(f"\n\n\tCURRENT POLICY:\n{policy_str}")
    return accuracy, policy_str


def get_state_action_table(q_table):
    table: list = []
    for i in range(len(q_table)):
        for j in range(len(q_table[0])):
            table.append([i, j])
    return table


def get_randomized_q_table(q_table, env):
    table: list = []
    for i in range(len(q_table)):
        for j in range(len(q_table[0])):
            table.append(random.randrange(env.action_space.n))
    return table


def get_zeroed_q_table(q_table, env):
    return np.zeros([env.observation_space.n, env.action_space.n])


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)




