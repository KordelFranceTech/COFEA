import copy
from torch import nn
# from util.data_rl import data_process as dp
from FEA.FEA.factorarchitecture import FactorArchitecture
from environments import env_frozen_lake, env_cliff_walking, env_racetrack, env_racetrack_v2, environment
from benchmark import get_best_policy, get_best_policy_osi, get_benchmark_policy, print_policy_string, k
from copy import deepcopy
from operator import attrgetter
import numpy as np
import random

global AGENT
global ENV
global BOUNDS


def get_bounds(map_size: str):
    bounds = []
    if map_size == "small":
        bounds = [(0,3), (0,11)]
    elif map_size == "large":
        bounds = [(0, 7), (0, 23)]
    elif map_size == "mega":
        bounds = [(0, 15), (0, 47)]
    elif map_size == "giga":
        bounds = [(0, 31), (0, 95)]
    elif map_size == "L":
        bounds = [(0, 11), (0, 37)]
    elif map_size == "R":
        bounds = [(0, 28), (0, 30)]
    elif map_size == "P":
        bounds = [(0, 30), (0, 30)]
    return bounds


def compute_reward(state: int):
    reward = 48 - (((4 - int(state / 12)) ** 2) + ((12 - (state % 12)) ** 2)) ** 0.5
    if state in [37, 38, 39, 40, 41, 42, 43, 44, 45, 46]:
        reward = -100
    return reward


def f(states):
    rewards = 0
    total_episodes = 1
    max_steps = 10
    totalReward = {
        'SarsaFeaAgent': [],
        'QLearningFeaAgent': [],
        'ExpectedSarsaFeaAgent': []
    }
    for i in range(len(states)):
        for episode in range(total_episodes):
            # Initialize the necessary parameters before
            # the start of the episode
            t = 0
            # state1 = env.reset()
            state1 = round(states[i])
            action1 = AGENT.choose_action(state1)
            episodeReward = 0
            while t < max_steps:

                # Getting the next state, reward, and other parameters
                state2, reward, done, info = ENV.step(action1)
                # reward = compute_reward(state2)

                # Choosing the next action
                action2 = AGENT.choose_action(state2)

                # Learning the Q-value
                AGENT.update(state1, state2, reward, action1, action2)
                # print(f'episode: {episode}\n\tstate: {state1}\taction: {action2}\treward: {reward}\tnew state: {state2}\ttarget: {target}')

                state1 = state2
                action1 = action2

                # Updating the respective values
                t += 1
                episodeReward += reward

                # If at the end of learning process
                if done:
                    break
            # Append the sum of reward at the end of the episode
            totalReward[type(AGENT).__name__].append(episodeReward)
        rewards += np.mean(totalReward[type(AGENT).__name__])
    reward_error = -float(rewards / float(len(states)) ** 2)
    return reward_error


# --- MAIN ---------------------------------------------------------------------+
class Particle(object):
    def __init__(self, f, size, position=None, factor=None, global_solution=None, lbest_pos=None):
        self.f = f
        self.lbest_fitness = float('inf')
        self.dim = size
        self.factor = factor
        if position is None:
            self.position = np.random.uniform(BOUNDS[0][0], BOUNDS[1][1], size=size)
            self.lbest_position = np.array([x for x in self.position])
        elif position is not None:
            self.position = position
            self.lbest_position = lbest_pos
            self.lbest_fitness = self.calculate_fitness(global_solution, lbest_pos)
        self.velocity = np.zeros(size)
        self.fitness = self.calculate_fitness(global_solution)

    def __le__(self, other):
        if self.fitness is float:
            return self.fitness <= other.fitness

    def __lt__(self, other):
        if self.fitness is float:
            return self.fitness < other.fitness

    def __gt__(self, other):
        if self.fitness is float:
            return self.fitness > other.fitness

    def __eq__(self, other):
        return (self.position == other.position).all()

    def __str__(self):
        return ' '.join(
            ['Particle with current fitness:', str(self.fitness), 'and best fitness:', str(self.lbest_fitness)])

    def set_fitness(self, fit):
        self.fitness = fit
        if fit < self.lbest_fitness:
            self.lbest_fitness = deepcopy(fit)
            self.lbest_position = np.array([x for x in self.position])

    def set_position(self, position):
        self.position = np.array(position)

    def update_individual_after_compete(self, global_solution=None):
        fitness = self.calculate_fitness(global_solution)
        if fitness < self.lbest_fitness:
            self.lbest_fitness = deepcopy(fitness)
        self.fitness = fitness
        return self

    def calculate_fitness(self, glob_solution, position=None):
        # print(self.position)
        if glob_solution is None:
            fitness = self.f(self.position)
        else:
            solution = [x for x in glob_solution]
            if position is None:
                for i, x in zip(self.factor, self.position):
                    solution[i] = x
            else:
                for i, x in zip(self.factor, position):
                    solution[i] = x
            fitness = self.f(np.array(solution))
        return fitness

    def update_particle(self, omega, phi, global_best_position, v_max, global_solution=None):
        self.update_velocity(omega, phi, global_best_position, v_max)
        self.update_position(global_solution)

    def update_velocity(self, omega, phi, global_best_position, v_max):
        velocity = [x for x in self.velocity]
        n = self.dim

        inertia = np.multiply(omega, velocity)
        phi_1 = np.array([random.random() * phi for i in range(n)])  # exploration
        personal_exploitation = self.lbest_position - self.position  # exploitation
        personal = phi_1 * personal_exploitation
        phi_2 = np.array([random.random() * phi for i in range(n)])  # exploration
        social_exploitation = global_best_position - self.position  # exploitation
        social = phi_2 * social_exploitation
        new_velocity = inertia + personal + social
        self.velocity = np.array([self.clamp_value(v, -v_max, v_max) for v in new_velocity])

    # def update_position(self, global_solution=None):
    #     lo, hi = BOUNDS[0][0], BOUNDS[1][1]
    #     position = self.velocity + self.position
    #     self.position = np.array([self.clamp_value(p, lo, hi) for p in position])
    #     self.fitness = self.calculate_fitness(global_solution)

    # update the particle position based off new velocity updates
    def update_position(self, global_solution=None):
        for i in range(0, self.dim):
            # print(f"position pre: {self.position_i[i]}")
            # self.position_i[i] = self.position_i[i] + self.velocity_i[i]
            # self.position_i[i] = self.position_i[i] + 1
            self.position[i], reward, done, info = ENV.step(np.argmax(AGENT.Q.tolist()[round(self.position[i])]))
            self.position[i] += self.velocity[i]
            # if done: break
            # print(f"position post: {self.position_i[i]}")

            # adjust maximum position if necessary
            if self.position[i] > BOUNDS[i][1]:
                self.position[i] = BOUNDS[i][1]

            # adjust minimum position if neseccary
            if self.position[i] < BOUNDS[i][0]:
                self.position[i] = BOUNDS[i][0]
        self.fitness = self.calculate_fitness(global_solution)


    def clamp_value(self, to_clamp, lo, hi):
        if lo < to_clamp < hi:
            return to_clamp
        if to_clamp < lo:
            return to_clamp
        return hi


class PSO(object):
    def __init__(self, generations, population_size, function, dim, factor=None, global_solution=None, omega=0.729, phi=1.49618):
        self.pop_size = population_size
        self.pop = [Particle(function, dim, factor=factor, global_solution=global_solution) for x in range(population_size)]
        pos = [p.position for p in self.pop]
        with open('pso2.o', 'a') as file:
            file.write(str(pos))
            file.write('\n')

        self.omega = omega
        self.phi = phi
        self.f = function
        self.dim = dim
        pbest_particle = Particle(function, dim, factor=factor, global_solution=global_solution)
        pbest_particle.set_fitness(float('inf'))
        self.pbest_history = [pbest_particle]
        self.gbest = pbest_particle
        self.v_max = abs((BOUNDS[1][1] - BOUNDS[0][0]))
        self.generations = generations
        self.current_loop = 0
        self.factor = np.array(factor)
        self.global_solution = global_solution

    def find_current_best(self):
        sorted_ = sorted(np.array(self.pop), key=attrgetter('fitness'))
        return Particle(self.f, self.dim, position=sorted_[0].position, factor=self.factor,
                 global_solution=self.global_solution, lbest_pos=sorted_[0].lbest_position)

    def find_local_best(self):
        pass

    def update_swarm(self):
        if self.global_solution is not None:
            global_solution = [x for x in self.global_solution]
        else:
            global_solution = None
        omega, phi, v_max = self.omega, self.phi, self.v_max
        global_best_position = [x for x in self.gbest.position]
        for p in self.pop:
            p.update_particle(omega, phi, global_best_position, v_max, global_solution)
            for k in range(0, len(p.position)):
                state1 = round(p.position[k])
                action1 = AGENT.choose_action(state1)
                state2, reward, done, info = ENV.step(action1)
                action2 = AGENT.choose_action(state2)
                AGENT.update(state1, state2, reward, action1, action2)
                if done: break
        curr_best = self.find_current_best()
        self.pbest_history.append(curr_best)
        if curr_best.fitness < self.gbest.fitness:
            self.gbest = curr_best

    def replace_worst_solution(self, global_solution):
        # find worst particle
        self.global_solution = np.array([x for x in global_solution])
        self.pop.sort(key=attrgetter('fitness'))
        print('replacing')
        print(self.pop[-1], self.pop[0])
        partial_solution = [x for i, x in enumerate(global_solution) if i in self.factor] # if i in self.factor
        self.pop[-1].set_position(partial_solution)
        self.pop[-1].set_fitness(self.f(self.global_solution))
        curr_best = Particle(self.f, self.dim, position=self.pop[0].position, factor=self.factor,
                 global_solution=self.global_solution, lbest_pos=self.pop[0].lbest_position)
        random.shuffle(self.pop)
        if curr_best.fitness < self.gbest.fitness:
            self.gbest = curr_best

    def run(self):
        for i in range(self.generations):
            self.update_swarm()
            self.current_loop += 1
            # print(self.gbest)
        return self.gbest.position



class FEA:
    def __init__(self, function, fea_runs, generations, pop_size, factor_architecture, base_algorithm, continuous=True,
                 seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.function = function
        self.fea_runs = fea_runs
        self.base_alg_iterations = generations
        self.pop_size = pop_size
        self.factor_architecture = factor_architecture
        self.dim = factor_architecture.dim
        self.base_algorithm = base_algorithm
        self.global_solution = None
        self.global_fitness = np.inf
        self.solution_history = []
        self.set_global_solution(continuous)
        self.subpopulations = self.initialize_factored_subpopulations()

    def run(self):
        for fea_run in range(self.fea_runs):
            for alg in self.subpopulations:
                # print('NEW SUBPOPULATION\n---------------------------------------------------------------')
                # alg.run(fea_run)
                alg.run()
            self.compete()
            self.share_solution()
            print('fea run ', fea_run, self.global_fitness)

    def set_global_solution(self, continuous):
        if continuous:
            self.global_solution = np.random.uniform(BOUNDS[0][0], BOUNDS[1][1], size=self.factor_architecture.dim)
            self.global_fitness = self.function(self.global_solution)
            self.solution_history = [self.global_solution]

    def initialize_factored_subpopulations(self):
        fa = self.factor_architecture
        alg = self.base_algorithm
        return [
            alg(function=self.function, dim=len(factor), generations=self.base_alg_iterations, population_size=self.pop_size, factor=factor, global_solution=self.global_solution)
            for factor in fa.factors]

    def share_solution(self):
        """
        Construct new global solution based on best shared variables from all swarms
        """
        gs = [x for x in self.global_solution]
        print('global fitness found: ', self.global_fitness)
        print('===================================================')
        for alg in self.subpopulations:
            # update fitnesses
            alg.pop = [individual.update_individual_after_compete(gs) for individual in alg.pop]
            # set best solution and replace worst solution with global solution across FEA
            alg.replace_worst_solution(gs)

    def compete(self):
        """
        For each variable:
            - gather subpopulations with said variable
            - replace variable value in global solution with corresponding subpop value
            - check if it improves fitness for said solution
            - replace variable if fitness improves
        Set new global solution after all variables have been checked
        """
        sol = [x for x in self.global_solution]
        f = self.function
        curr_fitness = f(self.global_solution)
        for var_idx in range(self.dim):
            best_value_for_var = sol[var_idx]
            for pop_idx in self.factor_architecture.optimizers[var_idx]:
                curr_pop = self.subpopulations[pop_idx]
                pop_var_idx = np.where(curr_pop.factor == var_idx)
                position = [x for x in curr_pop.gbest.position]
                var_candidate_value = position[pop_var_idx[0][0]]
                sol[var_idx] = var_candidate_value
                new_fitness = f(sol)
                if new_fitness < curr_fitness:
                    print('smaller fitness found')
                    curr_fitness = new_fitness
                    best_value_for_var = var_candidate_value
            sol[var_idx] = best_value_for_var
        self.global_solution = sol
        self.global_fitness = f(sol)
        self.solution_history.append(sol)



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
        if i == (int(env_racetrack.TERMINAL_STATE[0]*row_size) + env_racetrack.TERMINAL_STATE[1]):
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
        reward_map_str += f"{int(compute_reward(i))} "
        if (i + 1) % size == 0:
            reward_map_str += "\n"
    print(map_str)
    print("----")
    print(reward_map_str)


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


def train_fea_model(model, env, env_type, config, initial=[0, 0], num_particles=8, generations=20, fea_iter=5, debug=False):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training config
        criterion
    """
    global AGENT, ENV, BOUNDS
    AGENT = model
    ENV = env
    bounds = get_bounds(env_type)
    BOUNDS = bounds
    totalReward = {
        type(model).__name__: [],
    }

    if debug:
        print(f"model name: {type(model).__name__}")
        print(f"reward: {totalReward}\n")

    # pso = PSO(generations=10, population_size=8, function=f, dim=len(BOUNDS))
    # pso.run()
    # print_map()
    # env.reset()

    fa = FactorArchitecture(dim=len(bounds))
    fa.diff_grouping(f, 0.1)
    # fa.overlapping_diff_grouping(_function=f, epsilon=0.1, m=0)
    # fa.factors = fa.random_grouping(min_groups=5, max_groups=15, overlap=True)
    # fa.factors = fa.linear_grouping(group_size=7, offset=5)
    # fa.ring_grouping(group_size=2)
    fa.get_factor_topology_elements()
    fea = FEA(f, fea_runs=fea_iter, generations=generations, pop_size=num_particles, factor_architecture=fa, base_algorithm=PSO)
    fea.run()

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
    trajectory, current_policy, benchmark_policy = train_fea_model(model, env, env_type, config)
    #  return model
    return trajectory, current_policy, benchmark_policy


def get_policy_accuracy(current: list, benchmark: list):
    count = 0
    for i in range(0, len(current)):
        if current[i] == benchmark[i]:
            count += 1
    return count / len(current)


# def predict_prob(model, trajectories, config, device):
    probs = []
    # c_dict: dict = {}
    # # print(trajectories)
    # for t in trajectories:
    #     if len(t) > 1:
    #         t0 = t[0]
    #
    #         t1 = t[1]
    #         print(c_dict.keys())
    #         if t0 in c_dict.keys():
    #             # print(t0)
    #             # print(c_dict)
    #             t1x = c_dict[str(t0)]
    #             print(c_dict[t0])
    #             print(t1)
    #             x
    #             t2x = t1x.append(t1)
    #             c_dict[str(t0)] = t2x
    #         else:
    #             # print(t)
    #             # print(t0)
    #             # print(t[1])
    #             c_dict[str(t0)] = [t1]
    #     print(c_dict)
    #
    # for k in c_dict.keys():
    #     l = c_dict[k]
    #     l_dict: dict = {}
    #     for l0 in l:
    #         if l0 in l_dict.keys():
    #             l1 = l_dict[l0]
    #             l_dict[l0] = l1 + 1
    #         else:
    #             l_dict[l0] = 1



# def predict_prob(model, data, config, device):
#     model.eval()
#     dataloader = dp.get_dataloader(data, config)
#     probs = []
#     with torch.no_grad():
#         for batch_idx, (inputs, targets, _) in enumerate(dataloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             output = model(inputs)
#             prob = nn.functional.softmax(output, dim=1)
#             probs += [prob.data.cpu().numpy()]
#     return np.concatenate(probs)


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
    print(probs)
    # return np.concatenate(probs)
    return probs
    # k = []
    # for i in range(0, env.observation_space.n):
    #     i0 = []
    #     for j in range(0, env.action_space.n):
    #         i0.append(random.randrange(0, 100) / 100)
    #     k.append(i0)
    # print(k)
    # print(len(k))
    # return k
    # return [item for sublist in rewards for item in sublist]

# def evaluate(model, data, config, device):
#     model.eval()
#     correct = 0
#     total = 0
#     dataloader = dp.get_dataloader(data, config)
#     with torch.no_grad():
#         for batch_idx, (inputs, targets, _) in enumerate(dataloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#     acc = 100. * correct / total
#     print('Accuracy on Test data: %0.5f' % acc)
#     return acc


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




