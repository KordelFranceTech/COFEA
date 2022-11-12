import torch
from torch import nn
# from util.data_rl import data_process as dp
from benchmark import get_best_policy, get_best_policy_osi, benchmark_q_table
import numpy as np
import random



def train_model(model, env, config, debug=False):
    """
    train model given the dataloader the criterion,
    stop when epochs are reached
    params:
        model: model for training
        dataloader: training data
        config: training config
        criterion
    """
    global episodeReward
    episodeReward = 0
    totalReward = {
        type(model).__name__: [],
    }

    if debug:
        print(f"model name: {type(model).__name__}")
        print(f"reward: {totalReward}\n")
    global trajectory
    trajectory = []

    for epoch in range(0, config.epochs):
        # Initialize the necessary parameters before
        # the start of the episode
        t = 0
        state1 = env.reset()
        action1 = model.choose_action_osi(state1, env)
        episodeReward = 0
        while t < config.max_steps:

            # Getting the next state, reward, and other parameters
            state2, reward, done, info = env.step(action1)

            # Choosing the next action
            action2 = model.choose_action_osi(state2, env)

            # Learning the Q-value
            model.update(state1, state2, reward, action1, action2)
            # trajectory.append([state1, action1, state2, action2])
            trajectory.append([state1, action1])
            state1 = state2
            action1 = action2

            # Updating the respective values
            t += 1
            episodeReward += reward

            # If at the end of learning process
            if done:
                break
        # Append the sum of reward at the end of the episode
        totalReward[type(model).__name__].append(episodeReward)
    # print(f"q_table: {model.Q}")
    current_policy = get_best_policy_osi(pso_nn=model.Q, env=env)
    benchmark_policy = get_best_policy(benchmark_q_table)
    if debug:
        print(f"accuracy: {get_policy_accuracy(current_policy, benchmark_policy)}")

    # print(f"model name: {type(model).__name__}")
    # print(f"reward: {totalReward}\n")
    return trajectory, current_policy, benchmark_policy


def train(model, env, config):
    #  model = models.create(config.model_name)
    #  model = nn.DataParallel(model).cuda()
    # dataloader = dp.get_dataloader(train_data, config, is_training=True)
    trajectory, current_policy, benchmark_policy = train_model(model, env, config)
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
    # print(probs)
    return np.concatenate(probs)
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
            s, r, done, info = env.step(a)
            total_reward += r
            if done:
                rewards.append(total_reward)
                num_steps.append(i + 1)
                break
    env.close()
    # return 100*np.sum(rewards)/len(rewards)
    current_policy = get_best_policy(q_table=model.Q)
    benchmark_policy = get_best_policy(q_table=benchmark_q_table)
    accuracy = get_policy_accuracy(current_policy, benchmark_policy)
    return accuracy




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




