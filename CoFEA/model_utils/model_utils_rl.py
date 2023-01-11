from benchmark import get_best_policy, get_best_policy_osi, get_benchmark_policy, print_policy_string, k
import numpy as np
import random



def train_model(model, env, env_type, config, debug=False):
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
        action1 = model.choose_action(state1)
        episodeReward = 0
        while t < config.max_steps:

            # Getting the next state, reward, and other parameters
            state2, reward, done, info = env.step(action1)

            # Choosing the next action
            action2 = model.choose_action(state2)

            # Learning the Q-value
            model.update(state1, state2, reward, action1, action2)
            # trajectory.append([state1, action1, state2, action2])
            trajectory.append([state1, action1])
            state1 = state2
            action1 = action2

            # Updating the respective vaLues
            t += 1
            episodeReward += reward

            # If at the end of learning process
            if done:
                break
        # Append the sum of reward at the end of the episode
        totalReward[type(model).__name__].append(episodeReward)
    # print(f"q_table: {model.Q}")
    current_policy = get_best_policy(q_table=model.Q)
    benchmark_policy = get_best_policy(get_benchmark_policy(type(model).__name__))
    if debug:
        print(f"accuracy: {get_policy_accuracy(current_policy, benchmark_policy)}")

    # print(f"model name: {type(model).__name__}")
    # print(f"reward: {totalReward}\n")
    return trajectory, current_policy, benchmark_policy


def train(model, env, env_type, config):
    #  model = models.create(config.model_name)
    #  model = nn.DataParallel(model).cuda()
    # dataloader = dp.get_dataloader(train_data, config, is_training=True)
    trajectory, current_policy, benchmark_policy = train_model(model, env, env_type, config)
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
    print(probs)
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




