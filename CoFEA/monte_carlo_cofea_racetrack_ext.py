
import torch
from CoFEA import experiment as EXP
from CoFEA.model_utils import model_utils_rl_cofea_ext as mu
from CoFEA.util.data import data_process_rl as dp
from CoFEA.config import ConfigRL
from CoFEA.environments import env_frozen_lake, env_cliff_walking, env_racetrack, env_racetrack_v2, environment
import CoFEA.models_rl as models
import numpy as np
import torch.multiprocessing as mp


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)


def train_predict(net, train_data, untrain_data, test_data, config, device, pred_probs):
    mu.train(net, train_data, config, device)
    pred_probs.append(mu.predict_prob(net, untrain_data, configs[view], view))


def parallel_train(nets, train_data, data_dir, configs):
    processes = []
    for view, net in enumerate(nets):
        p = mp.Process(target=mu.train, args=(net, train_data, config, view))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()




def adjust_config(config, num_examples, iter_step):
    repeat = 20 * (1.1 ** iter_step)
    epochs = list(range(60, 20, -20))
    config.epochs = epochs[iter_step]
    config.epochs = int((50000 * repeat) // num_examples / 50)
    config.epochs = 20
    config.step_size = max(int(config.epochs // 3), 1)
    return config


def spaco_rl_osi(map,
                 map_type,
                 configs,
                 iter_steps=10,
                 gamma=0.8,
                 train_ratio=0.2,
                 regularizer='soft',
                 population_size=3,
                 n_generations = 3,
                 inertia_weight = 0.8,
                 cognitive_weight = 0.8,
                 social_weight = 0.8,
                 debug=False):
    """
    self-paced co-training model implementation based on Pytroch
    params:
    model_names: model names for spaco, such as ['resnet50','densenet121']
    data: dataset for spaco model
    save_pathts: save paths for two models
    iter_step: iteration round for spaco
    gamma: spaco hyperparameter
    train_ratio: initiate training dataset ratio
    """
    num_obs = len(configs)
    add_num = 40
    # train_env = gym.make('CliffWalking-v0')
    # untrain_env = gym.make('CliffWalking-v0')
    # test_env = gym.make('CliffWalking-v0')
    # train_env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    # untrain_env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    # test_env = gym.make('FrozenLake8x8-v1', is_slippery=False)
    # train_env = copy.deepcopy(map)
    # untrain_env = copy.deepcopy(map)
    # test_env = copy.deepcopy(map)
    map.reset()
    train_env = map
    untrain_env = map
    test_env = map
    # train_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    # untrain_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    # test_env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    global train_data
    train_data = []
    global untrain_data
    untrain_data = []
    test_data = []
    pred_probs = []
    test_preds = []
    sel_ids = []
    weights = []
    start_step = 0
    ###########
    # initiate classifier to get preidctions
    ###########

    for obs in range(num_obs):
        configs[obs] = adjust_config(configs[obs], 1000, 0)
        net = models.create(configs[obs].model_name, map)
        if debug:
            print(type(net))
        train_data, _, _ = mu.train(net, train_env, map_type, configs[obs])
        # acc = mu.evaluate(net, test_env, configs[obs], obs)
        # print(acc)
        # x
        # untrain_data = mu.get_randomized_q_table(net.Q, untrain_env)
        untrain_data, _, _ = mu.train(net, untrain_env, map_type, configs[obs])
        # train_data = train_data[:5]
        # untrain_data = untrain_data[:5]
        # print(mu.predict_prob(net, untrain_env, configs[obs], obs).shape)
        # print(np.concatenate(mu.predict_prob(net, untrain_env, configs[obs], obs),axis=0).shape)
        # x
        pred_probs.append(np.concatenate(mu.predict_prob(net, untrain_env, configs[obs], obs), axis=0).tolist())
        test_preds.append(np.concatenate(mu.predict_prob(net, test_env, configs[obs], obs), axis=0).tolist())
        # pred_probs.append(mu.predict_prob(net, untrain_env, configs[obs], obs).tolist())
        # test_preds.append(mu.predict_prob(net, test_env, configs[obs], obs).tolist())
        acc, p = mu.evaluate(net, test_env, configs[obs], obs)
        # print(f"accuracy is: {acc}")
        # save_checkpoint(
        #   {
        #     'state_dict': net.state_dict(),
        #     'epoch': 0,
        #   },
        #   False,
        #   fpath=os.path.join(

        #     'spaco/%s.epoch%d' % (configs[obs].model_name, 0)))
    # pred_y = [np.argmax(i) for i in pred_probs]
    # print(len(pred_probs[0]))
    pred_y = []
    for k in range(0, len(pred_probs[0])):
        # pred_y.append(np.array([np.argmax(i) for i in k]))
        a = pred_probs[0][k]
        b = pred_probs[1][k]
        # print(a)
        # print(b)
        pred_y.append(np.argmax([a, b]))
    # pred_y = pred_probs
    # print(len(pred_y))
    # print(len(pred_y[0]))


    # initiate weights for unlabled examples
    pred_probs = np.array(pred_probs)
    # print(pred_probs.shape)
    pred_probs = np.expand_dims(pred_probs, axis=0)
    # pred_probs = pred_probs.T
    # print(pred_probs.shape)

    pred_y = np.array(pred_y)
    # print(pred_y.shape)
    # print(pred_probs.shape)
    # print(pred_probs.shape)
    # print(pred_y[:5])
    # print(pred_probs[:5])
    for obs in range(0, 1):
        sel_id, weight = dp.get_ids_weights(pred_probs[obs],
                                            pred_y,
                                            train_data,
                                            add_num,
                                            gamma,
                                            regularizer)
        # import pdb;pdb.set_trace()
        sel_ids.append(sel_id)
        weights.append(weight)

    # start iterative training
    gt_y = test_env
    final_results: list = []
    for step in range(start_step, iter_steps):
        results: list = []
        for obs in range(0, 1):
            if debug:
                print('Iter step: %d, obs: %d, model name: %s' % (step+1,obs,configs[obs].model_name))

            # update sample weights
            sel_ids[obs], weights[obs] = dp.update_ids_weights(
              obs, pred_probs, sel_ids, weights, pred_y, train_data,
              add_num, gamma, regularizer)
            # update model parameter
            new_train_data, _ = dp.update_train_untrain_rl(
              sel_ids[obs], train_data, untrain_data, pred_y, weights[obs])
            configs[obs] = adjust_config(configs[obs], len(train_data), 0)
            new_train_data = train_data


            net = models.create(configs[obs].model_name, map)
            mu.train(net, train_env, map_type, configs[obs])

            # update y
            # print(pred_probs.shape)
            # (1, 2, 8000)
            # pred_probs.reshape(pred_probs.shape[0], pred_probs.shape[1])
            pred_probs[obs] = np.concatenate(mu.predict_prob(net, untrain_env, configs[obs], obs), axis=0)

            # evaluation current model and save it
            acc, p = mu.evaluate(net, test_env, configs[obs], obs)
            predictions = np.concatenate(mu.predict_prob(net, train_env, configs[obs], device=obs), axis=0)
            # save_checkpoint(
            #   {
            #     'state_dict': net.state_dict(),
            #     'epoch': step + 1,
            #     'predictions': predictions,
            #     'accuracy': acc
            #   },
            #   False,
            #   fpath=os.path.join(
            #     'spaco/%s.epoch%d' % (configs[view].model_name, step + 1)))
            test_preds[obs] = np.concatenate(mu.predict_prob(net, test_env, configs[obs], device=obs), axis=0)
            if debug:
                print(f"accuracy: {acc}\n\n_____")
            results.append(acc)

        final_results.append(results)
        add_num +=  4000 * num_obs
        fuse_y = []
        # print(len(test_preds)) # 2
        # print(len(test_preds[0])) #12000
        # print(len(test_preds[1])) #12000
        for k in range(0, len(test_preds[0])):
            a = test_preds[0][k]
            b = test_preds[1][k]
            fuse_y.append(np.argmax([a, b]))
        # fuse_y = np.array(fuse_y)
        # print(f"fuse_y: {fuse_y}")
        # print('Acc:%0.4f' % np.mean(fuse_y== gt_y))

    # i_res = 0.0
    # j_res = 0.0
    # for r in final_results:
    #     print(r)
    #     i_res += r[0]
    #     j_res += r[1]
    # i_res /= len(final_results)
    # j_res /= len(final_results)
    # return [i_res, j_res]
    # return final_results

    print(final_results)
    avg = sum([x[0] for x in final_results]) / len(final_results)
    print(avg)
    return avg


if __name__ == "__main__":
    e = EXP.ENV
    config1 = ConfigRL(model_name='e_sarsa_fea', env=e)
    config2 = ConfigRL(model_name='e_sarsa_fea', env=e)

    print(spaco_rl_osi(
          e["map"],
          e["type"],
          [config1, config2],
          iter_steps=EXP.COTRAIN_ITERS,
          gamma=EXP.COTRAIN_GAMMA,
          regularizer="soft"))
    print(f"\n\nTotal steps: {EXP.COUNTER}")
    print(f"Total swarm  updates: {EXP.SWARM_UPDATE_COUNTER}\n\n")
    EXP.COUNTER = 0
    EXP.SWARM_UPDATE_COUNTER = 0

# e = {"map":env_racetrack_v2.Racetrack(), "type": "L"}
# environment.set_environment(e)
# # env_cliff_walking.update_map_type()
# # print(e["map"].desc)
# print(f"obs space: {e['map'].observation_space}")
# print(f"action space: {e['map'].action_space}")
# config1 = ConfigRL(model_name='e_sarsa_fea', env=e)
# config2 = ConfigRL(model_name='e_sarsa_fea', env=e)
#
# print(spaco_rl_osi(
#       e["map"],
#       e["type"],
#       [config1, config2],
#       iter_steps=3,
#       gamma=0.8,
#       regularizer="soft"))
# e["map"].render()


# config1 = ConfigRL(model_name='q_learn_osi')
# config2 = ConfigRL(model_name='q_learn_osi')
#
# e = env_cliff_walking.CliffWalkingEnv()
# e.reset()
# environment.current_environment = e
# print(spaco_rl_osi(
#       e,
#       [config1, config2],
#       iter_steps=3,
#       gamma=0.8,
#       regularizer="soft"))
# x

#
# for i in range(0, 1000):
#     try:
#         spaco_rl_osi([config1, config2],
#           iter_steps=1,
#           gamma=0.8,
#           regularizer="soft")
#         sys.exit()
#     except AssertionError:
#         print("assertion error")
#     except IndexError:
#         print("index error")
#     except ValueError:
#         print("value error")


def gimme_results(N: int,
                  iter_steps: int = 1,
                  gamma: float = 0.3,
                  regularizer: str = "soft",
                  population_size=3,
                  n_generations=3,
                  inertia_weight=0.8,
                  cognitive_weight=0.8,
                  social_weight=0.8,
                  debug=False
                  ):
    results_dict: dict = {}
    results_dict["iter_steps"] = iter_steps
    results_dict["gamma"] = gamma
    results_dict["regularizer"] = regularizer
    results_dict["N"] = N
    results_dict["population_size"] = population_size
    results_dict["n_generations"] = n_generations
    results_dict["inertia_weight"] = inertia_weight
    results_dict["cognitive_weight"] = cognitive_weight
    results_dict["social_weight"] = social_weight
    maps = [{"map":env_cliff_walking.CliffWalkingEnv(), "type": "large"}]
    agents: list = ["e_sarsa_osi"]
    # agents: list = ["e_sarsa_osi", "q_learn_osi"]
    # agents: list = ["q_learn_osi"]
    for map in maps:
        environment.set_environment(map)
        for agent_i in agents:
            for agent_j in agents:
                # if agent_i == "sarsa" and agent_j == "q_learn":
                #     break
                # if agent_i == "q_learn" and agent_j == "sarsa":
                #     break
                # if agent_i == "q_learn_osi" and agent_j != "q_learn_osi":
                #     break
                # if agent_j == "q_learn_osi" and agent_i != "q_learn_osi":
                #     break

                print(f"agent i: {agent_i}")
                print(f"agent j: {agent_j}")

                config1 = ConfigRL(model_name=agent_i, env=map["map"])
                config2 = ConfigRL(model_name=agent_j, env=map["map"])
                success_runs: int = 0
                success_results: list = []

                # while success_runs < N:
                for v in range(100):
                    if success_runs == N:
                        break
                    try:
                        res = spaco_rl_osi(map["map"],
                                           [config1, config2],
                                           iter_steps=iter_steps,
                                           gamma=gamma,
                                           regularizer=regularizer,
                                           population_size=population_size,
                                           n_generations=n_generations,
                                           inertia_weight=inertia_weight,
                                           cognitive_weight=cognitive_weight,
                                           social_weight=social_weight,
                                           )
                        success_runs += 1
                        success_results.append(res)

                    except AssertionError:
                        print(f"{v} assertion error")
                    except IndexError:
                        print(f"{v} index error")
                    except ValueError:
                        print(f"{v} value error")
                    # except AttributeError:
                    #     print(f"{v} attribution error")
                model = f"{agent_i}-{agent_j}"
                # results_dict[model] = sum([i for i in success_results]) / N
                results_dict[model] = [success_results, sum([i for i in success_results]) / N]
    print(results_dict)
    return results_dict



# a = gimme_results(1, iter_steps=3, gamma=0.3, regularizer="soft", population_size=3, n_generations=3)
# with open('osi_results.txt', 'w') as f:
#     f.write(f"{a}\n\n")
# a = gimme_results(1, iter_steps=1, gamma=0.3, regularizer="soft", population_size=3, n_generations=3)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")
# a = gimme_results(1, iter_steps=1, gamma=0.3, regularizer="soft", population_size=10, n_generations=3)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")
# a = gimme_results(1, iter_steps=1, gamma=0.3, regularizer="soft", population_size=3, n_generations=10)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")
# a = gimme_results(1, iter_steps=1, gamma=0.3, regularizer="soft", population_size=10, n_generations=10)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")
# a = gimme_results(3, iter_steps=3, gamma=0.3, regularizer="soft", population_size=20, n_generations=3)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")


a = gimme_results(3, iter_steps=1, gamma=0.3, regularizer="soft", population_size=10, n_generations=3, inertia_weight=0.9)
with open('osi_results.txt', 'w') as f:
    f.write(f"{a}\n\n")
a = gimme_results(3, iter_steps=1, gamma=0.3, regularizer="soft", population_size=100, n_generations=3, inertia_weight=0.9)
with open('osi_results.txt', 'a') as f:
    f.write(f"{a}\n\n")
# a = gimme_results(3, iter_steps=2, gamma=0.3, regularizer="soft", population_size=10, n_generations=3, inertia_weight=0.9)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")
# a = gimme_results(3, iter_steps=2, gamma=0.3, regularizer="soft", population_size=100, n_generations=3, inertia_weight=0.9)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")


# a = gimme_results(3, iter_steps=1, gamma=0.3, regularizer="soft", population_size=10, n_generations=3, inertia_weight=0.5)
# with open('osi_results.txt', 'w') as f:
#     f.write(f"{a}\n\n")
# a = gimme_results(3, iter_steps=1, gamma=0.3, regularizer="soft", population_size=10, n_generations=3, inertia_weight=0.9)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")
# a = gimme_results(3, iter_steps=1, gamma=0.3, regularizer="soft", population_size=10, n_generations=3, inertia_weight=1.5)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")
# a = gimme_results(3, iter_steps=1, gamma=0.3, regularizer="soft", population_size=10, n_generations=3, inertia_weight=2.0)
# with open('osi_results.txt', 'a') as f:
#     f.write(f"{a}\n\n")


# a = gimme_results(100, iter_steps=1, gamma=0.5, regularizer="soft")
# with open('osi_results.txt', 'a') as f:
#         f.write(f"{a}\n\n")
# a = gimme_results(100, iter_steps=1, gamma=0.8, regularizer="soft")
# with open('osi_results.txt', 'a') as f:
#         f.write(f"{a}\n\n")
# a = gimme_results(100, iter_steps=3, gamma=0.5, regularizer="soft")
# with open('osi_results.txt', 'a') as f:
#         f.write(f"{a}\n\n")
# a = gimme_results(100, iter_steps=3, gamma=0.8, regularizer="soft")
# with open('osi_results.txt', 'a') as f:
#         f.write(f"{a}\n\n")
# a = gimme_results(10, iter_steps=1, gamma=0.3, regularizer="soft")
# with open('results_q.txt', 'a') as f:
#         f.write(f"{a}\n\n")
# a = gimme_results(10, iter_steps=1, gamma=0.5, regularizer="soft")
# with open('results_q.txt', 'a') as f:
#         f.write(f"{a}\n\n")
# a = gimme_results(10, iter_steps=3, gamma=0.3, regularizer="soft")
# with open('results_q.txt', 'a') as f:
#         f.write(f"{a}\n\n")
# a = gimme_results(10, iter_steps=3, gamma=0.5, regularizer="soft")
# with open('results_q.txt', 'a') as f:
#         f.write(f"{a}\n\n")
# a = gimme_results(10, iter_steps=3, gamma=0.8, regularizer="soft")
# with open('results.txt', 'a') as f:
#         f.write(f"{a}\n\n")

# a = gimme_results(10, iter_steps=3, gamma=3, regularizer="soft")
# with open('results_q.txt', 'a') as f:
#         f.write(f"{a}\n\n")

# gamma = 0.3
# {'e_sarsa-e_sarsa': [0.9791666666666666], 'e_sarsa-sarsa': [], 'e_sarsa-q_learn': [], 'sarsa-e_sarsa': [], 'sarsa-sarsa': [0.9166666666666666], 'sarsa-q_learn': [], 'q_learn-e_sarsa': [], 'q_learn-sarsa': [], 'q_learn-q_learn': [0.6458333333333334]}

# gamma 0.8
# {'e_sarsa-e_sarsa': [0.9166666666666666, 0.9270833333333333, 0.8958333333333333], 'e_sarsa-sarsa': [0.9166666666666667], 'e_sarsa-q_learn': [], 'sarsa-e_sarsa': [], 'sarsa-sarsa': [0.78125, 0.8125, 0.75], 'sarsa-q_learn': [], 'q_learn-e_sarsa': [], 'q_learn-sarsa': [], 'q_learn-q_learn': [0.6145833333333333, 0.6145833333333333, 0.6458333333333333]}

# {'e_sarsa-e_sarsa': [0.9166666666666666, 0.9166666666666666, 0.9375, 0.9166666666666666, 0.9166666666666666], 'e_sarsa-sarsa': [], 'sarsa-e_sarsa': [], 'sarsa-sarsa': [0.8055555555555555, 0.7916666666666666, 0.7569444444444443, 0.7708333333333334, 0.8263888888888888]}

"""
gamma = 0.8
Sarsa-E_Sarsa: 93.88%
E_Sarsa-E_Sarsa: 97.79%
Q-Q: 70.83%
Q-E_Sarsa: 60.42%
Sarsa-Sarsa: 75.0%
Q-Sarsa: 

gamma = 0.3
Sarsa-E_Sarsa: 93.88%
E_Sarsa-E_Sarsa: 97.79%
Q-Q: 60.42%
Q-E_Sarsa: 60.42%
Sarsa-Sarsa: 75.0%
Q-Sarsa: 
"""