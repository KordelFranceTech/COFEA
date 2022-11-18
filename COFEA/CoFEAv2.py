import model_utils_rl_osi as mu
from config import ConfigRL
from environments import env_frozen_lake, env_cliff_walking, environment
import models_rl as models



def spaco_rl_osi(map,
                 configs,
                 iter_steps=10,
                 gamma=0.0,
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
    net = models.create(configs[0].model_name)
    traj, current_policy, benchmark_policy, total_reward = mu.train(net, map, configs[0])
    acc, p = mu.evaluate(net, map, configs[0], 0)
    print(f"accuracy is: {acc}")
    print(f"total reward: {total_reward}")
    return p


#
config1 = ConfigRL(model_name='e_sarsa_osi')
config2 = ConfigRL(model_name='e_sarsa_osi')
# config1 = ConfigRL(model_name='q_learn_osi')
# config2 = ConfigRL(model_name='q_learn_osi')

e = env_cliff_walking.CliffWalkingEnv()
# e = env_frozen_lake.FrozenLakeEnv()
environment.current_environment = e
print(spaco_rl_osi(
      e,
      [config1, config2],
      iter_steps=3,
      gamma=0.8,
      regularizer="soft"))

x


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
    maps = [env_cliff_walking.CliffWalkingEnv()]
    # agents: list = ["e_sarsa_osi", "sarsa_osi", "q_learn_osi"]
    agents: list = ["e_sarsa_osi"]
    for map in maps:
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

                config1 = ConfigRL(model_name=agent_i)
                config2 = ConfigRL(model_name=agent_j)
                success_runs: int = 0
                success_results: list = []

                # while success_runs < N:
                for v in range(100):
                    if success_runs == N:
                        break
                    try:
                        res = spaco_rl_osi(map,
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
                        map.render()

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
    x
    return results_dict



# a = gimme_results(1, iter_steps=1, gamma=0.3, regularizer="soft", population_size=3, n_generations=3)
# with open('osi_results.txt', 'w') as f:
#     f.write(f"{a}\n\n")

a = gimme_results(1, iter_steps=10, gamma=0.3, regularizer="soft", population_size=10, n_generations=3)
with open('osi_results.txt', 'a') as f:
    f.write(f"{a}\n\n")
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