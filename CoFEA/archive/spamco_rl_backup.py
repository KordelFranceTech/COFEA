import os
import torch
import gym
import argparse
# import model_utils as mu
import model_utils_rl as mu
# from util.data import data_process as dp
from util.data import data_process_rl as dp
# from config import Config
from config import ConfigRL
from util.serialization import load_checkpoint, save_checkpoint
import datasets
# import models
import models_rl as models
import numpy as np
import torch.multiprocessing as mp

# parser = argparse.ArgumentParser(description='soft_spaco')
# parser.add_argument('-s', '--seed', type=int, default=0)
# parser.add_argument('-r', '--regularizer', type=str, default='hard')
# parser.add_argument('-d', '--dataset', type=str, default='cifar10')
# parser.add_argument('--gamma', type=float, default=0.3)
# parser.add_argument('--iter-steps', type=int, default=5)
# parser.add_argument('--num-per-class', type=int, default=400)

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
    #  epochs = list(range(300, 20, -20))
    #  config.epochs = epochs[iter_step]
    #  config.epochs = int((50000 * repeat) // num_examples)
    # config.epochs = 200
    # config.step_size = max(int(config.epochs // 3), 1)
    return config


def spaco(configs,
          iter_steps=10,
          gamma=0,
          train_ratio=0.2,
          regularizer='soft'):
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
    add_num = 4000
    train_env = gym.make('CliffWalking-v0')
    untrain_env = gym.make('CliffWalking-v0')
    test_env = gym.make('CliffWalking-v0')
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
        configs[obs] = adjust_config(configs[obs], 1, 0)
        net = models.create(configs[obs].model_name)
        print(type(net))
        train_data = mu.train(net, train_env, configs[obs])
        # untrain_data = mu.get_randomized_q_table(net.Q, untrain_env)
        untrain_data = mu.train(net, untrain_env, configs[obs])
        pred_probs.append(mu.predict_prob(net, untrain_env, configs[obs], obs).tolist())
        test_preds.append(mu.predict_prob(net, test_env, configs[obs], obs).tolist())
        acc = mu.evaluate(net, test_env, configs[obs], obs)
        print(f"accuracy is: {acc}")
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
    for obs in range(0, 1):
        sel_id, weight = dp.get_ids_weights(pred_probs[obs], pred_y,
                                            train_data, add_num, gamma,
                                            regularizer)
        # import pdb;pdb.set_trace()
        sel_ids.append(sel_id)
        weights.append(weight)



    # start iterative training
    gt_y = test_env
    for step in range(start_step, iter_steps):
        for obs in range(0, 1):
            print('Iter step: %d, obs: %d, model name: %s' % (step+1,obs,configs[obs].model_name))

            # update sample weights
            sel_ids[obs], weights[obs] = dp.update_ids_weights(
              obs, pred_probs, sel_ids, weights, pred_y, train_data,
              add_num, gamma, regularizer)
            # update model parameter
            new_train_data, _ = dp.update_train_untrain_rl(
              sel_ids[obs], train_data, untrain_data, pred_y, weights[obs])
            configs[obs] = adjust_config(configs[obs], 1, 0)
            new_train_data = train_data

            net = models.create(configs[obs].model_name)
            mu.train(net, train_env, configs[obs])

            # update y
            # print(pred_probs.shape)
            # pred_probs.reshape(pred_probs.shape[0], pred_probs.shape[1])
            pred_probs[obs] = mu.predict_prob(net, untrain_env,
                                               configs[obs], obs)

            # evaluation current model and save it
            acc = mu.evaluate(net, test_env, configs[obs], obs)
            predictions = mu.predict_prob(net, train_env, configs[obs], device=obs)
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
            test_preds[obs] = mu.predict_prob(net, test_env, configs[obs], device=obs)
        add_num +=  4000 * num_obs
        fuse_y = []
        for k in range(0, len(test_preds[0])):
            a = test_preds[0][k]
            b = test_preds[1][k]
            fuse_y.append(np.argmax([a, b]))
        fuse_y = np.array(fuse_y)
        print('Acc:%0.4f' % np.mean(fuse_y== gt_y))



config1 = ConfigRL(model_name='sarsa')
config2 = ConfigRL(model_name='sarsa')

dataset = "cifar10"
cur_path = os.getcwd()
logs_dir = os.path.join(cur_path, 'logs')
data_dir = os.path.join(cur_path, 'data', dataset)
# data = datasets.create(dataset, data_dir)

spaco([config1, config2],
      iter_steps=3,
      gamma=0.3,
      regularizer="soft")
