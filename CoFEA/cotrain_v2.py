import torch
from CoFEA import experiment as EXP
from CoFEA.model_utils import model_utils_rl_cofea as mu
from CoFEA.util.data import data_process_rl as dp
from CoFEA.config import ConfigRL
from CoFEA.environments import env_frozen_lake, env_cliff_walking, env_racetrack, env_racetrack_v2, environment
import CoFEA.models_rl as models
import numpy as np
import torch.multiprocessing as mp


def cotrain(map,
            map_type,
            configs,
            save_paths,
            iter_step=1,
            train_ratio=0.2):
    """
    cotrain model:
    params:
    model_names: model names such as ['resnet50','densenet121']
    data: dataset include train and untrain data
    save_paths: paths for storing models
    iter_step: maximum iteration steps
    train_ratio: labeled data ratio
    """
    num_obs = len(configs)
    add_num = 40
    map.reset()
    train_env = map
    untrain_env = map
    test_env = map

    assert iter_step >= 1
    # assert len(model_names) == 2 and len(save_paths) == 2
    # train_data,untrain_data = dp.split_dataset(data.trainval, train_ratio)
    train_data, untrain_data = train_env, untrain_env
    for step in range(iter_step):
        pred_probs = []
        add_ids = []
        for view in range(2):
            net = models.create(configs[view].model_name, map)
            model = mu.train(net,train_data,
                             map_type, configs[view])
            # data_params = mu.get_params_by_name(model_names[view])
            pred_probs.append(mu.predict_prob(
                net,untrain_data, configs[view], view))
            add_ids.append(dp.sel_idx(np.array(pred_probs[view]), train_env.observation_space.n))
            # torch.save(model.state_dict(),save_paths[view] +
            #            '.cotrain.epoch%d' % (step + 1))
            mu.evaluate(net, test_env, configs[view], view)

        pred_probs_np = np.array(pred_probs)
        pred_y = np.argmax(sum(pred_probs_np), axis=1)
        add_id = sum(add_ids)
        print(f"add id: {add_id}")
        train_data, untrain_data = dp.update_train_untrain(
            add_id,
            range(train_data.observation_space.n),
            range(untrain_data.observation_space.n),
            pred_y)


if __name__ == "__main__":
    # print(args.iter_step)
    # assert args.iter_step >= 1
    # dataset_dir = os.path.join(args.data_dir, args.dataset)
    # dataset = datasets.create(args.dataset, dataset_dir)
    # model_names = [args.arch1, args.arch2]
    # save_paths = [os.path.join(args.logs_dir, args.arch1),
    #               os.path.join(args.logs_dir, args.arch2)]
    # cotrain(model_names,dataset,save_paths,args.iter_step,args.train_ratio)
    e = EXP.ENV
    config1 = ConfigRL(model_name='e_sarsa_fea', env=e)
    config2 = ConfigRL(model_name='e_sarsa_fea', env=e)

    print(cotrain(
          e["map"],
          e["type"],
          [config1, config2],
          save_paths="./",
          iter_step=EXP.COTRAIN_ITERS,
          train_ratio=0.2))

    print(f"\n\nTotal steps: {EXP.COUNTER}")
    print(f"Total swarm  updates: {EXP.SWARM_UPDATE_COUNTER}\n\n")
    EXP.COUNTER = 0
    EXP.SWARM_UPDATE_COUNTER = 0