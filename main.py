import torch as th
import os

torch_seed = 0  # 0, 66, 2023
th.manual_seed(torch_seed)
th.cuda.manual_seed_all(torch_seed)  # if you are using multi-GPU.
th.backends.cudnn.benchmark = False
th.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(torch_seed)

import argparse
import copy
import configparser
import logging
from torch.utils.tensorboard.writer import SummaryWriter
from envs.cacc_env import CACCEnv
from agents.models import IA2C, IA2C_FP, MA2C_NC, IA2C_CU, QConseNet, MA2C_CNET, MA2C_DIAL
from trainer import (Counter, Trainer, Tester, Evaluator,
                     check_dir, copy_file, find_file,
                     init_dir, init_log, init_test_flag)


def parse_args():
    default_base_dir = r'C:/Users/chend/Downloads/MACACC/results/ia2c_catchup_v0'
    default_config_dir = './config/config_ia2c_catchup.ini'  # change self.r
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--option', type=str, required=False,
                        default="train", help="train or evaluate")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--evaluation-seeds', type=str, required=False,
                        default=','.join([str(i) for i in range(2000, 2500, 10)]),
                        help="random seeds for evaluation, split by ,")
    parser.add_argument('--demo', action='store_true', help="shows SUMO gui")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0):
    return CACCEnv(config)


def init_agent(env, config, total_step, seed):
    if env.agent == 'ia2c':
        return IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config, seed=seed)
    if env.agent == 'ia2c_qconsenet':
        # this is actually MACACC
        return QConseNet(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    elif env.agent == 'ia2c_fp':
        return IA2C_FP(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_nc':
        return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_cnet':
        # this is actually CommNet
        return MA2C_CNET(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    elif env.agent == 'ma2c_cu':
        # this is actually ConsensusNet
        return IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
    elif env.agent == 'ma2c_dial':
        return MA2C_DIAL(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    else:
        return None


def train(args):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    copy_file('main.py', dirs['data'])
    copy_file('trainer.py', dirs['data'])
    copy_file('agents/models.py', dirs['data'])
    copy_file('agents/policies.py', dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = init_env(config['ENV_CONFIG'])
    logging.info('Training: a dim %r, agent dim: %d' % (env.n_a_ls, env.n_agent))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    seed = config.getint('ENV_CONFIG', 'seed')
    model = init_agent(env, config['MODEL_CONFIG'], total_step, seed)
    model.load(dirs['model'], train_mode=True)

    # # calculate agents' distance in terms of parameter space
    # critic_w = [[] for _ in range(model.n_agent)]
    # for i in range(model.n_agent):
    #     for wt in model.policy[i].lstm_layer_c.parameters():
    #         critic_w[i].append(copy.deepcopy(wt.detach().numpy()))
    #
    # distance = np.zeros((8, 8))
    # for i in range(model.n_agent):
    #     for j in range(i + 1, model.n_agent):
    #         normalized_i = critic_w[i][1] / np.linalg.norm(critic_w[i][1], 'fro')
    #         normalized_j = critic_w[j][1] / np.linalg.norm(critic_w[j][1], 'fro')
    #         # normalized_i = critic_w[i][1]
    #         # normalized_j = critic_w[j][1]
    #         distance[i, j] = np.linalg.norm(normalized_i - normalized_j, 'fro')
    #         distance[j, i] = distance[i, j]
    #
    # mean_distances = np.mean(distance[np.triu_indices_from(distance, 1)])
    # variances = np.var(distance[np.triu_indices_from(distance, 1)])
    # print(mean_distances, variances)

    # disable multi-threading for safe SUMO implementation
    summary_writer = SummaryWriter(dirs['log'], flush_secs=10000)
    trainer = Trainer(env, model, global_counter, summary_writer, output_path=dirs['data'])
    trainer.run()

    # save model
    final_step = global_counter.cur_step
    model.save(dirs['model'], final_step)
    summary_writer.close()


def evaluate_fn(agent_dir, output_dir, seeds, port, demo):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file 
    config_dir = find_file(agent_dir + '/data/')
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = init_env(config['ENV_CONFIG'], port=port)
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config['MODEL_CONFIG'], 0, 0)
    if model is None:
        return
    model_dir = agent_dir + '/model/'
    if not model.load(model_dir):
        return

    # critic_w, actual_w = [[] for _ in range(model.n_agent)], [[] for _ in range(model.n_agent)]
    # for i in range(model.n_agent):
    #     for wt in model.policy[i].lstm_layer_c.parameters():
    #         wt_array = copy.deepcopy(wt.detach().numpy().ravel())
    #         wt_normalized =(wt_array - wt_array.mean(axis=0)) / wt_array.std(axis=0)
    #         critic_w[i].append(wt_normalized)
    #         actual_w[i].append(copy.deepcopy(wt.detach().numpy().ravel()))
    #
    # distance = np.zeros((model.n_agent, model.n_agent))
    # for i in range(model.n_agent):
    #     for j in range(i+1, model.n_agent):
    #         # normalized_i = critic_w[i][1]
    #         # normalized_j = critic_w[j][1]
    #         normalized_i = actual_w[i][1]
    #         normalized_j = actual_w[j][1]
    #         distance[i, j] = np.linalg.norm(normalized_i - normalized_j)
    #         distance[j, i] = distance[i, j]
    #
    # mean_distances = np.mean(distance[np.triu_indices_from(distance, 1)])
    # variances = np.var(distance[np.triu_indices_from(distance, 1)])
    # print(mean_distances, variances)

    # # version 1
    # for n_fig in range(100):
    #     # Plot bars
    #     indices = random.sample(range(0, len(critic_w[0][1])), 4)
    #     array_value, actual_data = [[] for _ in range(len(indices))], [[] for _ in range(len(indices))]
    #     for i in range(len(indices)):
    #         for j in range(model.n_agent):
    #             array_value[i].append(critic_w[j][1][indices[i]])
    #             actual_data[i].append(actual_w[j][1][indices[i]])
    #     # plot_bar(indices, array_value, array_value)
    #     plot_bar(indices, actual_data, actual_data, n_fig)
    #     print(n_fig, indices)

    # version 2
    # for n_fig in range(100):
    #     # Plot bars with error bars
    #     indices = random.sample(range(0, len(critic_w[0][1])), 10)
    #     array_value, actual_data = [[] for _ in range(len(indices))], [[] for _ in range(len(indices))]
    #     for i in range(len(indices)):
    #         for j in range(model.n_agent):
    #             array_value[i].append(critic_w[j][1][indices[i]])
    #             actual_data[i].append(actual_w[j][1][indices[i]])
    #
    #     plot_bar_v1(actual_data, n_fig)
    #     print(n_fig, indices)

    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir, gui=demo)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    if not args.demo:
        dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
        init_log(dirs['eva_log'])
        output_dir = dirs['eva_data']
    else:
        output_dir = None
    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, output_dir, seeds, 1, args.demo)


if __name__ == '__main__':
    args = parse_args()
    if args.option == 'train':
        train(args)
    else:
        evaluate(args)

    # train(args)
