"""
IA2C and MA2C algorithms
@author: Tianshu Chu
"""
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
from agents.utils import OnPolicyBuffer, MultiAgentOnPolicyBuffer, Scheduler
from agents.policies import (LstmPolicy, FPPolicy, ConsensusPolicy, QConseNetPolicy, NCMultiAgentPolicy,
                             CommNetMultiAgentPolicy, DIALMultiAgentPolicy)
import logging
import numpy as np


class IA2C:
    """
    The basic IA2C implementation with decentralized actor and centralized critic,
    limited to neighborhood area only.
    """

    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ia2c'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def add_transition(self, ob, naction, action, reward, value, done):
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        for i in range(self.n_agent):
            self.trans_buffer[i].add_transition(ob[i], naction[i], action[i], reward, value[i], done)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        for i in range(self.n_agent):
            obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(Rends[i], dt)
            if i == 0:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef,
                                        summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef)
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

    def forward(self, obs, done, nactions=None, out_type='p'):
        out = []
        if nactions is None:
            nactions = [None] * self.n_agent
        for i in range(self.n_agent):
            cur_out = self.policy[i](obs[i], done, nactions[i], out_type)
            out.append(cur_out)
        return out

    def load(self, model_dir, global_step=None, train_mode=False):
        save_file = None
        save_step = 0
        if os.path.exists(model_dir):
            if global_step is None:
                for file in os.listdir(model_dir):
                    if file.startswith('checkpoint'):
                        tokens = file.split('.')[0].split('-')
                        if len(tokens) != 2:
                            continue
                        cur_step = int(tokens[1])
                        if cur_step > save_step:
                            save_file = file
                            save_step = cur_step
            else:
                save_file = 'checkpoint-{:d}.pt'.format(global_step)
        if save_file is not None:
            file_path = model_dir + save_file
            checkpoint = torch.load(file_path)
            logging.info('Checkpoint loaded: {}'.format(file_path))
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            if train_mode:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.policy.train()
            else:
                self.policy.eval()
            return True
        logging.error('Can not find checkpoint for {}'.format(model_dir))
        return False

    def reset(self):
        for i in range(self.n_agent):
            self.policy[i]._reset()

    def save(self, model_dir, global_step):
        file_path = model_dir + 'checkpoint-{:d}.pt'.format(global_step)
        torch.save({'global_step': global_step,
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()},
                   file_path)
        logging.info('Checkpoint saved: {}'.format(file_path))

    def _init_algo(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                   total_step, seed, use_gpu, model_config):
        # init params
        self.n_s_ls = n_s_ls
        self.n_a_ls = n_a_ls
        self.identical_agent = False
        if (max(self.n_a_ls) == min(self.n_a_ls)):
            # note for identical IA2C, n_s_ls may have variant dims
            self.identical_agent = True
            self.n_s = n_s_ls[0]
            self.n_a = n_a_ls[0]
        else:
            self.n_s = max(self.n_s_ls)
            self.n_a = max(self.n_a_ls)
        self.neighbor_mask = neighbor_mask
        self.n_agent = len(self.neighbor_mask)
        self.reward_clip = model_config.getfloat('reward_clip')
        self.reward_norm = model_config.getfloat('reward_norm')
        self.n_step = model_config.getint('batch_size')
        self.n_fc = model_config.getint('num_fc')
        self.n_lstm = model_config.getint('num_lstm')
        self.device = torch.device("cpu")
        logging.info("Use cpu for pytorch")

        # self.device = torch.device("cuda:1")
        # logging.info("Use gpu for pytorch")

        self.policy = self._init_policy()
        self.policy.to(self.device)

        # init exp buffer and lr scheduler for training
        if total_step:
            self.total_step = total_step
            self._init_train(model_config, distance_mask, coop_gamma)

    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = np.sum(self.neighbor_mask[i])
            if self.identical_agent:
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i))
            else:
                na_dim_ls = []
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j])
                local_policy = LstmPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                          n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i),
                                          na_dim_ls=na_dim_ls, identical=False)
                # local_policy.to(self.device)
            policy.append(local_policy)
        return nn.ModuleList(policy)

    def _init_scheduler(self, model_config):
        # init lr scheduler
        self.lr_init = model_config.getfloat('lr_init')
        self.lr_decay = model_config.get('lr_decay')
        self.epsilon = 0.001
        # self.r = [1.0, 0.5, 1.5, 1.5]  # slowdown
        self.r = [0.5, 0.42, 0.3, 0.3]  # catchup

        if self.lr_decay == 'constant':
            self.lr_scheduler = Scheduler(self.lr_init, decay=self.lr_decay)
        else:
            lr_min = model_config.getfloat('lr_min')
            self.lr_scheduler = Scheduler(self.lr_init, lr_min, self.total_step, decay=self.lr_decay)

    def compute_epsilon(self, gloabl_step):
        epsilon = self.a / (1 + gloabl_step ** self.b)
        epsilon_lambda = epsilon * self.c / (1 + gloabl_step ** self.d)
        return epsilon, epsilon_lambda

    def _init_train(self, model_config, distance_mask, coop_gamma):
        # init lr scheduler
        self._init_scheduler(model_config)
        # init parameters for grad computation
        self.v_coef = model_config.getfloat('value_coef')
        self.e_coef = model_config.getfloat('entropy_coef')
        self.max_grad_norm = model_config.getfloat('max_grad_norm')
        # init optimizer
        alpha = model_config.getfloat('rmsp_alpha')
        epsilon = model_config.getfloat('rmsp_epsilon')
        self.optimizer = optim.RMSprop(self.policy.parameters(), self.lr_init,
                                       eps=epsilon, alpha=alpha)
        # init transition buffer
        gamma = model_config.getfloat('gamma')
        self._init_trans_buffer(gamma, distance_mask, coop_gamma)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = []
        for i in range(self.n_agent):
            # init replay buffer
            self.trans_buffer.append(OnPolicyBuffer(gamma, coop_gamma, distance_mask[i]))

    def _update_lr(self):
        # TODO: refactor this using optim.lr_scheduler
        cur_lr = self.lr_scheduler.get(self.n_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr


class QConseNet(IA2C):
    """Fully decentralized A2C with consensus update + quantization-based privacy-preserving schemes"""

    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ia2c_cu'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = np.sum(self.neighbor_mask[i])
            if self.identical_agent:  # True
                local_policy = QConseNetPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                               n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i))
            else:
                na_dim_ls = []
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j])
                local_policy = QConseNetPolicy(self.n_s_ls[i], self.n_a_ls[i], n_n, self.n_step,
                                               n_fc=self.n_fc, n_lstm=self.n_lstm, name='{:d}'.format(i),
                                               na_dim_ls=na_dim_ls, identical=False)
                # local_policy.to(self.device)
            policy.append(local_policy)
        return nn.ModuleList(policy)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        # complete the global updates
        self.optimizer.zero_grad()
        for i in range(self.n_agent):
            obs, nas, acts, dones, Rs, Advs = self.trans_buffer[i].sample_transition(Rends[i], dt)
            if i == 0:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef,
                                        summary_writer=summary_writer, global_step=global_step)
            else:
                self.policy[i].backward(obs, nas, acts, dones, Rs, Advs,
                                        self.e_coef, self.v_coef)

        """Version 3"""
        self.policy_temp = [[] for _ in range(self.n_agent)]
        # make deepcopy of agents' parameters
        for i in range(self.n_agent):
            for wt in self.policy[i].lstm_layer_c.parameters():
                self.policy_temp[i].append(copy.deepcopy(wt.detach()))
        """"""

        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

        # complete the consensus updates
        with torch.no_grad():

            """
            Original version
            """
            # for i in range(self.n_agent):
            #     # aggregate neighbors' weights
            #     wts = []
            #     for wt in self.policy[i].lstm_layer_c.parameters():
            #         wts.append(wt.detach())
            #
            #     neighbors = list(np.where(self.neighbor_mask[i] == 1)[0])
            #     for j in neighbors:
            #         for k, wt in enumerate(self.policy[j].lstm_layer_c.parameters()):
            #             wts[k] += wt.detach()
            #
            #     n = 1 + len(neighbors)
            #     for k in range(len(wts)):
            #         wts[k] /= n
            #
            #     # update the weights of agent i
            #     for param, wt in zip(self.policy[i].lstm_layer_c.parameters(), wts):
            #         param.copy_(wt)

            """
            MACACC update with w'_{i+1} = w_{i+1} + \sum{c_{ij} (w_{j+1} - w_{i+1})}
            """
            # for i in range(self.n_agent):
            #     # aggregate neighbors' weights
            #     wts = []
            #     for wt in self.policy[i].lstm_layer_c.parameters():
            #         wts.append(wt.detach())
            #
            #     neighbors = list(np.where(self.neighbor_mask[i] == 1)[0])
            #     for j in neighbors:
            #         for k, wt in enumerate(self.policy[j].lstm_layer_c.parameters()):
            #             wts[k] += self.epsilon * (wt.detach() - wts[k])
            #
            #     # update the weights of agent i
            #     for param, wt in zip(self.policy[i].lstm_layer_c.parameters(), wts):
            #         param.copy_(wt)

            """
            MACACC (n) update with w'_{i+1} = w_{i+1} + \sum{c_{ij} (Q(w_{j}) - Q(w_{i}))}
            Uncomment codes in backward(), and self.r
            """
            # quantize agent's state
            wt_q = [[] for _ in range(self.n_agent)]
            for i in range(self.n_agent):
                for k, wt in enumerate(self.policy_temp[i]):
                    # determine the largest abs parameter --> only need to do once
                    # if torch.max(torch.abs(wt)).item() > self.r[k]:
                    #     self.r[k] = torch.max(torch.abs(wt)).item()
                    bi = self._quantization(wt, k, n=1)
                    wt_q[i].append(self.r[k] * torch.sign(wt) * bi)

            for i in range(self.n_agent):
                wts = []
                for wt in self.policy[i].lstm_layer_c.parameters():
                    wts.append(wt.detach())

                # aggregate neighbors' weights
                neighbors = list(np.where(self.neighbor_mask[i] == 1)[0])
                for j in neighbors:
                    for k in range(len(wt_q[j])):
                        wts[k] += self.epsilon * (wt_q[j][k] - wt_q[i][k])

                # update the weights of agent i
                for param, wt in zip(self.policy[i].lstm_layer_c.parameters(), wts):
                    param.copy_(wt)

    def _quantization(self, wt, k, n):
        """
        Args:
            wt: parameters
            k: # layer of LSTN
            n: resolution of quantization

        Returns:

        """
        if n == 1:
            return torch.gt(torch.rand(wt.size()), torch.abs(wt) / self.r[k]).float()
        else:
            wt_normalied = torch.abs(wt) / self.r[k]
            wt_spread = wt_normalied * n
            m = torch.floor(wt_spread).type(torch.int32)
            bi = torch.gt(torch.rand(wt.size()), torch.abs(wt) * n / self.r[k] - m).float() * 1 / n + m / n
            # print(torch.max(bi))
            return bi


class IA2C_FP(IA2C):
    """
    In fingerprint IA2C, neighborhood policies (fingerprints) are also included.
    """

    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ia2c_fp'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        policy = []
        for i in range(self.n_agent):
            n_n = np.sum(self.neighbor_mask[i])
            # neighborhood policies are included in local state
            if self.identical_agent:
                n_s1 = int(self.n_s_ls[i] + self.n_a * n_n)
                policy.append(FPPolicy(n_s1, self.n_a, int(n_n), self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='{:d}'.format(i)))
            else:
                na_dim_ls = []
                for j in np.where(self.neighbor_mask[i] == 1)[0]:
                    na_dim_ls.append(self.n_a_ls[j])
                n_s1 = int(self.n_s_ls[i] + sum(na_dim_ls))
                policy.append(FPPolicy(n_s1, self.n_a_ls[i], int(n_n), self.n_step, n_fc=self.n_fc,
                                       n_lstm=self.n_lstm, name='{:d}'.format(i),
                                       na_dim_ls=na_dim_ls, identical=False))
        return nn.ModuleList(policy)


class MA2C_NC(IA2C):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ma2c_nc'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def add_transition(self, ob, p, action, reward, value, done):
        if self.reward_norm > 0:
            reward = reward / self.reward_norm
        if self.reward_clip > 0:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        if self.identical_agent:
            self.trans_buffer.add_transition(np.array(ob), np.array(p), action,
                                             reward, value, done)
        else:
            pad_ob, pad_p = self._convert_hetero_states(ob, p)
            self.trans_buffer.add_transition(pad_ob, pad_p, action,
                                             reward, value, done)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        self.optimizer.zero_grad()
        obs, ps, acts, dones, Rs, Advs = self.trans_buffer.sample_transition(Rends, dt)
        self.policy.backward(obs, ps, acts, dones, Rs, Advs, self.e_coef, self.v_coef,
                             summary_writer=summary_writer, global_step=global_step)
        if self.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.lr_decay != 'constant':
            self._update_lr()

    def forward(self, obs, done, ps, actions=None, out_type='p'):
        if self.identical_agent:
            return self.policy.forward(np.array(obs), done, np.array(ps),
                                       actions, out_type)
        else:
            pad_ob, pad_p = self._convert_hetero_states(obs, ps)
            return self.policy.forward(pad_ob, done, pad_p,
                                       actions, out_type)

    def reset(self):
        self.policy._reset()

    def _convert_hetero_states(self, ob, p):
        pad_ob = np.zeros((self.n_agent, self.n_s))
        pad_p = np.zeros((self.n_agent, self.n_a))
        for i in range(self.n_agent):
            pad_ob[i, :len(ob[i])] = ob[i]
            pad_p[i, :len(p[i])] = p[i]
        return pad_ob, pad_p

    def _init_policy(self):
        if self.identical_agent:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return NCMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                      self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                      n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)

    def _init_trans_buffer(self, gamma, distance_mask, coop_gamma):
        self.trans_buffer = MultiAgentOnPolicyBuffer(gamma, coop_gamma, distance_mask)


class IA2C_CU(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ma2c_cu'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        # return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
        #                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        if self.identical_agent:
            return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return ConsensusPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                   self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                   n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)

    def backward(self, Rends, dt, summary_writer=None, global_step=None):
        super(IA2C_CU, self).backward(Rends, dt, summary_writer, global_step)
        self.policy.consensus_update()


class MA2C_CNET(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ma2c_ic3'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return CommNetMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                           self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return CommNetMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                           self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                           n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)


class MA2C_DIAL(MA2C_NC):
    def __init__(self, n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                 total_step, model_config, seed=0, use_gpu=False):
        self.name = 'ma2c_dial'
        self._init_algo(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                        total_step, seed, use_gpu, model_config)

    def _init_policy(self):
        if self.identical_agent:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm)
        else:
            return DIALMultiAgentPolicy(self.n_s, self.n_a, self.n_agent, self.n_step,
                                        self.neighbor_mask, n_fc=self.n_fc, n_h=self.n_lstm,
                                        n_s_ls=self.n_s_ls, n_a_ls=self.n_a_ls, identical=False)
