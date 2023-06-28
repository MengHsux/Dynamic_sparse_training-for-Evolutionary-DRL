import numpy as np
import os
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as FF
import sparse_utils as sp
from utils import Hot_Plug

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
Device = torch.device("cuda" if USE_CUDA else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RLNN(nn.Module):
    def __init__(self, args):
        super(RLNN, self).__init__()
        self.args = args
        self.nonlinearity_actor = args.nonlinearity_actor
        self.nonlinearity_critic = args.nonlinearity_critic

    def set_params(self, params):
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())
            param.data.copy_(torch.from_numpy(params[cpt:cpt + tmp]).view(param.size()).to(Device))
            cpt += tmp

    def get_params(self):
        return copy.deepcopy(np.hstack([v.cpu().data.numpy().flatten() for v in self.parameters()]))

    def get_grads(self):
        return copy.deepcopy(np.hstack([v.grad.cpu().data.numpy().flatten() for v in self.parameters()]))

    def get_size(self):
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        if filename is None: return
        params = np.load('{}/{}.npy'.format(filename, net_name))
        self.set_params(params)

    def save_model(self, output, net_name):
        params = self.get_params()
        np.save('{}/{}.npy'.format(output, net_name), params)


class Actor(RLNN):
    def __init__(self, args, state_dim, action_dim, max_action,noHidNeurons,epsilonHid1,epsilonHid2):
        super(Actor, self).__init__(args)

        self.l1 = nn.Linear(state_dim, noHidNeurons)
        [self.noPar1, self.mask1] = sp.initializeEpsilonWeightsMask("actor first layer", epsilonHid1, state_dim,
                                                                    noHidNeurons)
        self.torchMask1 = torch.from_numpy(self.mask1).float().to(device)
        self.l1.weight.data.mul_(torch.from_numpy(self.mask1).float())

        self.l2 = nn.Linear(noHidNeurons, noHidNeurons)
        [self.noPar2, self.mask2] = sp.initializeEpsilonWeightsMask("actor second layer", epsilonHid2, noHidNeurons,
                                                                    noHidNeurons)
        self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
        self.l2.weight.data.mul_(torch.from_numpy(self.mask2).float())

        self.l3 = nn.Linear(noHidNeurons, action_dim)

        self.max_action = max_action
        self.to(Device)

    def forward(self, state):
        # Relu was used in original TD3
        if self.nonlinearity_actor == "relu":
            a = FF.relu(self.l1(state))
            a1 = FF.relu(self.l2(a))
            a = self.max_action * torch.tanh(self.l3(a1))
        # Elu was used in CERL
        elif self.nonlinearity_actor == "elu":
            a = FF.elu(self.l1(state))
            a1 = FF.elu(self.l2(a))
            a = self.max_action * torch.tanh(self.l3(a1))
        # Tanh was used in ERL, CEM-RL, and PDERL, this is basic setting
        else:
            a = torch.tanh(self.l1(state))
            a1 = torch.tanh(self.l2(a))
            a = self.max_action * torch.tanh(self.l3(a1))

        return a, a1

    def select_action(self, state):
        # Input state is np.array(), therefore, convert np.array() to tensor
        state = FloatTensor(state).unsqueeze(0)
        # Get action from current policy
        action, _ = self.forward(state)
        # Must be env.step(np.array* or lis*), therefore, convert tensor to np.array()
        return action.cpu().data.numpy().flatten()


class Critic(RLNN):
    def __init__(self, args, state_dim, action_dim,noHidNeurons,epsilonHid1,epsilonHid2):
        super(Critic, self).__init__(args)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, noHidNeurons)
        [self.noPar1, self.mask1] = sp.initializeEpsilonWeightsMask("critic Q1 first layer", epsilonHid1,
                                                                    state_dim + action_dim, noHidNeurons)
        self.torchMask1 = torch.from_numpy(self.mask1).float().to(device)
        self.l1.weight.data.mul_(torch.from_numpy(self.mask1).float())

        self.l2 = nn.Linear(noHidNeurons, noHidNeurons)
        [self.noPar2, self.mask2] = sp.initializeEpsilonWeightsMask("critic Q1 second layer", epsilonHid2, noHidNeurons,
                                                                    noHidNeurons)
        self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
        self.l2.weight.data.mul_(torch.from_numpy(self.mask2).float())

        self.l3 = nn.Linear(noHidNeurons, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, noHidNeurons)
        [self.noPar4, self.mask4] = sp.initializeEpsilonWeightsMask("critic Q2 first layer", epsilonHid1,
                                                                    state_dim + action_dim, noHidNeurons)
        self.torchMask4 = torch.from_numpy(self.mask4).float().to(device)
        self.l4.weight.data.mul_(torch.from_numpy(self.mask4).float())

        self.l5 = nn.Linear(noHidNeurons, noHidNeurons)
        [self.noPar5, self.mask5] = sp.initializeEpsilonWeightsMask("critic Q2 second layer", epsilonHid2, noHidNeurons,
                                                                    noHidNeurons)
        self.torchMask5 = torch.from_numpy(self.mask5).float().to(device)
        self.l5.weight.data.mul_(torch.from_numpy(self.mask5).float())

        self.l6 = nn.Linear(noHidNeurons, 1)

        self.to(Device)

    def forward(self, state, action):
        # The input of critic-Q is [state, action]
        sa = torch.cat([state, action], 1)

        # Relu was used in original TD3
        if self.nonlinearity_critic == "relu":
            q1 = FF.relu(self.l1(sa))
            q1 = FF.relu(self.l2(q1))
            q2 = FF.relu(self.l4(sa))
            q2 = FF.relu(self.l5(q2))
        # Elu was used in ERL, CERL, and PDERL
        elif self.nonlinearity_critic == "elu":
            q1 = FF.elu(self.l1(sa))
            q1 = FF.elu(self.l2(q1))
            q2 = FF.elu(self.l4(sa))
            q2 = FF.elu(self.l5(q2))
        # Leaky_relu was used in CEM-RL, this is basic setting
        else:
            q1 = FF.leaky_relu(self.l1(sa))
            q1 = FF.leaky_relu(self.l2(q1))
            q2 = FF.leaky_relu(self.l4(sa))
            q2 = FF.leaky_relu(self.l5(q2))
        q1 = self.l3(q1)
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        if self.nonlinearity_critic == "relu":
            q1 = FF.relu(self.l1(sa))
            q1 = FF.relu(self.l2(q1))
        elif self.nonlinearity_critic == "elu":
            q1 = FF.elu(self.l1(sa))
            q1 = FF.elu(self.l2(q1))
        else:
            q1 = FF.leaky_relu(self.l1(sa))
            q1 = FF.leaky_relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class Critic_Network(RLNN):
    def __init__(self, args, hidden_dim,noHidNeurons,epsilonHid1,epsilonHid2):
        super(Critic_Network, self).__init__(args)

        self.l1 = nn.Linear(hidden_dim,100)
        [self.noPar1, self.mask1] = sp.initializeEpsilonWeightsMask("critic network Q first layer", epsilonHid1,
                                                                    hidden_dim, 100)
        self.torchMask1 = torch.from_numpy(self.mask1).float().to(device)
        self.l1.weight.data.mul_(torch.from_numpy(self.mask1).float())

        self.l2 = nn.Linear(100,100)
        [self.noPar2, self.mask2] = sp.initializeEpsilonWeightsMask("critic network Q second layer", epsilonHid2, 100,
                                                                    100)
        self.torchMask2 = torch.from_numpy(self.mask2).float().to(device)
        self.l2.weight.data.mul_(torch.from_numpy(self.mask2).float())

        self.l3 = nn.Linear(100,1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = nn.functional.softplus(self.l3(x))
        return torch.mean(x)


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action, args):
        # Parameters about the neural net structure of critic and actor
        self.args = args
        self.max_action = max_action

        # Training batch size
        self.batch_size = args.batch_size
        self.discount = args.discount
        self.tau = args.tau

        # Action noise is added in the action of target Q
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip

        # Parameters for Asynchronous update frequency
        self.total_iterC = 0
        self.total_iterA = 0
        self.policy_freq = args.policy_freq

        # Parameters for sparse model
        self.noHidNeurons = 256
        self.epsilonHid1 = 40
        self.epsilonHid2 = 64
        self.setZeta = 0.05
        self.ascTopologyChangePeriod = 1000
        self.earlyStopTopologyChangeIteration = 1e8 #kind of never
        self.lastTopologyChangeCritic = False
        self.lastTopologyChangeActor = False
        self.ascStatsActor = []
        self.ascStatsCritic = []
        self.ascStatsCritic1 = [] # for critic network

        # Guided Beta
        self.guided_beta = args.guided_beta

        # Define critics and actors
        self.critic = Critic(args, state_dim, action_dim,self.noHidNeurons,self.epsilonHid1,self.epsilonHid2)
        self.actor = Actor(args, state_dim, action_dim, max_action,self.noHidNeurons,self.epsilonHid1,self.epsilonHid2)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        # meta critic
        self.feature_critic = Critic_Network(args, self.noHidNeurons + state_dim + action_dim, self.noHidNeurons,self.epsilonHid1,self.epsilonHid2).to(device)
        self.omega_optim = torch.optim.Adam(self.feature_critic.parameters(), lr=args.aux_lr,
                                            weight_decay=args.weight_decay)
        feature_net = nn.Sequential(*list(self.actor.children())[:-2])
        self.hotplug = Hot_Plug(feature_net)
        self.lr_actor = args.actor_lr
        self.loss_store = []

        # Define optimizer in which Adam is used
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.l2_rate)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.actor_lr, weight_decay=args.l2_rate)

    def select_action(self, state):
        # Call the select_action function of actor
        return self.actor.select_action(state)

    def train(self, replay_buffer):
        self.total_iterC += 1

        # Sample mini-batch from replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)

        # Sample replay buffer for meta test
        x_val, _, _, _, _ = replay_buffer.sample(self.batch_size)
        state_val = torch.FloatTensor(x_val).to(device)

        # Define target_Q used to estimate critic loss (=TD error)
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_a, _ = self.actor_target(next_state)
            next_action = (next_a + noise).clamp(-self.max_action, self.max_action)

            # Calculate the target_Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current_Q value
        current_Q1, current_Q2 = self.critic(state, action)

        # Calculate critic loss (=difference between target_Q and current_Q)
        critic_loss = FF.mse_loss(current_Q1, target_Q) + FF.mse_loss(current_Q2, target_Q)

        # Optimize the critic parameters
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optimizer.step()

        # Adapt the sparse connectivity
        if (self.lastTopologyChangeCritic == False):
            if (self.total_iterC % self.ascTopologyChangePeriod == 2):
                if (self.total_iterC > self.earlyStopTopologyChangeIteration):
                    self.lastTopologyChangeCritic = True
                [self.critic.mask1, ascStats1] = sp.changeConnectivitySET(self.critic.l1.weight.data.cpu().numpy(),
                                                                          self.critic.noPar1, self.critic.mask1,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask1 = torch.from_numpy(self.critic.mask1).float().to(device)
                [self.critic.mask2, ascStats2] = sp.changeConnectivitySET(self.critic.l2.weight.data.cpu().numpy(),
                                                                          self.critic.noPar2, self.critic.mask2,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask2 = torch.from_numpy(self.critic.mask2).float().to(device)
                [self.critic.mask4, ascStats4] = sp.changeConnectivitySET(self.critic.l4.weight.data.cpu().numpy(),
                                                                          self.critic.noPar4, self.critic.mask4,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask4 = torch.from_numpy(self.critic.mask4).float().to(device)
                [self.critic.mask5, ascStats5] = sp.changeConnectivitySET(self.critic.l5.weight.data.cpu().numpy(),
                                                                          self.critic.noPar5, self.critic.mask5,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask5 = torch.from_numpy(self.critic.mask5).float().to(device)
                self.ascStatsCritic.append([ascStats1, ascStats2, ascStats4, ascStats5])

        # Maintain the same sparse connectivity for critic
        self.critic.l1.weight.data.mul_(self.critic.torchMask1)
        self.critic.l2.weight.data.mul_(self.critic.torchMask2)
        self.critic.l4.weight.data.mul_(self.critic.torchMask4)
        self.critic.l5.weight.data.mul_(self.critic.torchMask5)

        if self.total_iterC % self.policy_freq == 0:
            self.total_iterA += 1

            # Compute actor loss
            a, feature_output = self.actor(state)
            actor_loss = -self.critic.Q1(state,a).mean()

            concat_output = torch.cat([feature_output, state, action], 1)
            loss_auxiliary = self.feature_critic(concat_output)

            # Optimize the actor parameters
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.hotplug.update(self.lr_actor)
            action_val, _ = self.actor(state_val)
            policy_loss_val = self.critic.Q1(state_val, action_val)
            policy_loss_val = -policy_loss_val.mean()
            policy_loss_val = policy_loss_val

            # Part2 of Meta-test stage
            loss_auxiliary.backward(create_graph=True)
            self.hotplug.update(self.lr_actor)
            action_val_new, _ = self.actor(state_val)
            policy_loss_val_new = self.critic.Q1(state_val, action_val_new)
            policy_loss_val_new = -policy_loss_val_new.mean()
            policy_loss_val_new = policy_loss_val_new

            utility = policy_loss_val - policy_loss_val_new
            utility = torch.tanh(utility)
            loss_meta = -utility

            # Meta optimization of auxilary network
            self.omega_optim.zero_grad()
            grad_omega = torch.autograd.grad(loss_meta, self.feature_critic.parameters())
            for gradient, variable in zip(grad_omega, self.feature_critic.parameters()):
                variable.grad.data = gradient
            self.omega_optim.step()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_optimizer.step()
            self.hotplug.restore()

            if (self.lastTopologyChangeActor == False):
                if (self.total_iterC % self.ascTopologyChangePeriod == 2):
                    if (self.total_iterC > self.earlyStopTopologyChangeIteration):
                        self.lastTopologyChangeActor = True
                    [self.actor.mask1, ascStats1] = sp.changeConnectivitySET(self.actor.l1.weight.data.cpu().numpy(),
                                                                             self.actor.noPar1, self.actor.mask1,
                                                                             self.setZeta, self.lastTopologyChangeActor,
                                                                             self.total_iterC)
                    self.actor.torchMask1 = torch.from_numpy(self.actor.mask1).float().to(device)
                    [self.actor.mask2, ascStats2] = sp.changeConnectivitySET(self.actor.l2.weight.data.cpu().numpy(),
                                                                             self.actor.noPar2, self.actor.mask2,
                                                                             self.setZeta, self.lastTopologyChangeActor,
                                                                             self.total_iterC)
                    self.actor.torchMask2 = torch.from_numpy(self.actor.mask2).float().to(device)
                    self.ascStatsActor.append([ascStats1, ascStats2])

                    [self.feature_critic.mask1, ascStats11] = sp.changeConnectivitySET(self.feature_critic.l1.weight.data.cpu().numpy(),
                                                                             self.feature_critic.noPar1, self.feature_critic.mask1,
                                                                             self.setZeta, self.lastTopologyChangeActor,
                                                                             self.total_iterC)
                    self.feature_critic.torchMask1 = torch.from_numpy(self.feature_critic.mask1).float().to(device)
                    [self.feature_critic.mask2, ascStats21] = sp.changeConnectivitySET(self.feature_critic.l2.weight.data.cpu().numpy(),
                                                                             self.feature_critic.noPar2, self.feature_critic.mask2,
                                                                             self.setZeta, self.lastTopologyChangeActor,
                                                                             self.total_iterC)
                    self.feature_critic.torchMask2 = torch.from_numpy(self.feature_critic.mask2).float().to(device)
                    self.ascStatsCritic1.append([ascStats11, ascStats21])

            # Maintain the same sparse connectivity for actor
            self.actor.l1.weight.data.mul_(self.actor.torchMask1)
            self.actor.l2.weight.data.mul_(self.actor.torchMask2)
            self.feature_critic.l1.weight.data.mul_(self.feature_critic.torchMask1)
            self.feature_critic.l2.weight.data.mul_(self.feature_critic.torchMask2)

            # Store the loss information
            tmp_loss = []
            tmp_loss.append(critic_loss.item())
            tmp_loss.append(actor_loss.item())
            tmp_loss.append(loss_auxiliary.item())
            tmp_loss.append(loss_meta.item())
            self.loss_store.append(tmp_loss)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                if len(param.shape) > 1:
                    self.update_target_networks(param, target_param, device)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                if len(param.shape) > 1:
                    self.update_target_networks(param, target_param, device)

    # Maintain sparsity in target networks
    def update_target_networks(self, param, target_param, device):
        current_density = (param != 0).sum()
        target_density = (target_param != 0).sum()  # torch.count_nonzero(target_param.data)
        difference = target_density - current_density
        # constrain the sparsity by removing the extra elements (smallest values)
        if (difference > 0):
            count_rmv = difference
            tmp = copy.deepcopy(abs(target_param.data))
            tmp[tmp == 0] = 10000000
            unraveled = self.unravel_index(torch.argsort(tmp.view(1, -1)[0]), tmp.shape)
            rmv_indicies = torch.stack(unraveled, dim=1)
            rmv_values_smaller_than = tmp[rmv_indicies[count_rmv][0], rmv_indicies[count_rmv][1]]
            target_param.data[tmp < rmv_values_smaller_than] = 0

    def unravel_index(self, index, shape):
        out = []
        for dim in reversed(shape):
            out.append(index % dim)
            index = index // dim
        return tuple(reversed(out))

    def print_sparsity(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if (len(target_param.shape) > 1):
                critic_current_sparsity = ((target_param == 0).sum().cpu().data.numpy() * 1.0 / (
                            target_param.shape[0] * target_param.shape[1]))
                print("target critic sparsity", critic_current_sparsity)

                critic_current_sparsity = (
                            (param == 0).sum().cpu().data.numpy() * 1.0 / (param.shape[0] * param.shape[1]))
                print("critic sparsity", critic_current_sparsity)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if (len(target_param.shape) > 1):
                critic_current_sparsity = ((target_param == 0).sum().cpu().data.numpy() * 1.0 / (
                            target_param.shape[0] * target_param.shape[1]))
                print("target actor sparsity", critic_current_sparsity)

                critic_current_sparsity = (
                            (param == 0).sum().cpu().data.numpy() * 1.0 / (param.shape[0] * param.shape[1]))
                print("actor sparsity", critic_current_sparsity)

    def saveAscStats(self, filename):
        np.savez(filename + "_ASC_stats.npz", ascStatsActor=self.ascStatsActor, ascStatsCritic=self.ascStatsCritic)

    def train_guided(self, replay_buffer, guided_param):
        self.total_iterC += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
        # Sample replay buffer for meta test
        x_val, _, _, _, _ = replay_buffer.sample(self.batch_size)
        state_val = torch.FloatTensor(x_val).to(device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_a, _ = self.actor_target(next_state)
            next_action = (next_a + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = FF.mse_loss(current_Q1, target_Q) + FF.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optimizer.step()

        # Adapt the sparse connectivity
        if (self.lastTopologyChangeCritic == False):
            if (self.total_iterC % self.ascTopologyChangePeriod == 2):
                if (self.total_iterC > self.earlyStopTopologyChangeIteration):
                    self.lastTopologyChangeCritic = True
                [self.critic.mask1, ascStats1] = sp.changeConnectivitySET(self.critic.l1.weight.data.cpu().numpy(),
                                                                          self.critic.noPar1, self.critic.mask1,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask1 = torch.from_numpy(self.critic.mask1).float().to(device)
                [self.critic.mask2, ascStats2] = sp.changeConnectivitySET(self.critic.l2.weight.data.cpu().numpy(),
                                                                          self.critic.noPar2, self.critic.mask2,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask2 = torch.from_numpy(self.critic.mask2).float().to(device)
                [self.critic.mask4, ascStats4] = sp.changeConnectivitySET(self.critic.l4.weight.data.cpu().numpy(),
                                                                          self.critic.noPar4, self.critic.mask4,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask4 = torch.from_numpy(self.critic.mask4).float().to(device)
                [self.critic.mask5, ascStats5] = sp.changeConnectivitySET(self.critic.l5.weight.data.cpu().numpy(),
                                                                          self.critic.noPar5, self.critic.mask5,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask5 = torch.from_numpy(self.critic.mask5).float().to(device)
                self.ascStatsCritic.append([ascStats1, ascStats2, ascStats4, ascStats5])

        # Maintain the same sparse connectivity for critic
        self.critic.l1.weight.data.mul_(self.critic.torchMask1)
        self.critic.l2.weight.data.mul_(self.critic.torchMask2)
        self.critic.l4.weight.data.mul_(self.critic.torchMask4)
        self.critic.l5.weight.data.mul_(self.critic.torchMask5)

        if self.total_iterC % self.policy_freq == 0:
            self.total_iterA += 1

            with torch.no_grad():
                guided_actor = copy.deepcopy(self.actor)
                guided_actor.set_params(guided_param)

            a11, feature_output11 = self.actor(state)
            a12, feature_output12 = guided_actor(state)
            distance = ((a11 - a12) ** 2).mean()
            actor_loss = -self.critic.Q1(state, a11).mean() + self.guided_beta * distance

            concat_output = torch.cat([feature_output11, state, action], 1)
            loss_auxiliary = self.feature_critic(concat_output)

            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.hotplug.update(self.lr_actor)
            action_val, _ = self.actor(state_val)
            policy_loss_val = self.critic.Q1(state_val, action_val)
            policy_loss_val = -policy_loss_val.mean()
            policy_loss_val = policy_loss_val

            # Part2 of Meta-test stage
            loss_auxiliary.backward(create_graph=True)
            self.hotplug.update(self.lr_actor)
            action_val_new, _ = self.actor(state_val)
            policy_loss_val_new = self.critic.Q1(state_val, action_val_new)
            policy_loss_val_new = -policy_loss_val_new.mean()
            policy_loss_val_new = policy_loss_val_new

            utility = policy_loss_val - policy_loss_val_new
            utility = torch.tanh(utility)
            loss_meta = -utility

            # Meta optimization of auxilary network
            self.omega_optim.zero_grad()
            grad_omega = torch.autograd.grad(loss_meta, self.feature_critic.parameters())
            for gradient, variable in zip(grad_omega, self.feature_critic.parameters()):
                variable.grad.data = gradient
            self.omega_optim.step()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
            self.actor_optimizer.step()
            self.hotplug.restore()

            if (self.lastTopologyChangeActor == False):
                if (self.total_iterC % self.ascTopologyChangePeriod == 2):
                    if (self.total_iterC > self.earlyStopTopologyChangeIteration):
                        self.lastTopologyChangeActor = True
                    [self.actor.mask1, ascStats1] = sp.changeConnectivitySET(self.actor.l1.weight.data.cpu().numpy(),
                                                                             self.actor.noPar1, self.actor.mask1,
                                                                             self.setZeta, self.lastTopologyChangeActor,
                                                                             self.total_iterC)
                    self.actor.torchMask1 = torch.from_numpy(self.actor.mask1).float().to(device)
                    [self.actor.mask2, ascStats2] = sp.changeConnectivitySET(self.actor.l2.weight.data.cpu().numpy(),
                                                                             self.actor.noPar2, self.actor.mask2,
                                                                             self.setZeta, self.lastTopologyChangeActor,
                                                                             self.total_iterC)
                    self.actor.torchMask2 = torch.from_numpy(self.actor.mask2).float().to(device)
                    self.ascStatsActor.append([ascStats1, ascStats2])

                    [self.feature_critic.mask1, ascStats11] = sp.changeConnectivitySET(
                        self.feature_critic.l1.weight.data.cpu().numpy(),
                        self.feature_critic.noPar1, self.feature_critic.mask1,
                        self.setZeta, self.lastTopologyChangeActor,
                        self.total_iterC)
                    self.feature_critic.torchMask1 = torch.from_numpy(self.feature_critic.mask1).float().to(device)
                    [self.feature_critic.mask2, ascStats21] = sp.changeConnectivitySET(
                        self.feature_critic.l2.weight.data.cpu().numpy(),
                        self.feature_critic.noPar2, self.feature_critic.mask2,
                        self.setZeta, self.lastTopologyChangeActor,
                        self.total_iterC)
                    self.feature_critic.torchMask2 = torch.from_numpy(self.feature_critic.mask2).float().to(device)
                    self.ascStatsCritic1.append([ascStats11, ascStats21])

            # Maintain the same sparse connectivity for actor
            self.actor.l1.weight.data.mul_(self.actor.torchMask1)
            self.actor.l2.weight.data.mul_(self.actor.torchMask2)
            self.feature_critic.l1.weight.data.mul_(self.feature_critic.torchMask1)
            self.feature_critic.l2.weight.data.mul_(self.feature_critic.torchMask2)

            # Store the loss information
            tmp_loss = []
            tmp_loss.append(critic_loss.item())
            tmp_loss.append(actor_loss.item())
            tmp_loss.append(loss_auxiliary.item())
            tmp_loss.append(loss_meta.item())
            self.loss_store.append(tmp_loss)

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                if len(param.shape) > 1:
                    self.update_target_networks(param, target_param, device)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                if len(param.shape) > 1:
                    self.update_target_networks(param, target_param, device)

    def train_critic(self, replay_buffer):
        self.total_iterC += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_a, _ = self.actor_target(next_state)
            next_action = (next_a + noise).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = FF.mse_loss(current_Q1, target_Q) + FF.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optimizer.step()

        # Adapt the sparse connectivity
        if (self.lastTopologyChangeCritic == False):
            if (self.total_iterC % self.ascTopologyChangePeriod == 2):
                if (self.total_iterC > self.earlyStopTopologyChangeIteration):
                    self.lastTopologyChangeCritic = True
                [self.critic.mask1, ascStats1] = sp.changeConnectivitySET(self.critic.l1.weight.data.cpu().numpy(),
                                                                          self.critic.noPar1, self.critic.mask1,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask1 = torch.from_numpy(self.critic.mask1).float().to(device)
                [self.critic.mask2, ascStats2] = sp.changeConnectivitySET(self.critic.l2.weight.data.cpu().numpy(),
                                                                          self.critic.noPar2, self.critic.mask2,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask2 = torch.from_numpy(self.critic.mask2).float().to(device)
                [self.critic.mask4, ascStats4] = sp.changeConnectivitySET(self.critic.l4.weight.data.cpu().numpy(),
                                                                          self.critic.noPar4, self.critic.mask4,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask4 = torch.from_numpy(self.critic.mask4).float().to(device)
                [self.critic.mask5, ascStats5] = sp.changeConnectivitySET(self.critic.l5.weight.data.cpu().numpy(),
                                                                          self.critic.noPar5, self.critic.mask5,
                                                                          self.setZeta, self.lastTopologyChangeCritic,
                                                                          self.total_iterC)
                self.critic.torchMask5 = torch.from_numpy(self.critic.mask5).float().to(device)
                self.ascStatsCritic.append([ascStats1, ascStats2, ascStats4, ascStats5])

        # Maintain the same sparse connectivity for critic
        self.critic.l1.weight.data.mul_(self.critic.torchMask1)
        self.critic.l2.weight.data.mul_(self.critic.torchMask2)
        self.critic.l4.weight.data.mul_(self.critic.torchMask4)
        self.critic.l5.weight.data.mul_(self.critic.torchMask5)

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if len(param.shape) > 1:
                self.update_target_networks(param, target_param, device)

    def train_actor(self, replay_buffer):
        self.total_iterA += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
        # Sample replay buffer for meta test
        x_val, _, _, _, _ = replay_buffer.sample(self.batch_size)
        state_val = torch.FloatTensor(x_val).to(device)

        a, feature_output = self.actor(state)
        actor_loss = -self.critic.Q1(state, a).mean()

        concat_output = torch.cat([feature_output, state, action], 1)
        loss_auxiliary = self.feature_critic(concat_output)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.hotplug.update(self.lr_actor)
        action_val, _ = self.actor(state_val)
        policy_loss_val = self.critic.Q1(state_val, action_val)
        policy_loss_val = -policy_loss_val.mean()
        policy_loss_val = policy_loss_val

        # Part2 of Meta-test stage
        loss_auxiliary.backward(create_graph=True)
        self.hotplug.update(self.lr_actor)
        action_val_new, _ = self.actor(state_val)
        policy_loss_val_new = self.critic.Q1(state_val, action_val_new)
        policy_loss_val_new = -policy_loss_val_new.mean()
        policy_loss_val_new = policy_loss_val_new

        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility

        # Meta optimization of auxilary network
        self.omega_optim.zero_grad()
        grad_omega = torch.autograd.grad(loss_meta, self.feature_critic.parameters())
        for gradient, variable in zip(grad_omega, self.feature_critic.parameters()):
            variable.grad.data = gradient
        self.omega_optim.step()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optimizer.step()
        self.hotplug.restore()

        if (self.lastTopologyChangeActor == False):
            if (self.total_iterA % self.ascTopologyChangePeriod == 2):
                if (self.total_iterA > self.earlyStopTopologyChangeIteration):
                    self.lastTopologyChangeActor = True
                [self.actor.mask1, ascStats1] = sp.changeConnectivitySET(self.actor.l1.weight.data.cpu().numpy(),
                                                                         self.actor.noPar1, self.actor.mask1,
                                                                         self.setZeta, self.lastTopologyChangeActor,
                                                                         self.total_iterA)
                self.actor.torchMask1 = torch.from_numpy(self.actor.mask1).float().to(device)
                [self.actor.mask2, ascStats2] = sp.changeConnectivitySET(self.actor.l2.weight.data.cpu().numpy(),
                                                                         self.actor.noPar2, self.actor.mask2,
                                                                         self.setZeta, self.lastTopologyChangeActor,
                                                                         self.total_iterA)
                self.actor.torchMask2 = torch.from_numpy(self.actor.mask2).float().to(device)
                self.ascStatsActor.append([ascStats1, ascStats2])

                [self.feature_critic.mask1, ascStats11] = sp.changeConnectivitySET(
                    self.feature_critic.l1.weight.data.cpu().numpy(),
                    self.feature_critic.noPar1, self.feature_critic.mask1,
                    self.setZeta, self.lastTopologyChangeActor,
                    self.total_iterC)
                self.feature_critic.torchMask1 = torch.from_numpy(self.feature_critic.mask1).float().to(device)
                [self.feature_critic.mask2, ascStats21] = sp.changeConnectivitySET(
                    self.feature_critic.l2.weight.data.cpu().numpy(),
                    self.feature_critic.noPar2, self.feature_critic.mask2,
                    self.setZeta, self.lastTopologyChangeActor,
                    self.total_iterC)
                self.feature_critic.torchMask2 = torch.from_numpy(self.feature_critic.mask2).float().to(device)
                self.ascStatsCritic1.append([ascStats11, ascStats21])

        # Maintain the same sparse connectivity for actor
        self.actor.l1.weight.data.mul_(self.actor.torchMask1)
        self.actor.l2.weight.data.mul_(self.actor.torchMask2)
        self.feature_critic.l1.weight.data.mul_(self.feature_critic.torchMask1)
        self.feature_critic.l2.weight.data.mul_(self.feature_critic.torchMask2)

        # Store the loss information
        tmp_loss = []
        tmp_loss.append(actor_loss.item())
        tmp_loss.append(loss_auxiliary.item())
        tmp_loss.append(loss_meta.item())
        self.loss_store.append(tmp_loss)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if len(param.shape) > 1:
                self.update_target_networks(param, target_param, device)

    def train_actor_guided(self, replay_buffer, guided_param):
        self.total_iterA += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch_size)
        # Sample replay buffer for meta test
        x_val, _, _, _, _ = replay_buffer.sample(self.batch_size)
        state_val = torch.FloatTensor(x_val).to(device)
        with torch.no_grad():
            guided_actor = copy.deepcopy(self.actor)
            guided_actor.set_params(guided_param)

        a11, feature_output11 = self.actor(state)
        a12, feature_output12 = guided_actor(state)
        distance = ((a11 - a12) ** 2).mean()
        actor_loss = -self.critic.Q1(state, a11).mean() + self.guided_beta * distance

        concat_output = torch.cat([feature_output11, state, action], 1)
        loss_auxiliary = self.feature_critic(concat_output)

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.hotplug.update(self.lr_actor)
        action_val, _ = self.actor(state_val)
        policy_loss_val = self.critic.Q1(state_val, action_val)
        policy_loss_val = -policy_loss_val.mean()
        policy_loss_val = policy_loss_val

        # Part2 of Meta-test stage
        loss_auxiliary.backward(create_graph=True)
        self.hotplug.update(self.lr_actor)
        action_val_new, _ = self.actor(state_val)
        policy_loss_val_new = self.critic.Q1(state_val, action_val_new)
        policy_loss_val_new = -policy_loss_val_new.mean()
        policy_loss_val_new = policy_loss_val_new

        utility = policy_loss_val - policy_loss_val_new
        utility = torch.tanh(utility)
        loss_meta = -utility

        # Meta optimization of auxilary network
        self.omega_optim.zero_grad()
        grad_omega = torch.autograd.grad(loss_meta, self.feature_critic.parameters())
        for gradient, variable in zip(grad_omega, self.feature_critic.parameters()):
            variable.grad.data = gradient
        self.omega_optim.step()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optimizer.step()
        self.hotplug.restore()

        if (self.lastTopologyChangeActor == False):
            if (self.total_iterA % self.ascTopologyChangePeriod == 2):
                if (self.total_iterA > self.earlyStopTopologyChangeIteration):
                    self.lastTopologyChangeActor = True
                [self.actor.mask1, ascStats1] = sp.changeConnectivitySET(self.actor.l1.weight.data.cpu().numpy(),
                                                                         self.actor.noPar1, self.actor.mask1,
                                                                         self.setZeta, self.lastTopologyChangeActor,
                                                                         self.total_iterA)
                self.actor.torchMask1 = torch.from_numpy(self.actor.mask1).float().to(device)
                [self.actor.mask2, ascStats2] = sp.changeConnectivitySET(self.actor.l2.weight.data.cpu().numpy(),
                                                                         self.actor.noPar2, self.actor.mask2,
                                                                         self.setZeta, self.lastTopologyChangeActor,
                                                                         self.total_iterA)
                self.actor.torchMask2 = torch.from_numpy(self.actor.mask2).float().to(device)
                self.ascStatsActor.append([ascStats1, ascStats2])

                [self.feature_critic.mask1, ascStats11] = sp.changeConnectivitySET(
                    self.feature_critic.l1.weight.data.cpu().numpy(),
                    self.feature_critic.noPar1, self.feature_critic.mask1,
                    self.setZeta, self.lastTopologyChangeActor,
                    self.total_iterC)
                self.feature_critic.torchMask1 = torch.from_numpy(self.feature_critic.mask1).float().to(device)
                [self.feature_critic.mask2, ascStats21] = sp.changeConnectivitySET(
                    self.feature_critic.l2.weight.data.cpu().numpy(),
                    self.feature_critic.noPar2, self.feature_critic.mask2,
                    self.setZeta, self.lastTopologyChangeActor,
                    self.total_iterC)
                self.feature_critic.torchMask2 = torch.from_numpy(self.feature_critic.mask2).float().to(device)
                self.ascStatsCritic1.append([ascStats11, ascStats21])

        # Maintain the same sparse connectivity for actor
        self.actor.l1.weight.data.mul_(self.actor.torchMask1)
        self.actor.l2.weight.data.mul_(self.actor.torchMask2)
        self.feature_critic.l1.weight.data.mul_(self.feature_critic.torchMask1)
        self.feature_critic.l2.weight.data.mul_(self.feature_critic.torchMask2)

        # Store the loss information
        tmp_loss = []
        tmp_loss.append(actor_loss.item())
        tmp_loss.append(loss_auxiliary.item())
        tmp_loss.append(loss_meta.item())
        self.loss_store.append(tmp_loss)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            if len(param.shape) > 1:
                self.update_target_networks(param, target_param, device)

    def save(self, filename):
        np.save(filename + "_critic.npy", self.critic.state_dict().data.cpu().numpy())
        np.save(filename + "_actor.npy", self.actor.state_dict().data.cpu().numpy())

    def load(self, filename):
        params_critic = np.laod(filename + "_critic.npy")
        self.critic.set_params(params_critic)
        self.critic_optimizer = copy.deepcopy(self.critic)
        params_actor = np.laod(filename + "_actor.npy")
        self.critic.set_params(params_actor)
        self.actor_target = copy.deepcopy(self.actor)





