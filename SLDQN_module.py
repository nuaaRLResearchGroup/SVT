import math
import random
import time
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import torchvision.transforms as T
import copy
# from sklearn.neighbors import KDTree
# import scipy.io as sio
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

use_cuda = torch.cuda.is_available()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print('run on GPU, num of GPU is', torch.cuda.device_count())
else:
    device = torch.device("cpu")
    print('run on CPU')

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

###########################

random.seed(time.time())

Transition = namedtuple('Transition',
                        ('state', 'next_state', 'action', 'reward'))
# 'state'--numpy, 'next_state'--numpy, 'action'--int, 'reward'--float

Transition_chain = namedtuple('Transition_chain',
                              ('net_input', 'next_net_input', 'action', 'reward'))



class ReplayMemory(object):
    def __init__(self, capacity, window, input_length):

        self.__window = window
        self.__input_length = input_length

        self.__capacity = capacity
        self.__memory = []
        self.__memory_chain = []

    def __len__(self):
        return len(self.__memory_chain)

    def reset(self):
        self.__memory = []
        self.__memory_chain = []

    def get_net_input(self, state):
        memory_length = len(self.__memory)
        if (memory_length <= self.__window):
            return None

        else:
            net_input = []
            for i in range(memory_length - self.__window, memory_length):
                net_input += self.__memory[i].state.tolist()  # state
                net_input.append(self.__memory[i].action)  # action
            net_input += state.tolist()
            net_input = np.array(net_input).reshape(-1)
            return net_input

    def push(self, state, next_state, action, R):
        # 输出net_input和next_net_input

        net_input = self.get_net_input(state)
        self.__memory.append(Transition(state, next_state, action, R))
        if (len(self.__memory) > self.__capacity):
            self.__memory.pop(0)
        next_net_input = self.get_net_input(next_state)
        if ((None is not net_input) and (None is not next_net_input)):
            self.__memory_chain.append(Transition_chain(net_input, next_net_input, action, R))
            if (len(self.__memory_chain) > self.__capacity):
                self.__memory_chain.pop(0)
        return net_input, next_net_input

    def sample(self, batch_size):
        return random.sample(self.__memory_chain, batch_size)

class DQN:

    def __init__(self, input_length, Num_action, memory_capacity, window, GAMMA=0.5, learning_begin=50, beta=0.7,
                 safe_mode=True):

        self.Num_action = Num_action
        self.memory = ReplayMemory(memory_capacity, window, input_length)
        self.GAMMA = GAMMA
        self.learning_begin = learning_begin
        self.beta = beta
        self.safe_mode = safe_mode
        if self.safe_mode:
            print('running on safe mode!')

        if use_cuda:
            self.CNN_Q_0 = CNN_Q((input_length + 1) * window + input_length, Num_action)
            self.CNN_Q_model = torch.nn.DataParallel(self.CNN_Q_0.cuda())
        else:
            self.CNN_Q_model = CNN_Q((input_length + 1) * window + input_length, Num_action)

        self.optimizer_Q = optim.SGD(self.CNN_Q_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-7)

    def reset(self):
        self.memory.reset()
        # self.model.reset()
        # self.steps_done = 0

        if use_cuda:
            self.CNN_Q_0.reset()
            self.CNN_Q_model = torch.nn.DataParallel(self.CNN_Q_0.cuda())  ## 每个episode结束后在GPU上重置网络模型
        else:
            self.CNN_Q_model.reset()

    def select_action(self, state, step):

        state = state.reshape(-1)
        net_input = self.memory.get_net_input(state)

        if (step <= self.learning_begin):
            return np.random.choice(range(self.Num_action), 1, replace=False)[0]

        if (None is not net_input):

            Q_value = self.CNN_Q_model(torch.from_numpy(net_input.reshape(1, -1)).float())

            temp = (math.e ** Q_value).reshape(-1)

            prob = temp / sum(temp)
            prob = prob.cpu().detach().numpy()

            # check NaN
            if np.isnan(prob.all()):
                print('NaN detected!')
                idx = random.choice(range(self.Num_action))
            else:
                idx = np.random.choice(range(self.Num_action), 1, replace=False, p=prob)[0]


        else:
            idx = random.choice(range(self.Num_action))

        return idx

    def update_memory(self, state, next_state, action, R):

        # 检查输入类型
        state = state.reshape(-1)
        next_state = next_state.reshape(-1)
        action = int(action)
        # R=int(R)

        self.memory.push(state, next_state, action, R)  # Transition = namedtuple('Transition',('state', 'next_state', 'action', 'reward'))

    def optimize_net_para(self, step, gamma_start, gamma_end, anneal_step, learning_begin, BATCH_SIZE=32):
        if len(self.memory) < BATCH_SIZE:
            return

        experience = self.memory.sample(BATCH_SIZE)
        batch = Transition_chain(*zip(*experience))

        def _cat_to_tensor(data, dev, dtype):
            return torch.cat([torch.tensor(np.array(data), dtype=dtype).to(dev)])

        # print(batch)
        # print(batch.next_net_input)
        next_states = _cat_to_tensor(batch.next_net_input, device, torch.float32)
        state_batch = _cat_to_tensor(batch.net_input, device, torch.float32)
        action_batch = _cat_to_tensor(batch.action, device, torch.long).view(-1, 1)
        reward_batch = _cat_to_tensor(batch.reward, device, torch.float32)
        # E_batch = _cat_to_tensor(batch.E, device, torch.float32)

        ########################## update #############################
        gamma_temp = gamma_start - (step - learning_begin) * (gamma_start - gamma_end) / anneal_step
        self.GAMMA = max(gamma_end, min(gamma_start, gamma_temp))

        state_action_Q_values = self.CNN_Q_model(state_batch).gather(1, action_batch)
        if not self.safe_mode:
            next_state_action_Q_values = self.CNN_Q_model(next_states).max(1)[0].detach()
            # next_state_action_Q_values = (self.CNN_Q_model(next_states)*self.CNN_E_model(next_states)).max(1)[0].detach()
        else:
            # next_state_action_Q_values = \
            #     (self.CNN_Q_model(next_states) * self.beta ** (1 / self.CNN_E_model(next_states))).max(1)[0].detach()
            next_state_action_Q_values = \
                (self.CNN_Q_model(next_states) * self.beta ** (1 / self.CNN_E_model(next_states))).max(1)[0].detach()
        expected_state_action_Q_values = (next_state_action_Q_values * self.GAMMA) + reward_batch
        lossQ = F.smooth_l1_loss(state_action_Q_values, expected_state_action_Q_values.unsqueeze(1))  # 原版loss

        # print("loss",loss)
        # # Optimize the model
        self.optimizer_Q.zero_grad()
        lossQ.backward()
        for param in self.CNN_Q_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_Q.step()


    def optimize_model(self, state, next_state, action, R, step, gamma_start=0.8, gamma_end=0.3, anneal_step=2000,
                       learning_begin=50, BATCH_SIZE=32):
        self.update_memory(state, next_state, action, R)
        self.optimize_net_para(step, gamma_start=gamma_start, gamma_end=gamma_end, anneal_step=anneal_step,
                               learning_begin=learning_begin, BATCH_SIZE=BATCH_SIZE)

    def hotbooting(self, times, HotbootingMemory, BATCH_SIZE=32):
        print('hotbooting...')
        self.memory = copy.deepcopy(HotbootingMemory)
        for _ in range(times):
            self.optimize_net_para(BATCH_SIZE)
