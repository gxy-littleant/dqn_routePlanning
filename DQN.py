# from sqlalchemy import false
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import gym
import matplotlib.pyplot as plt
import copy
from graphy import *
import os
import warnings
warnings.filterwarnings("ignore")
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'

parser.add_argument('--test_iteration', default=10, type=int)

args = parser.parse_args()


# hyper-parameters
BATCH_SIZE = 512
LR = 0.001
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 50000
Q_NETWORK_ITERATION = 1000

# env = gym.make("CartPole-v0")
# env = env.unwrapped

# action：8个方向
NUM_ACTIONS = 8
# state：横 纵 
NUM_STATES = 2

# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0,0.1)
        self.fc2 = nn.Linear(50,100)
        self.fc2.weight.data.normal_(0,0.1)
        self.fc3 = nn.Linear(100,80)
        self.fc3.weight.data.normal_(0,0.1)
        self.fc4 = nn.Linear(80,30)
        self.fc4.weight.data.normal_(0,0.1)
        self.out = nn.Linear(30,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):

        state = torch.unsqueeze(torch.FloatTensor(np.array(state,dtype=int)), 0).to(device) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)

            # action_value = action_value + np.random.normal(0, args.exploration_noise, size = action_dim)
            # print('action_value',action_value)

            # torch.max(input, dim) 函数 dim=1：行  dim=0：列
            action = torch.max(action_value, 1)[1].data.numpy()

            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            # action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES]).to(device)
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int)).to(device)
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2]).to(device)
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:]).to(device)

        # q_eval
        # gather函数的功能可以解释为根据 index 参数（即是索引）返回数组里面对应位置的值
        # dim=1 按列
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# def reward_func(env, x, x_dot, theta, theta_dot):
#     # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.5
#     # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5

#     reward = r1 + r2
#     return reward

    def save(self):
        torch.save(self.eval_net.state_dict(), directory +'/' + 'eval_net.pth')
        torch.save(self.target_net.state_dict(), directory +'/' + 'target_net.pth')

        print("====================================")
        print("Model has been saved...")
        print("====================================")
    def save_current(self,data):

        torch.save(self.eval_net.state_dict(), directory+"/"+str(data) + '_eval_net.pth')
        torch.save(self.target_net.state_dict(), directory+"/"+str(data)  + '_target_net.pth')
        print("====================================")
        print("Current Model has been saved...")
        print("====================================")
    def save_best(self):
        torch.save(self.eval_net.state_dict(), directory +'/' + 'eval_best_net.pth')
        torch.save(self.target_net.state_dict(), directory +'/' + 'target_best_net.pth')
        print("====================================")
        print("Best Model has been saved...")
        print("====================================")

    def load(self):
        self.eval_net.load_state_dict(torch.load(directory +'/' + 'eval_net.pth'))
        self.target_net.load_state_dict(torch.load(directory +'/' + 'target_net.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")    


def main(data):
    dqn = DQN()
    myEnv = env(data)
    episodes = 120*data
    log_interval = 200
    ep_max_steps = 100000


    if args.mode == 'test':
        print("开始测试....")
        dqn.load()  
        print("模型已加载....")
        test_log = []
        episode_count = 0
        for i in range(args.test_iteration):
            state = myEnv.reset()
            ep_r = 0
            count_iteration = 0
            temp = [state]
            
            while True:
                
                action = dqn.choose_action(state)
                next_state, reward, done = myEnv.step(action,state)
                ep_r += reward
                
                # temp.append([i, ep_r, count_iteration])
                
                if done :
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, count_iteration))
                    ep_r = 0
                    break
                state = next_state
                count_iteration+=1
                temp.append([action,next_state,reward,done,episode_count,count_iteration])
                test_log.append(temp)
            episode_count+=1

        pd.DataFrame(test_log).to_csv("./test_log.csv")
            

    elif args.mode == 'train':

        tar_dir = 'model_save'
        if len(os.listdir(tar_dir)) != 0:  # 目标文件夹内容为空的情况下
            dqn.load()
            print("模型已加载....")

        print("Collecting Experience....")
        # reward_list = []

        log_text = []
        episode_count = 0
        done_step=np.inf
        for i in range(episodes):
            state = myEnv.reset()
            ep_reward = 0
            print('state',state)

            count_iteration = 0
            step_temp=0
            for step_ in range(ep_max_steps):
                if step_%1000==0:
                    print('当前已经跑了 %d 步' %step_)
                temp = [state]
                action = dqn.choose_action(state)

                next_state, reward , done = myEnv.step(action,state)
                # print('state',state)
                # print('next_state',next_state)
                # print('action',action)
                # if(episode_count==0): break
                dqn.store_transition(state, action, reward, next_state)
                ep_reward += reward
                count_iteration+=1
                temp.append([action,next_state,reward,done,episode_count,count_iteration,ep_reward])
                log_text.append(temp)

                if dqn.memory_counter >= MEMORY_CAPACITY:
                    dqn.learn()

                    # print('dqn.eval_net.out.weight.grad',dqn.eval_net.out.weight.grad)
                if done:
                    print("Done!")
                    step_temp=step_
                    print("episode: {} , the episode reward is {}".format(i, ep_reward))
                    break
                # print("episode: {} , the episode reward is {}".format(i, ep_reward))
                state = next_state
            
            episode_count+=1
            if step_temp <done_step:
                print("模型保存....")
                dqn.save_best()
                print("episode: {} , the episode reward is {}".format(i, ep_reward))
                done_step=step_temp
            if i % log_interval == 0:
                print("模型保存....")
                dqn.save()
                print("episode: {} , the episode reward is {}".format(i, ep_reward))
        dqn.save_current(data)
        log_text = pd.DataFrame(log_text)
        log_text.to_csv(str(data)+'log_text_2_9.csv')

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    directory = 'model_save'
    main(300)
    main(400)
    main(500)

