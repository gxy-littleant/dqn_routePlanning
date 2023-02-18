import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import euclidean_distances
import random
from sklearn.cluster import KMeans
from sqlalchemy import column
class env(object):
    
    def __init__(self,data) -> None:
        self.img=cv2.imread("bohai_black&white.png")
        if len(self.img.shape)==3:
            self.img=self.img[:,:,0]
        self.solve_pos=[]

        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                if self.img[i,j]==255:
                    self.solve_pos.append([i,j])

        self.distination_all_set=np.loadtxt("tian,dalian,yantai,qinghuangdao.txt")
        # 随机打乱 random.shuffle这个函数他是return None
        
        # print("self.distination_all_set",self.distination_all_set)
        self.distination = []
        # print("self.distination_all_set",self.distination_all_set.shape)
        self.destination_name_list = ['天津','大连','烟台','秦皇岛']
        
        # self.destination_name = self.destination_name_list[3]
        self.last_dis=np.inf
        self.class_num=data
        self.kmodel= KMeans(n_clusters=self.class_num, random_state=0)
        self.kmodel.fit_predict(np.array(self.solve_pos))
        # print('1',self.kmodel.cluster_centers_)
        self.cluster_list=np.argmin(euclidean_distances(self.kmodel.cluster_centers_,np.asarray(self.solve_pos)),axis=1)
        # print('1',np.argmin(euclidean_distances(self.kmodel.cluster_centers_,np.asarray(self.solve_pos)),axis=1))
        self.reset_set=self.kmodel
        self.current_pos=self.reset()

    def reset(self):
        # print('1',self.kmodel.cluster_centers_)
        # print('self.cluster_list',self.cluster_list)
        index=random.sample(list(self.cluster_list),1)
        # print(index)
        row=np.array(self.solve_pos)[index[0]][0]
        col=np.array(self.solve_pos)[index[0]][1]

        
        np.random.shuffle(self.destination_name_list)
        self.destination_name = self.destination_name_list[random.randint(0,3)]

        if(self.destination_name=='天津'):
            self.distination = self.distination_all_set[0:40]
        elif(self.destination_name=='大连'):
            self.distination = self.distination_all_set[40:80]
        elif(self.destination_name=='烟台'):
            self.distination = self.distination_all_set[80:120]
        elif(self.destination_name=='秦皇岛'):
            # print('self.distination_all_set[120:160]',self.distination_all_set[120:160])
            self.distination = self.distination_all_set[120:160]
        print('destination_name:',self.destination_name)
        # print('destination shape:',np.array(self.distination).shape)
        # print('destination:',self.distination)
        return [row,col]
        # return np.array(self.solve_pos[-1])
        
    def compute_dis(self,state):
        # print('state shape:\n',np.array(state).shape)
        # print('distination shape:\n',np.array(self.distination).shape)
        dist_min_pos_index=np.argmin(euclidean_distances(np.array(state).reshape(1,-1),np.array(self.distination)))
        dist=np.min(euclidean_distances(np.array(state).reshape(1,-1),np.array(self.distination)))
        return dist_min_pos_index,dist
    
    def step(self,action,state):
        next_pos=[0,0]
        self.current_pos = state
        current_pos=self.current_pos
        done=False
        if action==0:
            temp_i=current_pos[0]-1
            temp_j=current_pos[1]
            # 超出边界
            if temp_i<0:
                reward=-20
                next_pos=current_pos
            else:
                # 在边界范围内
                temp=[temp_i,temp_j]
                # 不在可行解里
                if temp not in self.solve_pos:
                    reward=-15
                    next_pos=current_pos
                # 在可行解
                else:
                    next_pos=temp
                    index,dist=self.compute_dis(next_pos)
                    # 下一个目的地就是终点
                    if next_pos in self.distination:
                        reward=-1
                        done=True
                    # 距离越来越远
                    elif self.last_dis<dist:
                        reward=-10
                    # 方向没偏
                    else:
                        reward=-5
                    self.last_dis=dist
            self.current_pos=next_pos
            return next_pos,reward,done

        if action==1:
            temp_i=current_pos[0]-1
            temp_j=current_pos[1]+1
            # 超出范围
            if temp_i<0 or temp_j>=self.img.shape[1]:
                reward=-20
                next_pos=current_pos
                
            else:
                temp=[temp_i,temp_j]
                # 不在可行解里
                if temp not in self.solve_pos:
                    reward=-15
                    next_pos=current_pos
                    
                else:
                    next_pos=temp
                    index,dist=self.compute_dis(next_pos)
                    if next_pos in self.distination:
                        reward=-1
                        done=True
                    elif self.last_dis<dist:
                        reward=-10

                    else:
                        reward=-5
                    self.last_dis=dist
            self.current_pos=next_pos
            return next_pos,reward,done

        if action==2:
            temp_i=current_pos[0]
            temp_j=current_pos[1]+1
            if temp_i<0 or temp_j>=self.img.shape[1]:
                reward=-20
                next_pos=current_pos
            else:
                temp=[temp_i,temp_j]
                if temp not in self.solve_pos:
                    reward=-15
                    next_pos=current_pos
                else:
                    next_pos=temp
                    index,dist=self.compute_dis(next_pos)
                    if next_pos in self.distination:
                        reward=-1
                        done=True
                    elif self.last_dis<dist:
                        reward=-10
                    else:
                        reward=-5
                    self.last_dis=dist
            self.current_pos=next_pos
            return next_pos,reward,done

        if action==3:
            temp_i=current_pos[0]+1
            temp_j=current_pos[1]+1
            if temp_i>=self.img.shape[0] or temp_j>=self.img.shape[1]:
                reward=-20
                next_pos=current_pos
            else:
                temp=[temp_i,temp_j]
                if temp not in self.solve_pos:
                    reward=-15
                    next_pos=current_pos
                else:
                    next_pos=temp
                    index,dist=self.compute_dis(next_pos)
                    if next_pos in self.distination:
                        reward=-1
                        done=True
                    elif self.last_dis<dist:
                        reward=-10

                    else:
                        reward=-5
                    self.last_dis=dist
            self.current_pos=next_pos
            return next_pos,reward,done

        if action==4:
            temp_i=current_pos[0]+1
            temp_j=current_pos[1]
            if temp_i>=self.img.shape[0] or temp_j>=self.img.shape[1]:
                reward=-20
                next_pos=current_pos
            else:
                temp=[temp_i,temp_j]
                if temp not in self.solve_pos:
                    reward=-15
                    next_pos=current_pos
                else:
                    next_pos=temp
                    index,dist=self.compute_dis(next_pos)
                    if next_pos in self.distination:
                        reward=-1
                        done=True
                    elif self.last_dis<dist:
                        reward=-10

                    else:
                        reward=-5
                    self.last_dis=dist
            self.current_pos=next_pos
            return next_pos,reward,done

        if action==5:
            temp_i=current_pos[0]+1
            temp_j=current_pos[1]-1
            if temp_j<0 or temp_i>=self.img.shape[0]:
                reward=-20
                next_pos=current_pos
            else:
                temp=[temp_i,temp_j]
                if temp not in self.solve_pos:
                    reward=-15
                    next_pos=current_pos
                else:
                    next_pos=temp
                    index,dist=self.compute_dis(next_pos)
                    if next_pos in self.distination:
                        reward=-1
                        done=True
                    elif self.last_dis<dist:
                        reward=-10

                    else:
                        reward=-5
                    self.last_dis=dist
            self.current_pos=next_pos
            return next_pos,reward,done

        if action==6:
            temp_i=current_pos[0]
            temp_j=current_pos[1]-1
            if temp_j<0 or temp_i>=self.img.shape[0]:
                reward=-20
                next_pos=current_pos
            else:
                temp=[temp_i,temp_j]
                if temp not in self.solve_pos:
                    reward=-15
                    next_pos=current_pos
                else:
                    next_pos=temp
                    index,dist=self.compute_dis(next_pos)
                    if next_pos in self.distination:
                        reward=-1
                        done=True
                    elif self.last_dis<dist:
                        reward=-10

                    else:
                        reward=-5
                    self.last_dis=dist
            self.current_pos=next_pos
            return next_pos,reward,done

        if action==7:
            temp_i=current_pos[0]-1
            temp_j=current_pos[1]-1
            if temp_i<0 or temp_j<0:
                reward=-20
                next_pos=current_pos
            else:
                temp=[temp_i,temp_j]
                if temp not in self.solve_pos:
                    reward=-15
                    next_pos=current_pos
                else:
                    next_pos=temp
                    index,dist=self.compute_dis(next_pos)
                    if next_pos in self.distination:
                        reward=-1
                        done=True
                    elif self.last_dis<dist:
                        reward=-10

                    else:
                        reward=-5
                    self.last_dis=dist
            self.current_pos=next_pos
            return next_pos,reward,done
        
            
# a=env()
# ls=[0,1,2,3,4,5,6,7]
# action=random.sample(ls,1)

# state=a.get_current_state()
# print('init state:',state)

# for i in range(100):
#     print("----------------------------")
#     print("current state is ",state)
#     action=random.sample(ls,1)[0]
#     print("choose acion is ",action)
#     next_pos,reward,done=a.step(action)
#     print("log is ",next_pos,reward,done)
#     state = next_pos

