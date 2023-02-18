import matplotlib.pyplot as plt
import os

tar_dir = 'DQN\model_save'
if len(os.listdir(tar_dir)) == 0:  # 目标文件夹内容为空的情况下
    print("目标文件夹为空")

# X = plt.imread("DQN\map.png")
# plt.imshow(X)