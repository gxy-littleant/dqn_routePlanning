from sklearn.cluster import KMeans
import numpy as np
import cv2
from sklearn.metrics import euclidean_distances
import matplotlib.pyplot as plt
import random
def Cluster_group():
    img = cv2.imread("bohai_black&white.png")
    if len(img.shape)==3:
        img=img[:,:,0]

    solve_pos = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j]==255:
                solve_pos.append([i,j])

    kmeans = KMeans(n_clusters=500, random_state=0).fit(solve_pos)

    print(kmeans.labels_)
    print(kmeans.predict([[0, 0], [4, 4], [2, 1]]))
    return kmeans,np.asarray(solve_pos)

model,img_channel=Cluster_group()
cluster_list=np.argmin(euclidean_distances(model.cluster_centers_,img_channel),axis=1)
print(random.sample(list(cluster_list),1))
plt.scatter(img_channel[:,0],img_channel[:,1],s=0.5)
for i in cluster_list:
    plt.scatter(img_channel[i][0],img_channel[i][1],c="red",s=3)

plt.show()
# print(img_channel.shape)

