import matplotlib.pyplot as plt 
import random
import copy
import math
from itertools import product
import os
import numpy as np
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

H, W = (50, 50)
R = [5, 10]
UE = [x/2 for x in R]
cell_W, cell_H = (10, 10)

no_cells = H/cell_H * W/cell_W
targets = []
for h in range(int(abs(H/cell_H))):
    for w in range(int(abs(W/cell_W))):
        targets.append((w*cell_W + cell_W/2, h*cell_H + cell_H/2))


lower = [(ue, ue) for ue in UE]
upper = [(H-l[0], W-l[1]) for l in lower]



min_noS = [int((W*H)/(36*ue*ue)) for ue in UE]
max_noS = [int((W*H)/(4*ue*ue)) for ue in UE]




class ClusterSensors(object):
    def __init__(self, n_clusters=25):
        
        self.n_clusters = n_clusters
        
        self.centroids = {}
        self.ranges  = {}
        self.clusters = {}

        self.initialize()
    def initialize(self):
        
        for i in range(self.n_clusters):
            x = lower[0][0] + (upper[0][0] - lower[0][0])*random.random()
            y = lower[0][1] + (upper[0][1] - lower[0][1])*random.random()
            self.centroids[i] = [x,y]
            self.ranges[i] = 1

    def _distance(self, x, y):
        return math.sqrt((x[0]-y[0])**2 + (x[1] - y[1]) ** 2)

    def run(self, step=1000):
        # self.initialize()
        # for i in range(step):
        #     # assign
        #     self.clusters = {}
        #     for j, centroid in self.centroids.items():
        #         if centroid [0]== -1 and centroid[0] ==-1:
        #             continue
        #         self.clusters[j] = []
        #         for target in targets:
        #             dist = self._distance(target, centroid)
        #             if dist <= R[self.ranges[j]]:
        #                 self.clusters[j].append([target[0], target[1]])
                
        #     # update centroids
        #     for j, centroid in self.centroids.items():
        #         if len(self.clusters[j]) == 0:
        #             self.centroids[j] = [-1, -1]
        #             continue
        #         elif len(self.clusters[j]) == 1:
        #             continue
        #         new_centroid = np.mean(self.clusters[j], axis=0)
        #         self.centroids[j] = new_centroid
        
        # return self.centroids



        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(targets)
        predict = kmeans.predict(targets)
        print(kmeans.cluster_centers_)
        print(predict)
        return kmeans.cluster_centers_

        # mixtureGaussian = GaussianMixture(n_components=10).fit(targets)
        # params = mixtureGaussian.get_params()
        # predict = mixtureGaussian.predict_proba(targets)
        # print(predict)
        # print(params)

hsa = ClusterSensors(n_clusters=10)
sensors = hsa.run()
# sensors = list(sensors.values())
# new_sensors = []
# for s in sensors:
#     if list(s) not in new_sensors:
#         new_sensors.append(list(s))
# print(len(new_sensors))

new_sensors = sensors
fig, ax1 = plt.subplots(1)

xtarget = [x[0] for x in targets]
ytarget = [x[1] for x in targets]
xsensor_obj = [x[0] for x in new_sensors]
ysensor_obj = [x[1] for x in new_sensors]

ax1.scatter(xtarget, ytarget, marker=".")
ax1.scatter(xsensor_obj, ysensor_obj, marker="*")
for s in range(len(xsensor_obj)):
    if xsensor_obj[s] == -1:
        continue
    ax1.add_patch(plt.Circle((xsensor_obj[s], ysensor_obj[s]), R[0], color='r', alpha=0.5, fill=False))
ax1.set_aspect('equal', adjustable='datalim')
ax1.plot()
ax1.grid()

plt.show()
