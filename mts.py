#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 23:13:24 2019

@author: linjunqi
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
obj_len = 30
que_len = 30
gap = 1
total = 5000
diff = 500
INFF = 10000
index_list = []
distance_list = []
total_distance_list = []
#dtw算法有待改进，以可以处理周期性序列，算法复杂度可能可以改进
#指标可以改进
#pca降纬的方法可以改进
def DTW(s1, s2):    
    l1 = len(s1)
    l2 = len(s2)
    paths = np.full((l1 + 1, l2 + 1), np.inf) 
    paths[0, 0] = 0
    for i in range(l1):
        for j in range(l2):
            d = s1[i] - s2[j]
            cost = d ** 2
            paths[i + 1, j + 1] = cost + min(paths[i, j + 1], paths[i + 1, j], paths[i, j])

    paths = np.sqrt(paths)
    s = paths[l1, l2]
    return s, paths.T


def process_data(data, que_len, gap):
    bgn = 0
    end = que_len
    p_data = []
    while end<len(data)-diff:
        temp = data[bgn:end]
        p_data.append(temp)
        end += gap
        bgn += gap
    return np.array(p_data)    
    

def roll_search(pair_data,obj_data):
    min_distance = INFF
    search_index = 0 
    for index,slic in enumerate(pair_data):
        distance, e = DTW(slic,obj_data)
#        index_list.append(distance)
#        distance_list.append(index)     
        total_distance_list.append(distance)
        if distance<min_distance:
            min_distance = distance
            search_index = index
            index_list.append(search_index)
            distance_list.append(min_distance)
#            print("min distance is : %s"%(min_distance))
#            print("the index is : %s"%(search_index))
    return min_distance,search_index

def normalize(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

#def sigmoid(data):
#    return 1/(1+np.exp(-data))



one = np.array(pd.read_csv('./data/datanew.csv')['one'])[:total][np.newaxis,]
two = np.array(pd.read_csv('./data/datanew.csv')['two'])[:total][np.newaxis,]
data = np.concatenate([one, two], axis=0).T


obj_data = data[-obj_len:,]
pca=PCA(n_components=1)
pca.fit(data)
one_d_data = pca.transform(data)
one_d_train_data = one_d_data[:total-obj_len,:]
one_d_test_data = one_d_data[-obj_len:,:]
pair_data = process_data(one_d_train_data,que_len,gap)

min_distance, min_index = roll_search(pair_data,one_d_test_data)
result_list = np.array([index_list,distance_list])

min_distance_series = data[min_index:min_index+obj_len]
plt.figure(1)
ax1 = plt.subplot(2,2,1)
ax2 = plt.subplot(2,2,2)
ax3 = plt.subplot(2,1,2)
plt.sca(ax3)
plt.plot(total_distance_list,color = 'blue')
plt.title('Pair Distance')
plt.sca(ax2)
plt.plot(obj_data,color = 'red')
plt.title('Object Series')
plt.sca(ax1)
plt.plot(min_distance_series,color = 'red')
plt.title('Pair Series')

plt.show()
print(result_list.T)
print("min distance is %s"%(min_distance))
print("min distance index is %s"%(min_index))

