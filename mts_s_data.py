#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 15:04:19 2019

@author: linjunqi
"""


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import statsmodels.api as sm


"""
obj_len: 目标序列长度
que_len: 查询序列长度
gap: 滚动查询间隔
total: 选取样本时间长度
diff: 滚动停止距离
INFF: 无限大
cut_data: 从120起才有较多的全样数据，之后看情况怎么做
"""
obj_len = 18
que_len =18
gap = 1
total = 200
diff = 18
INFF = 100000
cut_data = 220
index_list = []
distance_list = []
total_distance_list = []


"""
to do list:
dtw算法有待改进，以可以处理周期性序列，算法复杂度可能可以改进
指标可以改进
pca降纬的方法可以改进
"""



"""
DTW算法，将两个时间序列做相似度对比
"""
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


"""
将数据分割成与目标数据相匹配的长度
"""
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
 
    
"""  
 对目标数据之前的时间序列做滚动搜寻，寻找匹配序列
"""
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
            print("min distance is : %s"%(min_distance))
            print("the index is : %s"%(search_index))
    return min_distance,search_index


"""
将数据进行标准化，避免差别过大的量纲影响分析结果
"""
def normalize(data):  
    return (data-np.min(data))/(np.max(data)-np.min(data))


"""
对时间序列做移动平均
"""
def moving_average(a, n=6) :  
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def sigmoid(x):
    return 1/(1+np.exp(-x))

"""
用前后行的均值代替中间的空缺值
"""
def mean_replace(np_arr):  
    nan_pos = np.array(np.where(np.isnan(np_arr))).T
    for pos in nan_pos:
        if pos-1>=0 and pos+1<len(np_arr):
            mean = (np_arr[pos+1] + np_arr[pos-1])/2
            np_arr[pos] = mean
    return np_arr


#macro_data = pd.read_excel('./data/macro_data.xls')
##test_macro_data = macro_data.fillna(axis=0,method='ffill')
#test_macro_data = macro_data[cut_data:]
#
#IIR = test_macro_data.iloc[:,1]

#def decomposition_to_resid(df_data):
#    

data= pd.read_excel('./data/macro_data.xls')[cut_data:]
data.columns = ['date','iir', 'cpi', 'ppi','m1','m2','fai','pmi','pmim','pr','pc']
data['date'] = pd.to_datetime(data['date'])
data.set_index("date", inplace=True)


#
fai= mean_replace(np.array(data.fai))
data = data.drop(['fai'],axis = 1)
data['fai'] = fai
data = data.fillna(axis=0,method='ffill')

res = sm.tsa.seasonal_decompose(data,two_sided=False)
resid = res.resid
resid = resid[12:]

resid.plot()
plt.show()


data = np.array(resid)
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
plt.plot(pd.DataFrame(obj_data))
plt.title('Object Series')
plt.sca(ax1)
plt.plot(pd.DataFrame(min_distance_series))
plt.title('Pair Series')

plt.show()
print(result_list.T)
print("min distance is %s"%(min_distance))
print("min distance index is %s"%(min_index))





