#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:28:52 2019

@author: linjunqi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

roll_len = 6
index_num = 10
def mtx_similar(arr1,arr2):
    '''
    将矩阵展平成向量，计算向量的乘积除以模长。
    注意有展平操作。
    :param arr1:矩阵1
    :param arr2:矩阵2
    :return:实际是夹角的余弦值，ret = (cos+1)/2
    '''
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    farr1 = arr1.ravel()
    farr2 = arr2.ravel()
    len1 = len(farr1)
    len2 = len(farr2)
    if len1 > len2:
        farr1 = farr1[:len2]
    else:
        farr2 = farr2[:len1]

    numer = np.sum(farr1 * farr2)
    denom = np.sqrt(np.sum(farr1**2) * np.sum(farr2**2))
    similar = numer / denom 
    return  (similar+1) / 2    



data = pd.read_excel('./data/matrix_data.xls')
data.columns = ['date','iir', 'cpi', 'ppi','m1','m2','fai','pmi','pmim','pr','pc']
data['date'] = pd.to_datetime(data['date'])
data.set_index("date", inplace=True)


obj_data = data[-roll_len:]
lev_data = data[:-roll_len]

result = []
min_index = 0

for i in range(len(lev_data)):
    block_mtx = np.zeros(shape=[roll_len, index_num])
    if i+roll_len<len(lev_data)+1:
        block_mtx = lev_data[i:i+roll_len]
        re = mtx_similar(obj_data, block_mtx)
        result.append(re)
    

result = np.array(result)
min_index = np.argmax(result)
plt.plot(result)
plt.show()


print(min_index)
date_index = list(lev_data.index)
begin_date = date_index[min_index]
end_date = date_index[min_index+6]
now_bgn = list(obj_data.index)[0]
now_end = list(obj_data.index)[5]
print("Now matrix is from %s to %s"%(str(now_bgn), str(now_end)))
print("the most similar matrix is from %s to %s"%(str(begin_date), str(end_date)))


