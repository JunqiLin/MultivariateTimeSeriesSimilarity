#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:09:47 2019

@author: linjunqi
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import talib 

roll_len = 6
index_num = 10
window_len = 3

#def mtx_similar(arr1,arr2):
#    '''
#    将矩阵展平成向量，计算向量的乘积除以模长。
#    注意有展平操作。
#    :param arr1:矩阵1
#    :param arr2:矩阵2
#    :return:实际是夹角的余弦值，ret = (cos+1)/2
#    '''
#    arr1 = np.array(arr1)
#    arr2 = np.array(arr2)
#    farr1 = arr1.ravel()
#    farr2 = arr2.ravel()
#    len1 = len(farr1)
#    len2 = len(farr2)
#    if len1 > len2:
#        farr1 = farr1[:len2]
#    else:
#        farr2 = farr2[:len1]
#
#    numer = np.sum(farr1 * farr2)
#    denom = np.sqrt(np.sum(farr1**2) * np.sum(farr2**2))
#    similar = numer / denom 
#    return  (similar+1) / 2    



data = pd.read_excel('./data/direction_mtx_data.xls')
data.columns = ['date','iir', 'cpi', 'ppi','m1','m2','fai','pmi','pmim','pr','pc']
data['date'] = pd.to_datetime(data['date'])
data.set_index("date", inplace=True)


colname = ['iir', 'cpi', 'ppi','m1','m2','fai','pmi','pmim','pr','pc']

donchian_high_mtx = pd.DataFrame(columns=colname)
donchian_low_mtx = pd.DataFrame(columns = colname)


#
res = sm.tsa.seasonal_decompose(data,two_sided=False)
res.plot()
trend_data = res.trend
trend_data = trend_data[12:]
#trend_data.plot()

#test = np.array([1,4,2,2]).astype(np.double)

def donchian(data, n , array = False):
    up = talib.MAX(data, n)
    down = talib.MIN(data, n)
    
    if array:
        return up, down
    return up[-1],down[-1]

def df_donchian(df_data, n ,high_mtx, low_mtx, array = False):

    for index, row in df_data.iteritems():
        up,down = donchian(row,n,True)
        high_mtx[index] = up
        low_mtx[index] = down
#        print("up is %s"%(up))
#        print("down is %s"%(down))
    return high_mtx, low_mtx
        
donchian_high_mtx, donchian_low_mtx = df_donchian(trend_data,3,donchian_high_mtx, donchian_low_mtx)

trend_data.to_csv('trend.csv')
donchian_high_mtx.to_csv('donchian_high_mtx.csv')
donchian_low_mtx.to_csv('donchian_low_mtx.csv')


#donchian_high_mtx = donchian_high_mtx[2:]
#donchian_high_mtx[1:] = donchian_high_mtx[:-1] 
#donchian_low_mtx = donchian_low_mtx[2:]
#data = np.array(data)
#data = np.reshape(data,[-1])

#donchian_high_mtx = np.array(donchian_high_mtx)
#donchian_low_mtx = np.array(donchian_low_mtx)
#donchian_high_mtx = np.reshape(donchian_high_mtx, [-1])
#donchian_low_mtx = np.reshape(donchian_low_mtx, [-1])
#
#
#direc_result = np.empty(data.shape) 
#
#direc_result = np.where(donchian_high_mtx - data <0,1,0)

#a = np.array([1,3,2,5,2])
#b = np.array([4,2,3,1,7])
#c = np.array([4,2,5,1,3])
#
#d = b-a
#e = np.where(d>0,1,0)
#print(e)

#a = donchian(test, 2, True)
#print(a)

#obj_data = data[-roll_len:]
#lev_data = data[:-roll_len]
#
#result = []
#min_index = 0
#
#for i in range(len(lev_data)):
#    block_mtx = np.zeros(shape=[roll_len, index_num])
#    if i+roll_len<len(lev_data)+1:
#        block_mtx = lev_data[i:i+roll_len]
#        re = mtx_similar(obj_data, block_mtx)
#        result.append(re)
#    
#
#result = np.array(result)
#min_index = np.argmax(result)
#plt.plot(result)
#plt.show()
#
#
#print(min_index)
#date_index = list(lev_data.index)
#begin_date = date_index[min_index]
#end_date = date_index[min_index+6]
#now_bgn = list(obj_data.index)[0]
#now_end = list(obj_data.index)[5]
#print("Now matrix is from %s to %s"%(str(now_bgn), str(now_end)))
#print("the most similar matrix is from %s to %s"%(str(begin_date), str(end_date)))


