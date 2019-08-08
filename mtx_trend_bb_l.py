#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 16:15:16 2019

@author: linjunqi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import talib 

roll_len = 18
index_num = 10
window_len = 3



def mtx_similar(arr1, arr2):

    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    if arr1.shape != arr2.shape:
        minx = min(arr1.shape[0],arr2.shape[0])
        miny = min(arr1.shape[1],arr2.shape[1])
        differ = arr1[:minx,:miny] - arr2[:minx,:miny]
    else:
        differ = arr1 - arr2
    dist = np.linalg.norm(differ, ord='fro')
    len1 = np.linalg.norm(arr1)
    len2 = np.linalg.norm(arr2)     
    denom = (len1 + len2) / 2
    similar = 1 - (dist / denom)
    return similar


#data = pd.read_excel('./data/direction_mtx_data.xls')
#data.columns = ['date','iir', 'cpi', 'ppi','m1','m2','fai','pmi','pmim','pr','pc']
#data['date'] = pd.to_datetime(data['date'])
#data.set_index("date", inplace=True)

data = pd.read_csv('./data/trend.csv',index_col = 0)
donchian_high_data = pd.read_csv('./data/donchian_high_mtx.csv',index_col = 0)[:-1]
donchian_low_data = pd.read_csv('./data/donchian_low_mtx.csv',index_col = 0)[:-1]

origin_data = pd.read_excel('./data/direction_mtx_data.xls')
origin_data.columns = ['date','iir', 'cpi', 'ppi','m1','m2','fai','pmi','pmim','pr','pc']
origin_data['date'] = pd.to_datetime(origin_data['date'])
origin_data.set_index("date", inplace=True)

donchian_high_data = np.array(donchian_high_data)
donchian_low_data = np.array(donchian_low_data)
data = np.array(data)
shape = data.shape


direc_result = np.zeros(data.shape) 
mask = data>donchian_high_data
direc_result[mask] = 1
mask = data<donchian_low_data
direc_result[mask] = -1

obj_data = direc_result[-roll_len:]
lev_data = direc_result[:-roll_len]

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
max_index = np.argmin(result)
#plt.plot(result)
#plt.show()
direc_result = pd.DataFrame(direc_result)
direc_result.to_csv('direction_mtx.csv')
#print("the most min index is %s"%(min_index))
#print("the most max idnex is %s"%(max_index))



ori_obj_data = origin_data[-18:]
ori_min_data = origin_data[min_index+12: min_index+12+18]
ori_max_data = origin_data[max_index+12: max_index+12+18]

print("the object data period is %s to %s"%(ori_obj_data.index[0],ori_obj_data.index[-1]))
print("the most similar time period is %s to %s"%(ori_min_data.index[0],ori_min_data.index[-1]))
print("the most unsimilar time period is %s to %s"%(ori_max_data.index[0],ori_max_data.index[-1]))
#
length = len(ori_obj_data)
ind = np.arange(length)
ori_obj_data = ori_obj_data.set_index(ind)
ori_min_data = ori_min_data.set_index(ind)
ori_max_data = ori_max_data.set_index(ind)

fig = plt.figure(figsize=(40,20))
ax1 = fig.add_subplot(2,5,1)
ax2 = fig.add_subplot(2,5,2)
ax3 = fig.add_subplot(2,5,3)
ax4 = fig.add_subplot(2,5,4)
ax5 = fig.add_subplot(2,5,5)
ax6 = fig.add_subplot(2,5,6)
ax7 = fig.add_subplot(2,5,7)
ax8 = fig.add_subplot(2,5,8)
ax9 = fig.add_subplot(2,5,9)
ax10 = fig.add_subplot(2,5,10)

ax1.plot(ori_obj_data['iir'])
ax1.plot(ori_min_data['iir'])
ax1.plot(ori_max_data['iir'])
ax1.legend(["object", "min","max"], loc='upper left')


ax2.plot(ori_obj_data['cpi'])
ax2.plot(ori_min_data['cpi'])
ax2.plot(ori_max_data['cpi'])
ax2.legend(["object", "min","max"], loc='upper left')

ax3.plot(ori_obj_data['ppi'])
ax3.plot(ori_min_data['ppi'])
ax3.plot(ori_max_data['ppi'])
ax3.legend(["object", "min","max"], loc='upper left')

ax4.plot(ori_obj_data['m1'])
ax4.plot(ori_min_data['m1'])
ax4.plot(ori_max_data['m1'])
ax4.legend(["object", "min","max"], loc='upper left')

ax5.plot(ori_obj_data['m2'])
ax5.plot(ori_min_data['m2'])
ax5.plot(ori_max_data['m2'])
ax5.legend(["object", "min","max"], loc='upper left')

ax6.plot(ori_obj_data['fai'])
ax6.plot(ori_min_data['fai'])
ax6.plot(ori_max_data['fai'])
ax6.legend(["object", "min","max"], loc='upper left')

ax7.plot(ori_obj_data['pc'])
ax7.plot(ori_min_data['pc'])
ax7.plot(ori_max_data['pc'])
ax7.legend(["object", "min","max"], loc='upper left')

ax8.plot(ori_obj_data['pmi'])
ax8.plot(ori_min_data['pmi'])
ax8.plot(ori_max_data['pmi'])
ax8.legend(["object", "min","max"], loc='upper left')

ax9.plot(ori_obj_data['pmim'])
ax9.plot(ori_min_data['pmim'])
ax9.plot(ori_max_data['pmim'])
ax9.legend(["object", "min","max"], loc='upper left')

ax10.plot(ori_obj_data['pr'])
ax10.plot(ori_min_data['pr'])
ax10.plot(ori_max_data['pr'])
ax10.legend(["object", "min","max"], loc='upper left')

fig.savefig('result.png',dpi = 100)
