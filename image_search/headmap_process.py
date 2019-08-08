#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:51:16 2019

@author: linjunqi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

data = pd.read_excel('./direction_mtx_data.xls')
data.columns = ['date','iir', 'cpi', 'ppi','m1','m2','fai','pmi','pmim','pr','pc']
data = data.drop(['date'],axis = 1)
#data['date'] = pd.to_datetime(data['date'])
#data.set_index("date", inplace=True)



def df_donchian(df_data, n ,high_mtx, low_mtx, array = False):

    for index, row in df_data.iteritems():
        up,down = donchian(row,n,True)
        high_mtx[index] = up
        low_mtx[index] = down
#        print("up is %s"%(up))
#        print("down is %s"%(down))
    return high_mtx, low_mtx

 
# cmap用matplotlib colormap
sns.heatmap(data, linewidths = 0.05, cmap='rainbow') 
# rainbow为 matplotlib 的colormap名称
ax2.set_title('matplotlib colormap')
ax2.set_xlabel('region')

ax2.set_ylabel('kind')

