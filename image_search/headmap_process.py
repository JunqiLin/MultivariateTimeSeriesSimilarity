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
import statsmodels.api as sm

import PIL.Image as Image
import os

import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )



IMAGES_PATH = '/Users/linjunqi/Desktop/MTS/image_search/'  # 图片集地址
IMAGES_FORMAT = ['.PNG','.png']  # 图片格式
IMAGE_SIZE = 256  # 每张小图片的大小
IMAGE_ROW = 2 # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 5  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = '/Users/linjunqi/Desktop/MTS/image_search/'  # 图片转换后的地址


roll_len = 18
index_num = 10

data = pd.read_excel('./heat_map.xls')
data.columns = ['date','iir', 'cpi', 'ppi','m1','m2','fai','pmi','pmim','pr','pc']

data['date'] = pd.to_datetime(data['date'])
data.set_index("date", inplace=True)

res = sm.tsa.seasonal_decompose(data,two_sided=False)


cycle = res.resid[12:]

#cycle.plot()
#cycle.to_csv('cycle.csv')
length = len(cycle)
ind = np.arange(length)
cycle = cycle.set_index(ind)

obj_data = cycle[-roll_len:]
lev_data = cycle[:-roll_len]


result = []

def image_compose(i):
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1])
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
    pic_name = str(i) + 'result.jpg'
    return to_image.save(pic_name) # 保存新图

#for i in range(len(lev_data)):
#    if i+roll_len<len(lev_data)+1:
#        block_mtx = lev_data[i:i+roll_len]
#        iir = pd.DataFrame(block_mtx['iir'])
#        cpi = pd.DataFrame(block_mtx['cpi'])
#        ppi = pd.DataFrame(block_mtx['ppi'])
#        m1 = pd.DataFrame(block_mtx['m1'])
#        m2 = pd.DataFrame(block_mtx['m2'])
#        fai = pd.DataFrame(block_mtx['fai'])
#        pmi = pd.DataFrame(block_mtx['pmi'])
#        pmim = pd.DataFrame(block_mtx['pmim'])
#        pr = pd.DataFrame(block_mtx['pr'])
#        pc = pd.DataFrame(block_mtx['pc'])
#        
#        plt.figure()
#        sns.heatmap(iir,yticklabels=False) 
#        plt.savefig("iir.png")
#        plt.figure()
#        sns.heatmap(cpi,yticklabels=False)
#        plt.savefig("cpi.png")
#        plt.figure()
#        sns.heatmap(ppi,yticklabels=False)
#        plt.savefig("ppi.png")
#        plt.figure()
#        sns.heatmap(m1,yticklabels=False)
#        plt.savefig("m1.png")
#        plt.figure()
#        sns.heatmap(m2,yticklabels=False)
#        plt.savefig("m2.png")
#        plt.figure()
#        sns.heatmap(fai,yticklabels=False)
#        plt.savefig("fai.png")
#        plt.figure()
#        sns.heatmap(pmi,yticklabels=False)
#        plt.savefig("pmi.png")
#        plt.figure()
#        sns.heatmap(pmim,yticklabels=False)
#        plt.savefig("pmim.png")
#        plt.figure()
#        sns.heatmap(pr,yticklabels=False)
#        plt.savefig("pr.png")
#        plt.figure()
#        sns.heatmap(pc,yticklabels=False)
#        plt.savefig("pc.png")
#        
#        image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
#               os.path.splitext(name)[1] == item]
#
#
#        if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
#            raise ValueError("合成图片的参数和要求的数量不能匹配！")
#
#        image_compose(i)
#    else:
#        print("FINISH")



        
#iir = pd.DataFrame(obj_data['iir'])
#cpi = pd.DataFrame(obj_data['cpi'])
#ppi = pd.DataFrame(obj_data['ppi'])
#m1 = pd.DataFrame(obj_data['m1'])
#m2 = pd.DataFrame(obj_data['m2'])
#fai = pd.DataFrame(obj_data['fai'])
#pmi = pd.DataFrame(obj_data['pmi'])
#pmim = pd.DataFrame(obj_data['pmim'])
#pr = pd.DataFrame(obj_data['pr'])
#pc = pd.DataFrame(obj_data['pc'])


#plt.figure()
#sns.heatmap(iir,yticklabels=False) 
#plt.savefig("iir.png")
#plt.figure()
#sns.heatmap(cpi,yticklabels=False)
#plt.savefig("cpi.png")
#plt.figure()
#sns.heatmap(ppi,yticklabels=False)
#plt.savefig("ppi.png")
#plt.figure()
#sns.heatmap(m1,yticklabels=False)
#plt.savefig("m1.png")
#plt.figure()
#sns.heatmap(m2,yticklabels=False)
#plt.savefig("m2.png")
#plt.figure()
#sns.heatmap(fai,yticklabels=False)
#plt.savefig("fai.png")
#plt.figure()
#sns.heatmap(pmi,yticklabels=False)
#plt.savefig("pmi.png")
#plt.figure()
#sns.heatmap(pmim,yticklabels=False)
#plt.savefig("pmim.png")
#plt.figure()
#sns.heatmap(pr,yticklabels=False)
#plt.savefig("pr.png")
#plt.figure()
#sns.heatmap(pc,yticklabels=False)
#plt.savefig("pc.png")

#image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
#               os.path.splitext(name)[1] == item]
#
#
#if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
#    raise ValueError("合成图片的参数和要求的数量不能匹配！")
 
# 定义图像拼接函数
#def image_compose(i):
#    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) #创建一个新图
#    for y in range(1, IMAGE_ROW + 1):
#        for x in range(1, IMAGE_COLUMN + 1):
#            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1])
#            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
#    pic_name = 'result'+str(i) + '.jpg'
#    return to_image.save(pic_name) # 保存新图


#image_compose() #调用函数


