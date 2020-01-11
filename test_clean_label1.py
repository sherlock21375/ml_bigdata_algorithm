# -*- coding: utf-8 -*-
# @Time    : 2020/1/10 10:50
# @Author  : Sherlock June
# @Email   : 1738502793@qq.com
# @File    : test_clean_label1.py
# @Software: PyCharm
import pandas as pd
import numpy as np
import re
pd.set_option('expand_frame_repr', False)  # 数据超过总宽度后，是否折叠显示，看一下更多的数据
# pd.options.display.max_rows = None  #取消输出列限制

train=pd.read_csv("../data/meinian_round1_train_20180408.csv",encoding='gbk')
train.columns=['vid','y1','y2','y3','y4','y5'] #修改原有的标签
print("----------1----------")
print(train["y3"].value_counts()) #统计y1值出现的次数
# print(train.loc[:,'y1'].value_counts()) #统计y1值出现的次数，另一种方式
# print(train.groupby(['y1']).size()['弃查']) #查看具体某列数据值的出现次数
# print(train.groupby(['vid'])['y1'].value_counts())
#输出y1中无标签样本，y2同理
train.drop(index=train.index[train['y1'].str.contains("未查|弃查")==True],inplace=True)

# 中位数替换y3中的不等号数值
range_number=list(set([s for s in train.y3 if ">" in str(s)]))
# print(range_number)
# print(len(range_number))
replace_range_number={}
for x in range_number:
    x_num=pd.to_numeric(x.replace(">",""),errors='raise')
    # print(x_num)  //将含有>数据更改
    m=pd.to_numeric(train.y3[pd.to_numeric(train.y3,errors='coerce')>x_num],errors='raise').median()  #中位数
    replace_range_number[x]=m
for i,v in replace_range_number.items():
    train['y3']=train['y3'].replace(i,v)
# 处理y3的伪文本
train.y3=train.y3.astype(str).str.replace("+","")
train.y3=train.y3.astype(str).str.replace("7.75轻度乳糜","7.75")
train.y3=train.y3.astype(str).str.replace("2.2.8","2.28")

# 处理标签异常的样本
train.iloc[:, 1:] = train.iloc[:, 1:].apply(lambda x: pd.to_numeric(x, errors="raise"), axis=0)
train.drop(index=train.index[train['y1'] == 0], inplace=True)
train.drop(index=train.index[train['y2'] == 0], inplace=True)
train.drop(index=train.index[train['y2'] > 10000], inplace=True)
train.drop(index=train.index[pd.isnull(train['y3'])], inplace=True)
train.drop(index=train.index[train['y5'] < 0], inplace=True)

train = train.reset_index(drop=True)
train.to_pickle("./data/train.pkl")

