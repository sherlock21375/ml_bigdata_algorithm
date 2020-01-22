# -*- coding: utf-8 -*-
# @Time    : 2020/1/22 18:47
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : lenses.py
# @Software: PyCharm
from make_tree import *
from visualization_tree import *

fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
# print(lense)
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLabels)
print("--------------------------------")
print("----------this is my tree------")
print(lensesTree)
print("--------------------------------")
createPlot(lensesTree)

