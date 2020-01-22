# -*- coding: utf-8 -*-
# @Time    : 2020/1/21 15:30
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : make_tree.py
# @Software: PyCharm

from countShannon_chooseBestF import *
import operator

def majorityCnt(classList):   #返回出现次数最多的分类名称
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    labels_new = labels[:]
    classList = [example[-1] for example in dataSet]   #得到最后的特征值列表
    # print("----------classList----------")
    # print(classList)
    if classList.count(classList[0]) == len(classList):  #说明此时的classList全是一种分类结果，每个分支下的所有实例都具有相同的分类结果
        # print("----------分类只有一种结果----------")
        # print(classList[0])
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet #程序遍历完所有划分数据集的属性
        # print("----------遍历完所有划分数据集的属性----------")   #本例中未使用到
        # print(majorityCnt(classList))
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels_new[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels_new[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  #提取最佳特征值组合
    # print("----------featValues----------")
    # print(featValues)
    uniqueVals = set(featValues)  #提取最佳特征值的unique，构造字典
    # print("----------uniqueVals----------")
    # print(uniqueVals)
    for value in uniqueVals:
        subLabels = labels_new[:]       #copy all of labels_new, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

# myDat,labels = createDataSet()
# print("----------output-myDat----------")
# print(myDat)
# print("----------output-labels---------")
# print(labels)
# myTree = createTree(myDat,labels)
# print("----------this is my tree-------")
# print(myTree)
# print("--------------------------------")
# print("----------finish make_tree------")
# print("--------------------------------")