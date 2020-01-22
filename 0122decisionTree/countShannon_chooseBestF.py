# -*- coding: utf-8 -*-
# @Time    : 2020/1/21 12:23
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : countShannon_chooseBestF.py
# @Software: PyCharm

from math import log

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the number of unique elements and their occurance
        currentLabel = featVec[-1]   #the last column
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # print(labelCounts)  #{'yes': 2, 'no': 3} 计算字典数目
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    # print("----------calcShannonEnt-shannonEnt---------")
    # print(shannonEnt)
    return shannonEnt

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def splitDataSet(dataSet, axis, value):     #分割数据集，筛查出第axis列值等于value的其他列
    retDataSet = []
    for featVec in dataSet:
        # print(featVec[axis]) #输出数据集第axis列
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting  #举例如果是axis=1，对应判定第2列，对列表操作则是第1列
            # print(reducedFeatVec)    #如果axis为0，那么列表输出时此处应该为空，通过下方直接extend所有
            reducedFeatVec.extend(featVec[axis+1:])
            # print(reducedFeatVec)
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)   #计算原始数据集香农熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features   第1个for循环遍历数据集中的所有特征值
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        # print("----------chooseBestFeatureToSplit-featList-%d---------" % i)
        # print(featList)
        uniqueVals = set(featList)       #get a set of unique values   集合，特征值
        # print("----------chooseBestFeatureToSplit-uniqueVals-%d-------" % i)
        # print(uniqueVals)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # print("----chooseBestFeatureToSplit-subDataSet--i:%d-value:%d--" % (i, value))
            # print(subDataSet)
            prob = len(subDataSet)/float(len(dataSet)) #划分出的数据集占全部的比例
            # print("----------chooseBestFeatureToSplit-prob--i:%d-value:%d--" % (i,value))
            # print(prob)
            newEntropy += prob * calcShannonEnt(subDataSet)
            # print("----------chooseBestFeatureToSplit-newEntropy--i:%d-value:%d--" % (i, value))
            # print(newEntropy)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        # print("----------chooseBestFeatureToSplit-infoGain--i:%d--" % i)
        # print(infoGain)
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer


# myDat,labels = createDataSet()
# print("----------output-myDat----------")
# print(myDat)
# print("----------output-labels---------")
# print(labels)
# shannonEnt = calcShannonEnt(myDat)
# print("----------output-shannonEnt-----")
# print(shannonEnt)
# splitDataResult = splitDataSet(myDat,0,1)
# print("----------output-splitDataResult-by-0-1----------")
# print(splitDataResult)
# splitDataResult = splitDataSet(myDat,0,0)
# print("----------output-splitDataResult-by-0-0----------")
# print(splitDataResult)
# bestFeature = chooseBestFeatureToSplit(myDat)
# print("----------output-bestFeature----------")
# print(bestFeature)