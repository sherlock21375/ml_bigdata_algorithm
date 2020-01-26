# -*- coding: utf-8 -*-
# @Time    : 2020/1/23 13:51
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : loadDataSet.py
# @Software: PyCharm
def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # vocabSet.append(document)   #如果使用这个，无法排除重复值，同时vocabSet应该设置为列表
        vocabSet = vocabSet | set(document)  # union of the two sets  |符号用于求两个集合的并集
        # print(vocabSet)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  #新建一个数目相同的list
    # print(len(returnVec))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# listOPosts,listClasses = loadDataSet()
# myVocabList = createVocabList(listOPosts)
# print(myVocabList)
# returnVec = setOfWords2Vec(myVocabList,listOPosts[0])#判断原来句子的每个单词在生成的不重复列表中具体位置，以第一句为例
# print(returnVec)
# trainMat = []
# for postinDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
# print(trainMat)