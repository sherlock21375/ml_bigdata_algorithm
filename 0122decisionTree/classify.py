# -*- coding: utf-8 -*-
# @Time    : 2020/1/21 18:57
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : classify.py
# @Software: PyCharm

from make_tree import *

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]  #此处做出修改
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    file = open(filename, 'wb+')
    pickle.dump(inputTree, file, 0)
    file.close()

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    data_dict = pickle.load(fr)
    print(data_dict)
    #也可以使用如下的方法，这样才能将str转化为byte
    # with open(filename, 'r') as data_file:
    #     data_dict = pickle.load(StrToBytes(data_file))
    # print(data_dict)
    return data_dict

result1 = classify(myTree,labels,[1,0])
print("-----classify_no surfacing:1_flippers:0_result:%s-----"%result1)
result2 = classify(myTree,labels,[1,1])
print("-----classify_no surfacing:1_flippers:1_result:%s-----"%result2)

#由于python2和3的区别，这里相较于源码做出了部分改动
#也可以保存为pkl格式
storeTree(myTree,"classifierStorage.txt")
grabTree("classifierStorage.txt")