欢迎查看我的博客，转载请注明：原创于http://blog.juneday.site/  作者：sherlock june
------------------------------------------
1.首先这里一个完整的代码，关于如何构建数据集，香农熵的计算，选择划分数据集的最佳特征值
------------------------------------------

    # -*- coding: utf-8 -*-
    # @Time    : 2020/1/21 12:23
    # @Author  : Sherlock June
    # @Email   : wangjun980213@163.com
    # @File    : countShannon.py
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
    
    
    myDat,labels = createDataSet()
    print("----------output-myDat----------")
    print(myDat)
    print("----------output-labels---------")
    print(labels)
    shannonEnt = calcShannonEnt(myDat)
    print("----------output-shannonEnt-----")
    print(shannonEnt)
    splitDataResult = splitDataSet(myDat,0,1)
    print("----------output-splitDataResult-by-0-1----------")
    print(splitDataResult)
    splitDataResult = splitDataSet(myDat,0,0)
    print("----------output-splitDataResult-by-0-0----------")
    print(splitDataResult)
    bestFeature = chooseBestFeatureToSplit(myDat)
    print("----------output-bestFeature----------")
    print(bestFeature)

![012101.PNG][1]

![012102.PNG][2]

2.下面构造决策树
-------

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
        bestFeatLabel = labels[bestFeat]
        myTree = {bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues = [example[bestFeat] for example in dataSet]  #提取最佳特征值组合
        # print("----------featValues----------")
        # print(featValues)
        uniqueVals = set(featValues)  #提取最佳特征值的unique，构造字典
        # print("----------uniqueVals----------")
        # print(uniqueVals)
        for value in uniqueVals:
            subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
            myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
        return myTree
    
    myDat,labels = createDataSet()
    print("----------output-myDat----------")
    print(myDat)
    myTree = createTree(myDat,labels)
    print("----------this is my tree----------")
    print(myTree)

![012103.PNG][3]

3.使用matplotlib库将生成树构造成可视化图形
---------------------------

    # -*- coding: utf-8 -*-
    # @Time    : 2020/1/21 16:23
    # @Author  : Sherlock June
    # @Email   : wangjun980213@163.com
    # @File    : visualization_tree.py
    # @Software: PyCharm
    from make_tree import *
    import matplotlib.pyplot as plt
    
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    leafNode = dict(boxstyle="round4", fc="0.8")
    arrow_args = dict(arrowstyle="<-")
    
    def getNumLeafs(myTree):
        numLeafs = 0
        firstStr = list(myTree.keys())[0]    #注意此处和书中代码不一致（下方类似），增加了将myTree.keys()转化为list,这是因为在
        # python2.x dict.keys 返回一个列表，但是在 Python 3.x 下，dict.keys 返回的是 dict_keys 对象，若需要转换为列表，
        # 请使用：list(dict.values()) list(dict.keys())
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
                numLeafs += getNumLeafs(secondDict[key])
            else:   numLeafs +=1
        return numLeafs
    
    def getTreeDepth(myTree):
        maxDepth = 0
        firstStr = list(myTree.keys())[0]
        secondDict = myTree[firstStr]
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
                thisDepth = 1 + getTreeDepth(secondDict[key])
            else:   thisDepth = 1
            if thisDepth > maxDepth: maxDepth = thisDepth
        return maxDepth
    
    def retrieveTree(i):
        listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                      {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                      ]
        return listOfTrees[i]
    
    def plotNode(nodeTxt, centerPt, parentPt, nodeType):
        createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
                 xytext=centerPt, textcoords='axes fraction',
                 va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
    
    def plotMidText(cntrPt, parentPt, txtString):
        xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
        yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
        createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
    
    def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
        numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
        depth = getTreeDepth(myTree)
        firstStr = list(myTree.keys())[0]     #the text label for this node should be this
        cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
        plotMidText(cntrPt, parentPt, nodeTxt)
        plotNode(firstStr, cntrPt, parentPt, decisionNode)
        secondDict = myTree[firstStr]
        plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
        for key in secondDict.keys():
            if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
                plotTree(secondDict[key],cntrPt,str(key))        #recursion
            else:   #it's a leaf node print the leaf node
                plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
                plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
                plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
        plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    #if you do get a dictonary you know it's a tree, and the first element will be another dict
    
    def createPlot(inTree):
        fig = plt.figure(1, facecolor='white')
        fig.clf() #Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
        axprops = dict(xticks=[], yticks=[])
        createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    #no ticks  #不含坐标系
        #createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
        plotTree.totalW = float(getNumLeafs(inTree))
        plotTree.totalD = float(getTreeDepth(inTree))
        plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
        plotTree(inTree, (0.5,1.0), '')
        plt.show()
    
    # def createPlot():
    #    fig = plt.figure(1, facecolor='white')
    #    fig.clf()
    #    createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    #    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    #    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    #    plt.show()
    
    myTree2 = retrieveTree(0)
    # myTree2 = retrieveTree(1)
    # createPlot()  #测试简单的构造
    print("----------this is my tree2 loaded for test----------")
    print(myTree2)
    num_leaves = getNumLeafs(myTree2)
    print("----------myTree2-num_leaves----------")
    print(num_leaves)
    tree_depth = getTreeDepth(myTree2)
    print("----------myTree2-tree_depth----------")
    print(tree_depth)
    createPlot(myTree2)
![012105.png][4]

![012106.PNG][5]

4.使用具体数据进行分类
------------

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

![012107.PNG][6]

5.来个实例吧，生成隐形眼镜的决策树
------------------

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

![012201.PNG][7]


  [1]: https://www.juneday.site/blog/usr/uploads/2020/01/2015104714.png
  [2]: https://www.juneday.site/blog/usr/uploads/2020/01/325143944.png
  [3]: https://www.juneday.site/blog/usr/uploads/2020/01/3233185094.png
  [4]: https://www.juneday.site/blog/usr/uploads/2020/01/180914163.png
  [5]: https://www.juneday.site/blog/usr/uploads/2020/01/3343913577.png
  [6]: https://www.juneday.site/blog/usr/uploads/2020/01/2267598776.png
  [7]: https://www.juneday.site/blog/usr/uploads/2020/01/851374375.png
