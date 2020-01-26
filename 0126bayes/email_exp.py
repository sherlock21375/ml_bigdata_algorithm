# -*- coding: utf-8 -*-
# @Time    : 2020/1/25 14:33
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : email_exp.py
# @Software: PyCharm
from numpy import *
from trainNb_test import *
def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W+', bigString)  #与源代码相比针对正则表达式做出了部分修改 *改为+，指的是出现1个及以上
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26):
        # print('email/spam/%d.txt' %i)
        wordList = textParse(open('email/spam/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # print('email/ham/%d.txt' % i)
        wordList = textParse(open('email/ham/%d.txt' %i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    trainingSet = list(range(50)); testSet=[]           #create test set  #此处trainingSet定义做了修改
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error",docList[docIndex])
    print('the error rate is: ',float(errorCount)/len(testSet))
    #return vocabList,fullText

# string2 = "Hi Peter,With Jose out of town, do you want to?"
# a = textParse(string2)
# print(a)

# spamTest()
# wa = 10
# print(textParse(open('email/spam/%d.txt'%wa).read()))
# print(textParse(open('email/ham/%d.txt'%wa).read()))