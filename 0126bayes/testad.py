# -*- coding: utf-8 -*-
# @Time    : 2020/1/25 20:06
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : testad.py
# @Software: PyCharm
import feedparser
from email_exp import *
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)  #此处修改python3中已经没有 “iteritems” 这个属性了，现在属性是：“ items ”
    return sortedFreq[:30]
def stopWords():
    import re
    wordList =  open('stopword.txt').read() # see http://www.ranks.nl/stopwords
    listOfTokens = re.split(r'\W*', wordList)
    return [tok.lower() for tok in listOfTokens]
    print('read stop word from \'stopword.txt\':',listOfTokens)
    return listOfTokens

def localWords(feed1,feed0):
    import feedparser
    docList=[]; classList = []; fullText =[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words
    #采取stopword时的处理措施
    # stopWordList = stopWords()
    # for stopWord in stopWordList:
    #     if stopWord in vocabList:
    #         vocabList.remove(stopWord)
    #         print('Removed: ', stopWord)
    # for pairW in top30Words:
    #     if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2*minLen); testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(list(trainingSet)[randIndex])
    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:   #train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    errorCount = 0
    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ',float(errorCount)/len(testSet))
    return vocabList,p0V,p1V

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V=localWords(ny,sf)
    topNY=[]; topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])

ny = feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf = feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
# print(ny['entries'])
getTopWords(ny,sf)