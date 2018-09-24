# -*- coding: UTF-8 -*-
import numpy as np
import random

def loadDataSet():
    """
    函数说明:创建实验样本

    Parameters:
        无
    Returns:
        postingList - 实验样本切分的词条
        classVec - 类别标签向量
    """
    postingList = [
        ['my','dog','has','flea', \
        'problem','help','please'],
        ['maybe', 'not', 'take', 'him', \
        'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', \
        'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
        'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
        ]
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

def createVocabList(dataSet):
    """
    函数说明:将切分的实验样本词条整理成不重复的词条列表，也就是词汇表

    Parameters:
        dataSet - 整理的样本数据集
    Returns:
        vocabSet - 返回不重复的词条列表，也就是词汇表
    """
    vocabSet = set([]) #创建一个空的不重复列表
    for document in dataSet:
        vocabSet = vocabSet | set(document) # |: 取并集, union of two sets
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    函数说明:根据vocabList词汇表，将inputSet向量化，
    bag-of-words model, which increments the word vector rather than
    setting the word vector to 1 for a given index.
    Parameters:
        vocabList - createVocabList返回的列表，也就是词汇表
        inputSet - 切分的词条列表
    Returns:
        returnVec - 文档向量,词集模型
    """
    returnVec = [0] * len(vocabList)  #创建一个其中所含元素都为0的向量
    for word in inputSet: #遍历每个词条
        if word in vocabList: #如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary" % (word))
    return returnVec #返回文档向量

def bagOfWords2VecMN(vocabList, inputSet):
    """
    函数说明:根据vocabList词汇表，构建词袋模型

    Parameters:
        vocabList - createVocabList返回的列表
        inputSet - 切分的词条列表
    Returns:
        returnVec - 文档向量,词袋模型
    """
    returnVec = [0] * len(vocabList)  #创建一个其中所含元素都为0的向量
    for word in inputSet: #遍历每个词条
        if word in vocabList: #如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
    return returnVec #返回词袋模型

def trainNB0(trainMatrix, trainCategory):
    """
    函数说明:朴素贝叶斯分类器训练函数

    Parameters:
        trainMatrix - 训练文档矩阵，即setOfWords2Vec返回的returnVec构成的矩阵
        trainCategory - 训练类别标签向量，即loadDataSet返回的classVec
    Returns:
        p0Vect - 非侮辱类的条件概率数组
        p1Vect - 侮辱类的条件概率数组
        pAbusive - 文档属于侮辱类的概率
    """
    numTrainDocs = len(trainMatrix) #计算训练的文档数目 6
    numWords = len(trainMatrix[0]) #计算每篇文档的词条数 32
    pAbusive = sum(trainCategory)/float(numTrainDocs) #文档属于侮辱类的概率 3/6=0.5
    #numerator
    #p0Num = np.zeros(numWords) #创建numpy.zeros数组,词条出现数初始化为0
    #p1Num = np.zeros(numWords)
    p0Num = np.ones(numWords) # Laplace Smoothing 拉普拉斯平滑
    p1Num = np.ones(numWords)
    #denominator
    #p0Denom = 0.0  #分母初始化为0
    #p1Denom = 0.0
    p0Denom = 2.0 # Laplace Smoothing
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else: #统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #p1Vect = p1Num/p1Denom
    #P0Vect = p0Num/p0Denom
    p1Vect = np.log(p1Num/p1Denom) # 取对数，防止下溢出 underflow 由于太多很小的数相乘造成
    P0Vect = np.log(p0Num/p0Denom)
    return P0Vect, p1Vect, pAbusive #返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    Parameters:
        vec2Classify - 待分类的词条数组
        p0Vec - 非侮辱类的条件概率数组
        p1Vec -侮辱类的条件概率数组
        pClass1 - 文档属于侮辱类的概率
    Returns:
        0 - 属于非侮辱类
        1 - 属于侮辱类
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1) #对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listPosts, listClasses = loadDataSet() #创建实验样本
    myVocabList = createVocabList(listPosts) #创建词汇表
    trainMat=[]
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc)) #将实验样本向量化
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses)) #训练朴素贝叶斯分类器
    testEntry = "love my dalmation".split() #测试样本1
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry)) #测试样本向量化
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb)) #执行分类并打印分类结果
    testEntry = "stupid garbage".split() #测试样本2
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    """
    函数说明:接收一个大字符串并将其解析为字符串列表

    Parameters:
        无
    Returns:
        无
    """
    import re
    listOfTokens = re.split(r'\W+', bigString) #将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    return [tok.lower() for tok in listOfTokens if len(tok) >2 ]  #去除单个字母,space，大写变成小写

def spamTest():
    """
    函数说明:测试朴素贝叶斯分类器

    Parameters:
        无
    Returns:
        无
    """
    docList = []; classList = []; fullText = []
    for i in range(1, 26): #遍历25个txt文件
        wordList = textParse(open('ch04-Bayes/email/spam/%d.txt' % i, 'r', encoding='utf-8', errors='ignore').read()) #读取每个垃圾邮件，字符串转换成字符串列表
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #标记垃圾邮件，1表示垃圾文件
        wordList = textParse(open('ch04-Bayes/email/ham/%d.txt' % i, 'r', encoding='utf-8', errors='ignore').read())  #读取每个非垃圾邮件，字符串转换成字符串列表
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #标记非垃圾邮件，1表示垃圾文件
    vocabList = createVocabList(docList) #创建词汇表，不重复
    #print('\n vocab List: ', vocabList)

    trainingSet = list(range(50)); testSet = []  #创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(10):  #从50个邮件中，随机挑选出40个作为训练集,10个做测试集
        randIndex = int(random.uniform(0, len(trainingSet))) #随机选取索索引值
        testSet.append(trainingSet[randIndex]) #添加测试集的索引值
        del(trainingSet[randIndex]) #在训练集列表中删除添加到测试集的索引值
    #print('\n test set: ', testSet)

    trainMat = []; trainClasses = [] #创建训练集矩阵和训练集类别标签系向量
    for docIndex in trainingSet: #遍历训练集
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))  #将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex]) #将类别添加到训练集类别标签系向量中
    #print('\n train matrix: ', trainMat)

    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses)) #训练朴素贝叶斯模型
    #print('p0V:', p0V)
    #print('p1V:', p1V)
    #print('pSpam:', pSpam)

    errorCount = 0  #错误分类计数
    for docIndex in testSet: #遍历测试集
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])  #测试集的词集模型
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]: #如果分类错误, 错误计数加1
            errorCount += 1
            print('classification error: ', docList[docIndex])
    print('the error rate is: ', float(errorCount)/len(testSet))

def stopWordsOld():
    wordList = textParse(open('ch04-Bayes/stopword.txt').read())
    return wordList
    """import re
    wordList =  open('ch04-Bayes/stopword.txt').read()
    listOfTokens = re.split(r'\W+', wordList)
    return [tok.lower() for tok in listOfTokens]
    print('read stop word from \'stopword.txt\':',listOfTokens)
    return listOfTokens """


def calcMostFreq(vocabList, fullText): #从fullText中找出最高频的前30个单词
    import operator
    freqDict = {}
    for token in vocabList: #统计词汇表里所有单词的出现次数
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30] ##返回字典

def localWords(feed1, feed0): #两份RSS文件分别经feedparser解析，得到2个字典
    print('feed1 entries length: ', len(feed1['entries']), '\nfeed0 entries length: ', len(feed0['entries']))
    #entries条目包含多个帖子，miNLen记录帖子数少的数目，怕越界
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    print('\nmin Length: ', minLen)

    docList = [] #一条条帖子组成的List, 帖子拆成了单词
    classList = [] #标签列表
    fullText = [] #所有帖子的所有单词组成的List

    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])#取出帖子内容，并拆成词
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #SF

    vocabList = createVocabList(docList) #创建词汇表
    print('\nVocabList is ',len(vocabList))

    # Removes stop words
    print('\nRemoving stop words')
    stopWordList = stopWords('ch04-Bayes/stopword.txt')
    #stopWordList = stopWordsOld()
    for stopWord in set(stopWordList):
        if stopWord in vocabList:
            vocabList.remove(stopWord)

    # Removes most frequently occurring words
    # 从fulltext中找出最高频的30个单词，并从vocabList中去除它们
    """ top30Words = calcMostFreq(vocabList, fullText)
    for (word,count) in top30Words:
        if word in vocabList:
            vocabList.remove(word) """

    print('\nremaining: ',len(vocabList))

    trainingSet = list(range(2*minLen))
    testSet=[]
    testCount = int(2*minLen*0.2)
    print('total count of test: ', testCount)
    for i in range(testCount): #随机选取20%的数据，建立测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) #将训练集中的每一条数据，转化为词向量
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses)) #开始训练

    # 用测试数据，测试分类器的准确性
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        classifiedClass = classifyNB(np.array(wordVector), p0V, p1V, pSpam)
        originalClass = classList[docIndex]
        if classifiedClass != originalClass:
            errorCount += 1
            #print('classification error: ', docList[docIndex])
            print('\n',docList[docIndex],'\nis classified as: ',classifiedClass,', while the original class is: ',originalClass)
    print('the error rate is: ', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny,sf):
    import operator
    vocabList, p0V, p1V = localWords(ny,sf)
    topNY = [];topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY **')
    for item in sortedNY:
        print(item[0])


def textProcessing(folderPath, testSize = 0.2):
    """
    函数说明:中文文本处理

    Parameters:
        folderPath - 文本存放的路径
        testSize - 测试集占比，默认占所有数据集的百分之20
    Returns:
        allWordList - 按词频降序排序的训练集列表
        trainDataList - 训练集列表
        testDataList - 测试集列表
        trainClassList - 训练集标签列表
        testClassList - 测试集标签列表
    """
    import os
    import jieba

    folderList = os.listdir(folderPath) #查看folderList下的文件
    dataList =[] #数据集数据
    classList = [] #数据集类别

    for folder in folderList: #遍历每个子文件夹
        newFolderPath = os.path.join(folderPath, folder) #根据子文件夹，生成新的路径
        files = os.listdir(newFolderPath) #存放子文件夹下的txt文件的列表

        j = 1
        for file in files: #遍历每个txt文件
            if j > 100: #每类txt样本数最多100个
                break
            with open(os.path.join(newFolderPath, file), 'r', encoding='utf-8') as f: #打开txt文件
                raw = f.read()
            wordCut = jieba.cut(raw, cut_all=False) #精简模式，返回一个可迭代的generator
            wordList = list(wordCut) #generator转换为list

            dataList.append(wordList) #添加数据集数据
            classList.append(folder) #添加数据集类别
            j += 1
        #print(dataList)
        #print(classList)
    dataClassList = list(zip(dataList, classList)) #zip压缩合并，将数据与标签对应压缩
    random.shuffle(dataClassList)  # 将dataClassList乱序
    index = int(len(dataClassList) * testSize) + 1 #训练集和测试集切分的索引值
    trainList = dataClassList[index:] #训练集
    testList = dataClassList[:index] #测试集
    trainDataList, trainClassList = zip(*trainList) #训练集解压缩
    testDataList, testClassList = zip(*testList) #测试集解压缩

    allWordDict = {} #统计训练集词频
    for wordList in trainDataList:
        for word in wordList:
            if word in allWordDict.keys():
                allWordDict[word] += 1
            else:
                allWordDict[word] = 1
    #根据键的值倒序排序
    allWordTuple = sorted(allWordDict.items(), key=lambda f:f[1], reverse =True)
    allWordList, allWordNums = zip(*allWordTuple) #解压缩
    allWordList = list(allWordList) #转换成列表
    return allWordList, trainDataList, testDataList, trainClassList, testClassList

def stopWords(wordFile):
    """
    函数说明:读取文件里的内容，并去重

    Parameters:
        wordsFile - 文件路径
    Returns:
        wordsSet - 读取的内容的set集合
    """
    wordSet = set()
    with open(wordFile, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                wordSet.add(word)
    return wordSet

def wordDict(allWordList, deleteN, stopWordSet =set()):
    """
    函数说明:文本特征选取

    Parameters:
        all_wordsallWordList_list - 训练集所有文本列表
        deleteN - 删除词频最高的deleteN个词
        stopWordSet - 指定的结束语
    Returns:
        featureWord - 特征集
    """
    featureWord = [] #特征列表
    n = 1
    for t in range(deleteN, len(allWordList), 1):
        if n > 1000: #feature_words的维度为1000
            break
        #如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not allWordList[t].isdigit() and allWordList[t] not in stopWordSet and 1 < len(allWordList[t]) < 5:
            featureWord.append(allWordList[t])
        n += 1
    return featureWord

def textFeatures(trainDataList, testDataList, featureWord):
    def textFeature(text, featureWord):
        textWords = set(text)
        features = [1 if word in textWords else 0 for word in featureWord]
        return features
    trainFeatureList = [textFeature(text, featureWord) for text in trainDataList]
    testFeatureList = [textFeature(text, featureWord) for text in testDataList]
    return trainFeatureList, testFeatureList


def textClassifier(trainFeatureList, testFeatureList, trainClassList, testClassList):
    from sklearn.naive_bayes import MultinomialNB
    classifier = MultinomialNB().fit(trainFeatureList, trainClassList)
    testAccuracy = classifier.score(testFeatureList, testClassList)
    return testAccuracy

def testRSS():
    import feedparser
    ny=feedparser.parse('https://newyork.craigslist.org/search/res?format=rss')
    sf=feedparser.parse('https://sfbay.craigslist.org/search/res?format=rss')
    #ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
    #sf=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
    # 构建的这个分类器的作用是给出一条帖子，判断（猜测）它是来自那个地区的。New York是1，San Francisco是0.
    vocabList,pSF,pNY = localWords(ny,sf)

def testTopWords():
    import feedparser
    ny=feedparser.parse('https://newyork.craigslist.org/search/res?format=rss')
    sf=feedparser.parse('https://sfbay.craigslist.org/search/res?format=rss')
    getTopWords(ny,sf)

def testCN():
    folderPath = 'ch04-Bayes/SogouC/Sample'
    #textProcessing(folderPath)
    allWordList, trainDataList, testDataList, trainClassList, testClassList = textProcessing(folderPath, testSize=0.2)
    #print(allWordList)

    stopWordFile = 'ch04-Bayes/stopwords_cn.txt'
    stopWordSet = stopWords(stopWordFile)
    #print(stopWordSet)
    #featureWord = wordDict(allWordList, 100, stopWordSet)
    #print(featureWord)

    testAccuracyList = []

    """ #test deleteN value
    deleteNs = range(0, 1000, 20)

    for deleteN in deleteNs:
        featureWord = wordDict(allWordList, deleteN, stopWordSet)
        trainFeatureList, testFeatureList = textFeatures(trainDataList, testDataList, featureWord)
        testAccuracy = textClassifier(trainFeatureList, testFeatureList, trainClassList, testClassList)
        testAccuracyList.append(testAccuracy)
    #print(testAccuracyList)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(deleteNs, testAccuracyList)
    plt.title('Relationshit of deleteNs and testAccuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('testAccuracy')
    plt.show() """

    featureWord = wordDict(allWordList, 500, stopWordSet)
    trainFeatureList, testFeatureList = textFeatures(trainDataList, testDataList, featureWord)
    testAccuracy = textClassifier(trainFeatureList, testFeatureList, trainClassList, testClassList)
    testAccuracyList.append(testAccuracy)
    ave = lambda c: sum(c) / len(c)
    print(ave(testAccuracyList))


if __name__ == '__main__':
    """docList = []; classList=[]
    for i in range(1, 26):
        wordList = textParse(open('ch04-Bayes/email/spam/%d.txt' % i, 'r', encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('ch04-Bayes/email/ham/%d.txt' % i, 'r', encoding='utf-8', errors='ignore').read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    print(vocabList) """

    #spamTest()


    #testRSS()
    #testTopWords()
    testCN()