# -*- coding: utf-8 -*-
from math import log
import operator
from imp import reload # to enable reload in py3

def calcShannonEnt(dataSet):
    """
    函数说明:计算给定数据集的经验熵(香农熵)
    
    Parameters:
        dataSet - 数据集
    Returns:
        shannonEnt - 经验熵(香农熵)
    """
    #返回数据集的行数
    numEntries = len(dataSet)
    #保存每个标签(Label)出现次数的字典
    labelCounts = {}
    #对每组特征向量进行统计
    for featVec in dataSet:
        #提取标签(Label)信息，最后一列
        currentLabel = featVec[-1]
        #如果标签(Label)没有放入统计次数的字典,添加进去
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        #Label计数+1
        labelCounts[currentLabel] += 1 
        #print(labelCounts)
    #经验熵(香农熵)
    shannonEnt = 0.0
    #计算香农熵
    for key in labelCounts:
        #选择该标签(Label)的概率
        prob = float(labelCounts[key])/numEntries
        #利用公式计算
        shannonEnt -= prob * log(prob, 2)
        #print(shannonEnt)
    #返回经验熵(香农熵)
    return shannonEnt
    
def createDataSet():
    """
    函数说明:创建测试数据集
     
    Parameters:
        无
    Returns:
        dataSet - 数据集
        labels - 特征标签
    """
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
    
def createDataSet2():
        """
    函数说明:创建测试数据集
     
    Parameters:
        无
    Returns:
        dataSet - 数据集
        labels - 特征标签
    """
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['no_loan', 'loan']             #分类属性
    return dataSet, labels                #返回数据集和分类属性   

def createDataSet3():
        """
    函数说明:创建测试数据集
     
    Parameters:
        无
    Returns:
        dataSet - 数据集
        labels - 特征标签
    """
    dataSet = [[0, 0, 0, 0, 'no'],                        #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['Age', 'Job', 'House', 'Credit']             #分类属性
    return dataSet, labels                #返回数据集和分类属性   

def splitDataSet(dataSet, axis, value):
    """
    函数说明:按照给定特征划分数据集
    
    Parameters:
        dataSet - 待划分的数据集
        axis - 划分数据集的特征
        value - 需要返回的特征的值
    Returns:
        无
    """
    #创建返回的数据集列表
    retDataSet = []
    #遍历数据集
    for featVec in dataSet:
        if featVec[axis] == value:
            #去掉axis特征
            reducedFeatVec = featVec[:axis]
            #将符合条件的添加到返回的数据集
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet #返回划分后的数据集
    
def chooseBestFeature(dataSet):
    """
    函数说明:选择最优特征
    
    Parameters:
        dataSet - 数据集
    Returns:
        bestFeature - 信息增益最大的(最优)特征的索引值
    """
    #特征数量
    numFeatures = len(dataSet[0]) - 1
    #计算数据集的香农熵H(D)
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0 #信息增益
    bestFeature = -1 #最优特征的索引值
    for i in range(numFeatures): #遍历所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) #创建set集合{},元素不可重复
        newEntropy = 0.0 #经验条件熵H(D|A)
        for value in uniqueVals: #计算信息增益
            subDataSet = splitDataSet(dataSet, i, value) #subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet)) #计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) #根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy #信息增益
        print("Info gain of %d feature is %.3f" % (i, infoGain))#打印每个特征的信息增益
        if infoGain > bestInfoGain:#计算信息增益
            bestInfoGain = infoGain #更新信息增益，找到最大的信息增益
            bestFeature = i #记录信息增益最大的特征的索引值
    return bestFeature #返回信息增益最大的特征的索引值
    
def majorityCnt(classList):
    """
    函数说明:统计classList中出现此处最多的元素(类标签)
    
    Parameters:
    classList - 类标签列表
Returns:
    sortedClassCount[0][0] - 出现此处最多的元素(类标签)
    """
    classCount = {} # create a dict whose keys are unique values in classList
    for vote in classList: # takes a lit of class names
        if vote not in classCount.key():
            classCount[vote] = 0
        classCount[vote] += 1 # get frequency of occurrence of each class label from classList
    sortedClasscount = sorted(classCount.item(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # return the class that occurs with the greatest frequency

def createTree(dataSet, labels):
    """
    函数说明:创建决策树
     
    Parameters:
        dataSet - 训练数据集
        labels - 分类属性标签
        featLabels - 存储选择的最优特征标签
    Returns:
        myTree - 决策树
    """
    classList = [example[-1] for example in dataSet]
    #print(classList)
    if classList.count(classList[0]) == len(classList): # stop when all classes are equal
        #print("all classes are equal")
        return classList[0]
    if len(dataSet[0]) == 1: # when no more features, return majority
        #print("no more features")
        return majorityCnt(classList)
    #print("main")
    bestFeat = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeat]
    #featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
                           
    return myTree  
   