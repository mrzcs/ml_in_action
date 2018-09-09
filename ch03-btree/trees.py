# -*- coding: utf-8 -*-
from math import log

def calcShannonEnt(dataSet):
    """
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
            #Label计数
            labelCounts[currentLabel] += 1
    #经验熵(香农熵)
    shannonEnt = 0.0
    #计算香农熵
    for key in labelCounts:
        #选择该标签(Label)的概率
        prob = float(labelCounts[key])/numEntries
        #利用公式计算
        shannonEnt -= prob * log(prob, 2)
    #返回经验熵(香农熵)
    return shannonEnt
    
def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 0, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
    
def createDataSet2():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
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

    