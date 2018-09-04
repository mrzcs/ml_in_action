# - *- coding: utf- 8 - *-
import numpy as np
import operator as op
#from imp import reload # to enable py3 reload module

def createDataSet():
    #create a dataset
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    #create labels
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0] 
    #inX: the input vector to classify 
    #dataSet: the full matrix of training samples
    #labels: a vector of labels
    #k: number of NN to use in the voting
    
    #距离度量 度量公式为欧氏距离 Euclidian distance
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #np.tile(A, reps): Construct an array by repeating A the number of times given by reps.
    sqDiffMat = diffMat ** 2 # square
    sqDistances = sqDiffMat.sum(axis=1) ## sum value in horizon 平方和
    distances = sqDistances**0.5 ## square root 开根号
    
    #将距离排序：从小到大
    sortedDistIndicies = distances.argsort()
    #选取前K个最短距离， 选取这K个中最多的分类类别
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key = op.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

