# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 20:55:23 2018

@author: kcsz
"""
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8") #设置结点格式
leafNode = dict(boxstyle="round4", fc="0.8")  #设置叶结点格式
arrow_args = dict(arrowstyle="<-") #定义箭头格式
    
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, #绘制结点
                    xy=parentPt, 
                    xycoords='axes fraction', 
                    xytext=centerPt, 
                    textcoords='axes fraction',
                    va="center", 
                    ha="center", 
                    bbox=nodeType, 
                    arrowprops=arrow_args)
    
def createPlot1():
    fig = plt.figure(1, facecolor='white') #创建fig
    fig.clf() #清空fig
    createPlot.ax1 = plt.subplot(111, frameon=False) #去掉x、y轴
    # plotNode(nodeTxt, centerPt, parentPt, nodeType)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)  #绘制 decision node 
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode) #绘制 leaf node
    plt.show()   #显示绘制结果 

def getNumLeafs(myTree):
    """
    函数说明:获取决策树叶子结点的数目
     
    Parameters:
        myTree - 决策树
    Returns:
        numLeafs - 决策树的叶子结点的数目
    """
    numLeafs = 0  #初始化叶子
    #firstStr = next(iter(myTree)) #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = list(myTree.keys())[0]
    #print("first: %s" % firstStr) #no surfacing
    secondDict = myTree[firstStr] #获取下一组字典 
    #print("second: %s" % secondDict) ## {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}
    #print(secondDict.keys())
    for key in secondDict.keys():
        #print(type(secondDict[key]).__name__)
        if type(secondDict[key]).__name__ == 'dict': #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            #print(getNumLeafs(secondDict[key]))
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs +=1 
        #print("numLeafs: %d" % numLeafs)
    return numLeafs

def getTreeDepth(myTree):
    """
    函数说明:获取决策树的层数
     
    Parameters:
        myTree - 决策树
    Returns:
        maxDepth - 决策树的层数
    """
    maxDepth = 0 #初始化决策树深度
    #firstStr = next(iter(myTree)) #python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    firstStr = list(myTree.keys())[0]
    #print("first: %s" % firstStr)
    secondDict = myTree[firstStr] #获取下一个字典
    #print("second: %s" % secondDict)
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict': #测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        #print(thisDepth)
        if thisDepth > maxDepth:  #更新层数
            maxDepth = thisDepth
    return maxDepth

def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    """
    函数说明:标注有向边属性值 plots text between child and parent
     
    Parameters:
        cntrPt、parentPt - 用于计算标注位置
        txtString - 标注的内容
    Returns:
        无
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0] #计算标注位置
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
    """
    函数说明:绘制决策树
    
    Parameters:
        myTree - 决策树(字典)
        parentPt - 标注的内容
        nodeTxt - 结点名
    Returns:
        无
    """
    #decisionNode = dict(boxstyle="sawtooth", fc="0.8") #设置结点格式
    #leafNode = dict(boxstyle="round4", fc="0.8") #设置叶结点格式
    numLeafs = getNumLeafs(myTree) #获取决策树叶结点数目，决定了树的宽度
    #depth = getTreeDepth(myTree) #获取决策树层数
    getTreeDepth(myTree) 
    #firstStr = next(iter(myTree))
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff) #中心位置
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree. yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
    
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW =  float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()