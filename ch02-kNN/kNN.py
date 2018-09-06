# -*- coding: utf-8 -*-
import numpy as np
import operator as op
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.font_manager import FontProperties
from imp import reload # to enable reload in py3
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as knc

def createDataSet():
    #create a dataset
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    #create labels
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def createDataSet2():
    #create a dataset
    group = np.array([[3, 104], [2, 100], [1, 81], [101, 10], [99, 5], [98, 2]])
    #create labels
    labels = ['Romance', 'Romance', 'Romance', 'Action', 'Action', 'Action']
    return group, labels
    
def classify0(inX, dataSet, labels, k):
    """
    Parameters:
        inX: the input vector to classify 
        dataSet: the full matrix of training samples
        labels: a vector of labels
        k: number of NN to use in the voting
    
    Retruns: 
        sortedClassCount[0][0]: result of classification
    """
    
    #np.shape[0]: 返回dataSet的行数
    dataSetSize = dataSet.shape[0] 
    
    #距离度量 度量公式为欧氏距离 Euclidian distance
    #在列向量方向上重复inX共1次(横向)，行向量方向上重复inX共dataSetSize次(纵向)
    #np.tile(A, reps): Construct an array by repeating A the number of times given by reps.
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat ** 2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    #sum(axis=1)：sum value in horizon
    sqDistances = sqDiffMat.sum(axis=1) 
    #square root 开方，计算出距离
    distances = sqDistances**0.5 
    #返回distances中元素从小到大排序后的索引值
    #argsort返回索引值，从0开始
    sortedDistIndicies = distances.argsort() 

    #选取前K个最短距离， 选取这K个中最多的分类类别
    
    #定义记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        
        #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #py2
    #sortedClassCount = sorted(classCount.iteritems(), key = op.itemgetter(1), reverse=True)
    
    #py3中用 items()替换python2中的 iteritems()
    #key=operator.itemgetter(1)根据字典的值进行排序
    #key=operator.itemgetter(0)根据字典的键进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key = op.itemgetter(1), reverse=True)
    
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

def file2matrix2(filename): 
    """
    to handle datingTestSet2.txt
    Parameters:
        filename - 文件名
    Returns:
        returnMat - 特征矩阵
        classLabelVector - 分类Label向量
    """
    #open filename
    fr = open(filename)
    arrayLines = fr.readlines()
    #get num of line
    numberOfLines = len(arrayLines)
    #generate all 0 matrix, numberOfLines x 3
    #返回的NumPy矩阵,解析完成的数据: matrix of numberOfLines行 x 3列
    returnMat = np.zeros((numberOfLines, 3))
    #返回的分类标签向量
    classLabelVector = [] 
    #行的索引值
    index = 0
    for line in arrayLines:
        line = line.strip() #删除空白符(包括'\n','\r','\t',' ')
        #split line with tab
        listFromLine = line.split('\t')
        #将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        #根据文本中标记的喜欢的程度进行分类,1代表不喜欢,2代表魅力一般,3代表极具魅力
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector
    
    
def file2matrix(filename): 
    """
    to handle datingTestSet.txt
    Parameters:
        filename - 文件名
    Returns:
        returnMat - 特征矩阵
        classLabelVector - 分类Label向量
    """
    fr = open(filename)
    arrayLines = fr.readlines()
    numberOfLines = len(arrayLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = [] 
    index = 0
    labels = {'didntLike':1,'smallDoses':2,'largeDoses':3}
    for line in arrayLines:
        line = line.strip()
        #split line with tab
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(labels[listFromLine[-1]])
        index += 1
    return returnMat, classLabelVector    

def showData(dataMat, labels):
    """
    Parameters:
        dataMat - 特征矩阵
        labels - 分类Label
    Returns:
        无
    """
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows = 2, ncols =2, sharex = False, sharey = False, figsize = (13, 8))
    labelNum = len(dataMat)
    labelColor = []
    for i in labels:
        if i == 1:
            labelColor.append('black')
        elif i == 2:
            labelColor.append('orange')
        elif i == 3:
            labelColor.append('red')
    """
    'Number of frequent flyer miles earned per year' 1st col
    'Percentage of time spent playing video games' 2nd col
    'Liters of ice cream consumed weekly' 3rd col
    """        
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=dataMat[:,0], y=dataMat[:,1], color=labelColor, s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title('flyer v.s. game')
    axs0_xlabel_text = axs[0][0].set_xlabel('Number of frequent flyer miles earned per year') 
    axs0_ylabel_text = axs[0][0].set_ylabel('Percentage of time spent playing video games') 
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=dataMat[:,0], y=dataMat[:,2], color=labelColor, s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title('flyer v.s. icecream')
    axs1_xlabel_text = axs[0][1].set_xlabel('Number of frequent flyer miles earned per year') 
    axs1_ylabel_text = axs[0][1].set_ylabel('Liters of ice cream consumed weekly') 
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')
    
    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=dataMat[:,1], y=dataMat[:,2], color=labelColor, s=15, alpha=.5)
    axs2_title_text = axs[1][0].set_title('game v.s. icecream')
    axs2_xlabel_text = axs[1][0].set_xlabel('Percentage of time spent playing video games') 
    axs2_ylabel_text = axs[1][0].set_ylabel('Liters of ice cream consumed weekly') 
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    
    #设置图例
    didntLike = mlines.Line2D([],[],color='black',marker='.',markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([],[],color='orange',marker='.',markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([],[],color='red',marker='.',markersize=6, label='largeDoses')
    
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses]) 
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    
    #显示图片
    plt.show()
    
def autoNorm(dataSet):
    """
    Parameters:
        dataSet - 特征矩阵
    Returns:
        normDataSet - 归一化后的特征矩阵
        ranges - 数据范围
        minVals - 数据最小值
    """
    #newValue = (oldValue-min)/(max-min)
     #获得数据的最大/小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值和最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数，生成空矩阵
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m,1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    #返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals

def datingClassTest():
    #取所有数据的百分之十
    hoRatio = 0.10 # pct of test data
    filename = "datingTestSet2.txt"
    datingDataMat, datingLabels = file2matrix2(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获得normMat的行数
    m = normMat.shape[0]
    #测试数据的个数
    numTestVecs = int(m*hoRatio) 
    #分类错误计数
    errorCount = 0.0
    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        #(inX, dataSet, labels, k)
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],4)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %100.2f%%" % (errorCount/float(numTestVecs)*100))
    
def classifyPerson():
    #输出结果
    resultList = ['not at all', 'in small doses', 'in large doses']
    #三维特征用户输入
    percentTats = float(input("pct of the time spent playing video games? "))
    ffMiles = float(input("frequent flier miles earned per year? "))
    iceCream = float(input("liters of ice cream consumed per weekly? "))
    #打开并处理数据
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    #训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #生成NumPy数组,测试集
    inArr = np.array([ffMiles, percentTats, iceCream])
    #测试集归一化
    norminArr = (inArr - minVals) / ranges
    #返回分类结果
    classifierResult =  classify0(norminArr, normMat, datingLabels, 3)
    #打印结果
    print("You will probably like this person: %s" % resultList[classifierResult - 1])
    
def img2vector(filename):
    """
    Parameters:
        filename - 文件名
    Returns:
        returnVect - 返回的二进制图像的1x1024向量
    """
    #创建1x1024零向量
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        #每一行的前32个元素依次添加到returnVect中
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    #返回转换后的1x1024向量
    return returnVect
    
def handwritingClassTest():
    #测试集的Labels
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = listdir("trainingDigits")
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m,1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #fileStr = fileNameStr.split('.')[0]
        #获得分类的数字
        classNumStr = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumStr)
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector("trainingDigits/%s" % fileNameStr)
        
    #返回testDigits目录下的文件列表
    testFileList = listdir("testDigits")
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #fileStr = fileNameStr.split('.')[0]
        #获得分类的数字
        classNumStr = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)
        #获得预测结果
        classifierResult =  classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n the total number of errors is: %d" % errorCount)
    print("\n the total error rate is: %100.2f%%" % (errorCount/float(mTest)*100))
    
def handwritingClassTestSKL():
    #测试集的Labels
    hwLabels = []
    #返回trainingDigits目录下的文件名
    trainingFileList = listdir("trainingDigits")
    #返回文件夹下文件的个数
    m = len(trainingFileList)
    #初始化训练的Mat矩阵,测试集
    trainingMat = np.zeros((m,1024))
    #从文件名中解析出训练集的类别
    for i in range(m):
        #获得文件的名字
        fileNameStr = trainingFileList[i]
        #fileStr = fileNameStr.split('.')[0]
        #获得分类的数字
        classNumStr = int(fileNameStr.split('_')[0])
        #将获得的类别添加到hwLabels中
        hwLabels.append(classNumStr)
        #将每一个文件的1x1024数据存储到trainingMat矩阵中
        trainingMat[i,:] = img2vector("trainingDigits/%s" % fileNameStr)
    
    """
    SK-learn method
    """
    #构建kNN分类器
    neigh =knc(n_neighbors=3, algorithm='auto')
    #拟合模型, trainingMat为测试矩阵,hwLabels为对应的标签
    neigh.fit(trainingMat, hwLabels)
    
    #返回testDigits目录下的文件列表
    testFileList = listdir("testDigits")
    #错误检测计数
    errorCount = 0.0
    #测试数据的数量
    mTest = len(testFileList)
    #从文件中解析出测试集的类别并进行分类测试
    for i in range(mTest):
        #获得文件的名字
        fileNameStr = testFileList[i]
        #fileStr = fileNameStr.split('.')[0]
        #获得分类的数字
        classNumStr = int(fileNameStr.split('_')[0])
        #获得测试集的1x1024向量,用于训练
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)
        #获得预测结果
        #classifierResult =  classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        classifierResult = neigh.predict(vectorUnderTest)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n the total number of errors is: %d" % errorCount)
    print("\n the total error rate is: %100.2f%%" % (errorCount/float(mTest)*100))