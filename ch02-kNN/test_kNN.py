# -*- coding: utf-8 -*-
import kNN
import matplotlib.pyplot as plt
import numpy as np

# #test case 1
# group, labels = kNN.createDataSet2() 
# #print(group)
# #print(labels)
# test = [18,90]
# kIN = 3
# test_result = kNN.classify0(test, group, labels, kIN)
# print("Test result: %s" % (test_result))

# test case 2
#datingDataMat, datingLabels = kNN.file2matrix('datingTestSet.txt')

#datingDataMat, datingLabels = kNN.file2matrix2('datingTestSet2.txt')
#kNN.showData(datingDataMat,datingLabels)

#print(datingDataMat)
#print(datingLabels)
#fig = plt.figure()
#ax = fig.add_subplot(111)
# show col 2 and 3 of matrix in the plot
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])

# ax.scatter(datingDataMat[:,1], datingDataMat[:,2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
#plt.show()

# test case 3
#datingDataMat, datingLabels = kNN.file2matrix2('datingTestSet2.txt')
#normDataMat, ranges, minVals = kNN.autoNorm(datingDataMat)
#print(normDataMat)
#print(ranges)
#print(minVals)

#test case 4
#kNN.datingClassTest()

#test case 5
#kNN.classifyPerson()

#test case 6
#kNN.handwritingClassTest()
kNN.handwritingClassTestSKL()
