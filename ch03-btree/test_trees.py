# -*- coding: utf-8 -*-
import trees

dataMat, labels = trees.createDataSet2()
print(dataMat)
result = trees.calcShannonEnt(dataMat)
print("result is: %.2f" % result)