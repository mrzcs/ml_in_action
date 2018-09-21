import bayes

if __name__ == '__main__':
    postingList, classVec = bayes.loadDataSet()
    print('postingList:\n')
    for each in postingList:
        print(each)
    print(classVec)

    myVocabList = bayes.createVocabList(postingList)
    print('myVocabList:\n',myVocabList)

    trainMat = []
    for postinDoc in postingList:
        trainMat.append(bayes.setOfWords2Vec(myVocabList, postinDoc))
    print('trainMat:\n', trainMat)

    p0V, p1V, pAb = bayes.trainNB0(trainMat, classVec)
    print('p0V:\n', p0V)
    print('p1V:\n', p1V)
    print('classVec:\n', classVec)
    print('pAb:\n', pAb)

    bayes.testingNB()