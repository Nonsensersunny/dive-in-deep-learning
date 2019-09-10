# Deep Learning in Action

## *Preface*

> Steps of deep learning

1. Collecting data
2. Data inputing
3. Analyzing data
4. Training algorithm
5. Testing algorithm
6. Applying algorithm

> Basics of *NumPy*

```python
from numpy import *
random.rand(4, 4)	# random 2-dimension array
random.rand(4, 4, 2)	# random 3-dimension array
randomMt = mat(random.rand(4, 4))	# random 4*4 matrix
matInv = randomMt.I	# inversion of a matrix
randomEye = randomMt*matInv		# multiplication of matrice
randomEye - eye(4)	# error of eye
```

> FAQs of *NumPy*: [Click here for more info.](https://www.cnblogs.com/ningskyer/articles/7607457.html)

> Environment: Python 3.7.4

 ## 1. k-Nearest Neighbor

> Classify through distance between different eigan values.

```python
from numpy import *
import operator

# data set
group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
labels = ['A', 'A', 'B', 'B']

# kNN
# inX: data to be classified
# dataSet: training data set
# labels: label vectors
# k: nearest neighbor amount
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]	# `shape` returns rows and columns of a matrix
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet	# `tile(matrix, (a, b))` returns a new matrix by copy with rows by `a` time(s) and columns by `b` time(s)
    sqDiffMat = diffMat**2	# each number squares
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5 # each number roots
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

## 2. Decision Tree

> Shannon Entropy: $H=-\sum^n_{i=0}p(x_i)\log_2p(x_i)$

```python
from math import log
import operator
import matplotlib.pyplot as plt

# calculate Shannon Entropy
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# split dataset via certain feature
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # the following 2 steps reunited array element by reducing axis
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# choose best feature to split
def chooseBestFeatToSplit(dataSet):
    numFeat = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeat):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat

# count majority
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# create tree
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):	# recursion exit on all labels being the same
        return classList[0]
    if len(dataSet[0]) == 1:	# recursion exit on removing all features
        return majorityCnt(classList)
    bestFeat = chooseBestFeatToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])	# remove non-best-feature
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]	# array in Python passed with reference
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree
    
# get leaf amount
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# get tree depth
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth

# padding text between child node and parent node
def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)
    
# plot node
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt, textcoords='axes fraction', va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

# plot tree
def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD
    
# create tree plot
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
    
# apply decision tree
def classify(inTree, featLabels, testVec):
    firstStr = list(inTree.keys())[0]
    secondDict = inTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# cache decision tree
def storeTree(inTree, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(inTree, f)
    
# fetch decision tree
def grabTree(filename):
    import pickle
    with open(filename, 'rb') as f:
        return pickle.load(f)
```

> Test

```python
filename = 'data/lenses.txt'
with open(filename, 'r') as f:
    lenses = [inst.strip().split('\t') for inst in f.readlines()]
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses, lensesLabels)
    createPlot(lensesTree)
```

## 3. Naive Bayes

> Bayes principle: $p(c_i|x, y)={{p(x, y|c_i)p(c_i)}\over{p(x, y)}}$
>
> - if $p(c_1|x, y)>p(c_2|x, y)$, then target belongs to category $c_1$
> - if $p(c_1|x, y)<p(c_2|x, y)$, then target belongs to category $c_2$

```python
from numpy import *

# create vocabulary list
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# words to vector based on set-of-words model
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s not in the vocabulary" % word)
    return returnVec

# words to vector based on bag-of-words model
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

# naive Bayes classifier trainer
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # initialize numerators and denominators
    # to avoid getting 0 after several float that < 1 multiply
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

# naive Bayes classifier
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
    
# text parser
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]
```

> Test

```python
filename = 'data/bayes.txt'
dataSet = [] 
with open(filename, 'r') as f: 
    dataSet = [line.strip().split(' ') for line in f.readlines()]
category = [0, 1, 0, 1, 0, 1]
vocabList = createVocabList(dataSet)
trainMat = []
for data in dataSet:
    trainMat.append(bagOfWords2Vec(vocabList, data))
p0V, p1V, pAb = trainNB0(array(trainMat), array(category))

testEntry = ['love', 'my', 'dalmation']
thisData = array(bagOfWords2Vec(vocabList, testEntry))
print(testEntry, ' classified as: ', classifyNB(thisData, p0V, p1V, pAb))

testEntry = ['stupid', 'garbage']
thisData = array(bagOfWords2Vec(vocabList, testEntry))
print(testEntry, ' classified as: ', classifyNB(thisData, p0V, p1V, pAb))
```

## 4. Logistic Regression

> 1. Sigmoid function: $\sigma(z)={{1}\over{1+e^{-z}}}$
>
> 2. Gradient ascent algorithm: $w:=w+\alpha\triangledown_wf(w)$
> 3. Gradient descent algorithm: $w:=w-\alpha\triangledown_wf(w)$

```python
from numpy import *

# sigmoid function
def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

# gradient ascent
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m, n = shape(dataMatrix)
    alpha = 0.00001
    maxCycles = 1000
    weights = ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

# random gradient ascent algorithm
def stocGradAscent0(dataMatrix, classLabels):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# optimized random gradient ascent algorithm
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# plot best fit
def plotBestFit(weights, dataMat, labelMat):
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (weights[0] + weights[1]*x)/weights[2]
    ax.plot(x, array(y.tolist()[0]))	# apply gradAscent(...)
    # ax.plot(x, y)						# apply stocGradAscent0(...)
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.show()
```

> Test

```python
filename = 'data/logRes.txt'
dataMat = []; labelMat = []
with open(filename, 'r') as f:
    for line in f.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))    

plotBestFit(stocGradAscent0(array(dataMat), labelMat), dataMat, labelMat)
plotBestFit(stocGradAscent1(array(dataMat), labelMat), dataMat, labelMat)
plotBestFit(gradAscent(array(dataMat), labelMat), dataMat, labelMat)
```

