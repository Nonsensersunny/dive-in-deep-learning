# Machine Learning in Action

## *Preface*

> Steps of machine learning

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
    
# classifier
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0
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

## 5. SVM

> Inducement of SVM: [Click here for more](https://www.cnblogs.com/xxrxxr/p/7535587.html#catalog)

## 6. Adaboost

## 7. Linear Regression

### 7.1 Standard Linear Regression

> Assume: $y=wx$, for sample dataset $x^T$, square error: $\sum_{i=1}^m(y_i-x_i^Tw)^2$, then in order to get $w$, take the derivative of $w$: $X^T(Y-Xw)$, and then $\hat{w}=(X^TX)^{-1}X^Ty$.

```python
from numpy import *

# linear regression
def standRegres(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    # compute the determinant of matrix xTx to judge whether it's singular
    if linalg.det(xTx) == 0.0:
        print("The matrix is singular, cannot do inverse")
        return
    return xTx.I * (xMat.T*yMat)

# plot regressive line
def plotRegres(xArr, yArr):
    import matplotlib.pyplot as plt
    ws = standRegres(xArr, yArr)
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat*ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy*ws
    ax.plot(xCopy[:, 1], yHat)
    plt.show()
```

> Test

```python
filename = 'data/linearReg.txt'
numFeat = len(open(filename).readline().split(' ')) - 1
dataMat = []; labelMat = []
with open(filename, 'r') as f:
    for line in f.readlines():
        lineArr = []
        curLine = line.strip().split(' ')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
```

### 7.2 Locally Weighted Linear Regression

> Giving weight to sample points near the regressive line: $\hat{w}=(X^TWX)^{-1}X^TWy$, and in which $w$ is a weight matrix. Like kernel function in SVM, kernel also applied here to weight the sample points near the line, Gaussian kernel is mostly applied: $x(i, i)=\exp({{|x^{(i)}-x|}\over{-2{k^2}}})$

```python
from numpy import *

# locally weighted linear regression
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye(m))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("The matrix is singular, cannot do inverse")
        return
    return testPoint * ws

# get all approximate point via lwlr(...)
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

# plot regressive line
def plotRegres(xArr, yArr):
    import matplotlib.pyplot as plt
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    xMat = mat(xArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s=2, c='red')
    plt.show()
```

### 7.3 Ridge Regression

> Once feature number is greater than sample amount, problems occur when computing $(X^TX)^{-1}$ a.d ridge regression is applied to solve this kind of problem via adding a $m\times{m}$ eye: $\lambda{I}$ and then do inverse of $X^TX+\lambda{I}$. Then: $\hat{w}=(X^TX+\lambda{I})^{-1}X^Ty$.

```python
from numpy import *

# ridge regression
def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("The matrix is singular, cannot do inverse")
        return
    return denom.I * (xMat.T*yMat)

# test ridge regression
def ridgeTest(xArr, yArr):
    xMat = mat(xArr); yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans)/xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat
```

## 8. CART

> Since the deletion of feature applied in decision tree is too rapid and DT is unable to deal with continuous features, then CART, a kind of tree construction algorithm is applied.

```python
from numpy import *

# binary split dataset
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

# CART
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = creteTree(rSet, leafType, errType, ops)
    return retTree

# regressive leaf
def regLeaf(dataSet):
    return mean(dataSet[:, -1])

# regressive error
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

# choose best split point
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    to1S = ops[0]; to1N = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < to1N) or (shape(mat1)[0] < to1N): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < to1S:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat[0])[0] < to1N) or (shape(mat1)[0] < to1N):
        return None, leafType(dataSet)
    return bestIndex, bestValue
```

> Test

```python
filename = 'data/cart.txt'
dataMat = []
with open(filename, 'r') as f:
    for line in f.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
```

## 9. K-means

```python
from numpy import *

# distance of 2 vectors
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

# random centroids
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids

# K-means
def kMeans(dataSet, k, distMeans=distEclud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex: clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

# bisecting K-means
def biKmeans(dataSet, k, distMeans=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeans(mat(centroid0), dataSet[j, :])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeans)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is : ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return mat(centList), clusterAssment
```

> Test

```python
filename = 'data/k-means.txt'
datMat = []
with open(filename, 'r') as f:
    datMat = mat([list(map(float, line.strip().split('\t'))) for line in f.readlines()])
    
```

## 10. Apriori

> Apriori principle: the frequency of a subset follows the set $=>$ Inverse negative proposition: the super-sets of a infrequent set are also of low frequency.

```python
# create c(1)(1)
def createC1(dataSet):
    c1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in c1:
                c1.append([item])
    c1.sort()
    return list(map(frozenset, c1))

# get the confidence of the elements of a set
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():	ssCnt[can] = 1
                else:						ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

# create Ck
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

# Apriori
def apriori(dataSet, minSupport=0.5):
    c1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, c1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

# generate relation rule set
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

# calculate confidence
def calConf(freqSet, H, supportData, brl, minConf=0.7):
    prunnedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunnedH.append(conseq)
    return prunnedH

# appraise rules from consequence
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if len(freqSet) > (m + 1):
        Hmp1=  apprioriGen(H, m+1)
        Hmp1 = calConf(freqSet, Hmp1, supportData, brl, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1m, supportData, brl, minConf)
```

