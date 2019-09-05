# Deep Learning in Action

## *Preface*

> Steps of deep learning

1. Collecting data
2. Data inputing
3. Analyzing data
4. Training algorithm
5. Testing algorithm
6. Applying algorithm

> Basics of *Numpy*

```python
from numpy import *
random.rand(4, 4)	# random 2-dimension array
random.rand(4, 4, 2)	# random 3-dimension array
randomMt = mat(random.rand(4, 4))	# random 4*4 matrix
matInv = randomMt.I	# inversion of a matrix
randomEye = randomMt*matInv		# multiplication of matrice
randomEye - eye(4)	# error of eye
```

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
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```

