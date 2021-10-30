# K-NN This is a kind of Classification Algorithm in Machine Learning which can be used to classify the data
# Author: Shine Gao
import numpy as np
import operator

"""
Instruction of Function: Create dataset
 
Parameters:
    None
Returns:
    group - dataset
    labels - classified flag
Modify:
    2021-10-30
"""


def createDataSet():
    group = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])  # Four couples of 2-dimension characteristics
    labels = ['Romance film', 'Romance film', 'Action Movies',
              'Action Movies']  # The labels of four couples of 2-dimension characteristics
    return group, labels


"""
Instruction of Function: kNN algorithm,classifier
 
Parameters:
    inX - The data which can be used to classify (Testing Data)
    dataSet - Data used for training (Training Data)
    labels - Classified Flag
    k - Parameter of kNN, to select the number of minimum distance of k  
Returns:
    sortedClassCount[0][0] - Result of classifying 
 
Modify:
    2021-10-30
"""

def classify0(inX, dataset, labels, k):
    dataSetSize = dataset.shape[0] # in numpy class, shape[0] return the numbers of row of the dataset
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataset # tile function is to extend the inX set(as a whole part) to (dataSetSize, 1) dimension
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortDisIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortDisIndices[i]]

        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

        return sortedClassCount[0][0]

if __name__ == '__main__':
    group, labels = createDataSet()     # Create dataset
    test = [101, 20]
    test_class = classify0(test, group, labels, 3)
    print(test_class)     # Print dataset

