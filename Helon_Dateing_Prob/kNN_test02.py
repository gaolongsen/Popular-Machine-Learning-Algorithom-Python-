# -*- coding: UTF-8 -*-

from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator

"""
Instruction of Function: kNN Algorithm, Classifier

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


def classify0(inX, dataSet, labels, k):
    # numpy function shape[0] return the number of lines of dataSet
    dataSetSize = dataSet.shape[0]
    # In column row direction, duplicate inX one time; in row direction,duplicate inX dataSetSize times
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # Square calculation by subtraction of two-dimensional features
    sqDiffMat = diffMat ** 2
    # All elements in sum() to be added together, elements in sum(0) column added, sum(1) row added
    sqDistances = sqDiffMat.sum(axis=1)
    # Square to calculate the distance
    distances = sqDistances ** 0.5
    # Returns the index value of the elements in distances sorted from smallest to largest
    sortedDistIndices = distances.argsort()
    # Define a dictionary to record the number of categories
    classCount = {}
    for i in range(k):
        # Fetch the category of the first k elements
        voteIlabel = labels[sortedDistIndices[i]]
        # dict.get(key,default=None),get() method in dictionary,it will return the specified key value. If the value is not in the dictionary, retutn default value.
        # Count category times
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # In Python3, replace iteritems() in python2 with items() 
    # key=operator.itemgetter(1) Sorting by dictionary values
    # key=operator.itemgetter(0) Sorting by dictionary keys
    # reverse Descending Sorting Dictionary
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    #
    # The category with the highest number of returns is the category to be classified
    return sortedClassCount[0][0]


"""
Function Description: Open and parse the file, then classify the data: 
'1' represents dislike, '2' represents general charm, '3' represents very charming
Parameters:
	filename - the name of the file
Returns:
	returnMat - Matrix only with eigenvector in diagonal 
	classLabelVector - classifying vector for labels 

Modify:
	2021-10-30
"""


def file2matrix(filename):
    # Open the file, then make the file to the specified coding
    fr = open(filename, 'r', encoding='utf-8')
    # Read the contend in the file
    arrayOLines = fr.readlines()
    # For all the UTF-8 files which contained BOM, we should remove BOM, or it will make error
    arrayOLines[0] = arrayOLines[0].lstrip('\ufeff')
    # Get the line of the file
    numberOfLines = len(arrayOLines)
    # Return the Matrix of Numpy. The range of lines for parsed data with 5 rows and 3 columns
    returnMat = np.zeros((numberOfLines, 3))
    # The returned vector of classification labels
    classLabelVector = []
    # Index value for the rows
    index = 0

    for line in arrayOLines:
        # s.strip(rm)，when rm is empty, delete blank characters by default('\n','\r','\t',' ')
        line = line.strip()
        # s.split(str="",num=string,cout(str)) --> Slice the string according to the '\t' separator。
        listFromLine = line.split('\t')
        # The first three columns of the data are extracted and stored in the NumPy matrix of returnMat,
        # which is the feature matrix
        returnMat[index, :] = listFromLine[0:3]
        # Classification according to the degree of liking marked in the text, 1 means dislike, 2 means average charm, 3 means very charming
        # For datingTestSet2.txt, the last label has been processed and the labels have been changed to 1, 2, 3
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector


"""
Data Visualization
Function Description: Visualization Data
Parameters:
	datingDataMat - feature Matrix 
	datingLabels - Classifying Label
Returns:
	None
Modify:
	2021-10-30
"""


def showdatas(datingDataMat, datingLabels):

    # The fig canvas is separated into 1 row and 1 column, without sharing the x and y axes, and the size of the fig canvas is (13,8)
    # Function Description:Open and parse the file, classify the data: 1 represents dislike, 2 represents general charm, 3 represents very charming
    # When nrow=2, nclos=2, the fig canvas is divided into four regions, axs[0][0] means the first row of the first region
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    # Draw a scatter plot, using the first (frequent flyer routine) and second (game) columns of the datingDataMat matrix to draw scatter data, with a scatter size of 15 and a transparency of 0.5
    axs[0][0].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 1], color=LabelsColors, s=15, alpha=.5)
    # Set title, x-axis label, y-axis label
    axs0_title_text = axs[0][0].set_title(u'Rate of Frequent flyer miles and Time consumed playing video games')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'Frequent flyer miles')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'Time consumed playing video games')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    # Plot scatter diagram, use the data in first and the third colum of datingDataMat matrix to plot the figure. The size of the scatter points is 15, transparency is 0.5  
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # Set the titlef for both x-axis and y-axis 
    axs1_title_text = axs[0][1].set_title(u'Frequent flyer miles earned per year vs. liters of ice cream consumed per week')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'flyer miles earned per year')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'Liters of ice cream consumed per week')
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # Plot scatter diagram, use the data in second(playing the game) and the third colum(eating iceCream) of datingDataMat matrix to plot the figure. The size of the scatter points is 15, transparency is 0.5  
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # Set the titlef for both x-axis and y-axis 
    axs2_title_text = axs[1][0].set_title(u'Percentage of time consumed playing video games vs. liters of ice cream consumed per week')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'Percentage of time consumed by playing video games')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'Liters of ice cream consumed per week')
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # Setting legend
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # Adding legend
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # Showing pictures
    plt.show()


"""
Function Description: Normalize the Data
Parameters:
	dataSet - Characteristic matrix
Returns:
	normDataSet - Normlized Characteristic matrix
	ranges - Range of the data
	minVals - The minimum value of the data

Modify:
	2020-10-30
"""


def autoNorm(dataSet):
    # Got the minimum value of the data
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # The range of the minimum and maximum values
    ranges = maxVals - minVals
    # shape(dataSet)return the number of the columns amd rows of dataSet matrix
    normDataSet = np.zeros(np.shape(dataSet))
    # return the number of rows of dataSet
    m = dataSet.shape[0]
    # Primary value minus the minimum value
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # The normalized data is obtained by dividing the difference between the maximum and minimum values
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # Returns the normalized data result, data range, and minimum value
    return normDataSet, ranges, minVals


"""
Function Description: Classifier Testing Function
Take ten percent of the data as test data to test the correctness of the classifier
Parameters:
	None
Returns:
	None

Modify:
	2020-10-30
"""


def datingClassTest():
    # Opne the name of the file
    filename = "datingTestSet.txt"
    # Store the returned feature matrix and classification vector into the datingDataMat and datingLabels respectively
    datingDataMat, datingLabels = file2matrix(filename)
    showdatas(datingDataMat, datingLabels)
    # Take ten percent of all data
    hoRatio = 0.10
    # Data normalization, return the normalized matrix, data range, data minimum
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # Got the number of the columns of normMat
    m = normMat.shape[0]
    # The number of ten percent of the testing data 
    numTestVecs = int(m * hoRatio)
    # Classification error counting
    errorCount = 0.0

    for i in range(numTestVecs):
        # The numTestVecs are used as the test set, and the m-numTestVecs are used as the training set.
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        print("Classifying Result:%s\tReal Category:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("Rate of Error:%f%%" % (errorCount / float(numTestVecs) * 100))


"""
Function Description: By inputting the 3D features of a person, the output is classified
Parameters:
	None
Returns:
	None

Modify:
	2020-10-30
"""


def classifyPerson():
    # Output result
    resultList = ['Dislike', 'Just a little like', 'Really love']
    # 3-D features of the user imput
    precentTats = float(input("Percentage of time spent playing video games:"))
    ffMiles = float(input("Number of frequent flyer miles earned per year:"))
    iceCream = float(input("Liters of ice cream consumed per week:"))
    # The name of the file opened
    filename = "datingTestSet.txt"
    # Open and process data
    datingDataMat, datingLabels = file2matrix(filename)
    # Normalize  the training set
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # Generate the array of Numpy and testing set
    inArr = np.array([ffMiles, precentTats, iceCream])
    # Normalize the testing set
    norminArr = (inArr - minVals) / ranges
    # Return the classifying results
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    # Print the result
    print("Maybe you %s this guy" % (resultList[classifierResult - 1]))


"""
Function Description: main function
Parameters:
	None
Returns:
	None

Modify:
	2021-10-30
"""
if __name__ == '__main__':
    datingClassTest()
