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
        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # Count category times数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # python3中用items()替换python2中的iteritems()
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
        # 将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
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
可视化数据
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

    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:, 0], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'Frequent flyer miles earned per year vs. liters of ice cream consumed per week')
    axs1_xlabel_text = axs[0][1].set_xlabel(u'flyer miles earned per year')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'Liters of ice cream consumed per week')
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:, 1], y=datingDataMat[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'Percentage of time consumed playing video games vs. liters of ice cream consumed per week')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'Percentage of time consumed by playing video games')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'Liters of ice cream consumed per week')
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    # 设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='largeDoses')
    # 添加图例
    axs[0][0].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[0][1].legend(handles=[didntLike, smallDoses, largeDoses])
    axs[1][0].legend(handles=[didntLike, smallDoses, largeDoses])
    # 显示图片
    plt.show()


"""
函数说明:对数据进行归一化

Parameters:
	dataSet - 特征矩阵
Returns:
	normDataSet - 归一化后的特征矩阵
	ranges - 数据范围
	minVals - 数据最小值

Modify:
	2017-03-24
"""


def autoNorm(dataSet):
    # 获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 最大值和最小值的范围
    ranges = maxVals - minVals
    # shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回dataSet的行数
    m = dataSet.shape[0]
    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals


"""
函数说明:分类器测试函数
取百分之十的数据作为测试数据，检测分类器的正确性

Parameters:
	无
Returns:
	无

Modify:
	2017-03-24
"""


def datingClassTest():
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    showdatas(datingDataMat, datingLabels)
    # 取所有数据的百分之十
    hoRatio = 0.10
    # 数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 获得normMat的行数
    m = normMat.shape[0]
    # 百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    # 分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        # 前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 4)
        print("分类结果:%s\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" % (errorCount / float(numTestVecs) * 100))


"""
函数说明:通过输入一个人的三维特征,进行分类输出

Parameters:
	无
Returns:
	无

Modify:
	2017-03-24
"""


def classifyPerson():
    # 输出结果
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    # 三维特征用户输入
    precentTats = float(input("玩视频游戏所耗时间百分比:"))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    # 打开的文件名
    filename = "datingTestSet.txt"
    # 打开并处理数据
    datingDataMat, datingLabels = file2matrix(filename)
    # 训练集归一化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    # 生成NumPy数组,测试集
    inArr = np.array([ffMiles, precentTats, iceCream])
    # 测试集归一化
    norminArr = (inArr - minVals) / ranges
    # 返回分类结果
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    # 打印结果
    print("你可能%s这个人" % (resultList[classifierResult - 1]))


"""
函数说明:main函数

Parameters:
	None
Returns:
	无

Modify:
	2017-03-24
"""
if __name__ == '__main__':
    datingClassTest()
