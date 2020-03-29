# encoding: utf-8
# 导入testSet中的数据，利用K均值算法对进行分类
from numpy import *
import operator
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from csv import reader
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
def str_column_to_int(dataset):
    for i in range(len(dataset)):
        if dataset[i][60]=='R':
            dataset[i][60]=0
        else:
            dataset[i][60]=1

filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(0, len(dataset[0])-1):
     str_column_to_float(dataset, i)
str_column_to_int(dataset)
for i in dataset:
    i.pop()
dataset=mat(dataset)

# 求两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]  # n是列数
    centroids = mat(zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])  # 找到第j列最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)  # 求第j列最大值与最小值的差
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)  # 生成k行1列的在(0, 1)之间的随机数矩阵
    return centroids

def KMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]  # 数据集的行
    clusterAssment = mat(zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 遍历数据集中的每一行数据
            minDist = inf
            minIndex = -1
            for j in range(k):  # 寻找最近样本点
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:  # 更新最小距离和质心下标
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2  # 记录最小距离质心下标，最小距离的平方
        print(centroids)
        for cent in range(k):  # 更新质心位置
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获得距离同一个质心最近的所有点的下标，即同一簇的坐标
            centroids[cent, :] = mean(ptsInClust, axis=0)  # 求同一簇的坐标平均值，axis=0表示按列求均值
    return centroids, clusterAssment

def checker_sonar(clusterAssment):
    right=0.0
    checker1=[0,0]
    for i in range(0,97):
        if clusterAssment[:, 0][i] == 1:
            checker1[0] += 1
        else:
            checker1[1] += 1
    right+=max(checker1)
    #print(checker1)
    checker2=[0,0]
    for i in range(0,111):
        if clusterAssment[:, 0][i+97] == 1:
            checker2[0] += 1
        else:
            checker2[1] += 1
    right+=max(checker2)
    print(checker2)
    print('分类正确的个数是:', right)
    answer = right / 208 * 100
    c=answer
    print("准确率：" + str(answer) + "%")
    return c

if __name__ == "__main__":
    dataSet =dataset
    #print(dataSet)
    #print(type(dataSet))
    centroids, clusterAssment = KMeans(dataSet, 2)
    a=checker_sonar(clusterAssment)
    #print(a)






