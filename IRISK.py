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
        if dataset[i][4]=='setosa':
            dataset[i][4]=0
        elif dataset[i][4]=='versicolor':
            dataset[i][4]=1
        else:
            dataset[i][4]=2
filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(0, len(dataset[0])-1):
     str_column_to_float(dataset, i)
str_column_to_int(dataset)
def gety(dataset):
    y=[]
    for i in dataset:
        y.append(i[4])
    return y
y=gety(dataset)
y=array(y)
for i in dataset:
    i.pop()
dataset=mat(dataset)

def loadDataSet(fileName):
    dataSet = []
    f = open(fileName)
    for line in f.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataSet.append(fltLine)
    return mat(dataSet)

# 求两个向量的欧式距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))
#随机生成K个聚类中心
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
            for j in range(k):  # 寻找最近质心
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

def checker_iris(clusterAssment):
    right = 0.0
    for k in range(0, 3):
        checker = [0, 0, 0]
        for i in range(0, 50):
            if clusterAssment[:,0][k*50+i] == 1:
                checker[0] += 1
            elif clusterAssment[:,0][k*50+i]==0:
                checker[1]+=1
            else:
                checker[2] += 1
        right += max(checker)  # 累加分类正确的个数
        print(checker)
    print('分类正确的个数是:', right)
    answer = right / 150 * 100
    c=answer
    print("准确率：" + str(answer) + "%")
    return c

if __name__ == "__main__":
    dataSet =dataset
    #print(dataSet)
    #print(type(dataSet))
    centroids, clusterAssment = KMeans(dataSet, 3)
    a=checker_iris(clusterAssment)
    #print(a)
    # centroids, clusterAssment = biKmeans(dataSet, 4)
    #showCluster(dataSet, 3, clusterAssment, centroids)




