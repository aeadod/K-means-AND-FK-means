import copy
import math
import random
import time
import numpy as np
import operator
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from csv import reader
global MAX  # 用于初始化隶属度矩阵U
MAX = 10000.0
global Epsilon  # 结束条件
Epsilon = 0.0000001
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
            dataset[i][60]=1
        else:
            dataset[i][60]=2

filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(0, len(dataset[0])-1):
     str_column_to_float(dataset, i)
str_column_to_int(dataset)
dataset=np.array(dataset)

#利用random库中的函数将原始数据的顺序进行随机重排，在order中记录原始数据重排的顺序
def randomize_data(data):
    order = list(range(0, len(data)))
    random.shuffle(order)
    new_data = [[] for i in range(0, len(data))]
    for index in range(0, len(order)):
        new_data[index] = data[order[index]]
    return new_data, order
#利用上一个函数保存的顺序信息对数据进行还原
def de_randomise_data(data, order):
    new_data = [[] for i in range(0, len(data))]
    for index in range(len(order)):
        new_data[order[index]] = data[index]
    return new_data

def print_matrix(list):
    for i in range(0, len(list)):
        print(list[i])
#初始化U矩阵，使每个样本满足归一化条件
def initialize_U(data, cluster_number):
    global MAX
    U = []
    for i in range(0, len(data)):
        current = []
        rand_sum = 0.0
        for j in range(0, cluster_number):
            dummy = random.randint(1, int(MAX))
            current.append(dummy)
            rand_sum += dummy
        for j in range(0, cluster_number):
            current[j] = current[j] / rand_sum
        U.append(current)
    return U

#计算特征空间中两个样本点的欧氏距离。
def distance(point, center):
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)
#给定的最小阈值来判断算法是否停止。
#如果某次迭代中隶属度矩阵每个元素的前后变化都小于阈值的话，停止更新隶属度矩阵并退出算法
def end_conditon(U, U_old):
    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True
#对于每个样本将隶属度最大的那个类设置为要把它分的类别，
#即把隶属度设置为1，把其他的隶属度设置为0
def normalise_U(U):
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U
# m的最佳取值范围为[1.5，2.5]
def fuzzy(data, cluster_number, m):
    # 初始化隶属度矩阵U
    U = initialize_U(data, cluster_number)
    print_matrix(U)
    # 循环更新U
    while (True):
        # 创建它的副本，以检查结束条件
        U_old = copy.deepcopy(U)
        # 计算聚类中心
        C = []
        for j in range(0, cluster_number):
            current_cluster_center = []
            for i in range(0, len(data[0])):
                dummy_sum_num = 0.0
                dummy_sum_dum = 0.0
                for k in range(0, len(data)):
                    # 分子
                    dummy_sum_num += (U[k][j] ** m) * data[k][i]
                    # 分母
                    dummy_sum_dum += (U[k][j] ** m)
                # 第i列的聚类中心
                current_cluster_center.append(dummy_sum_num / dummy_sum_dum)
            # 第j簇的所有聚类中心
            C.append(current_cluster_center)
        # 创建一个距离向量, 用于计算U矩阵。
        distance_matrix = []
        for i in range(0, len(data)):
            current = []
            for j in range(0, cluster_number):
                current.append(distance(data[i], C[j]))
            distance_matrix.append(current)
        # 更新U
        for j in range(0, cluster_number):
            for i in range(0, len(data)):
                dummy = 0.0
                for k in range(0, cluster_number):
                    dummy += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2 / (m - 1))# 分母
                U[i][j] = 1 / dummy
        if end_conditon(U, U_old):
            print("结束聚类")
            break
    U = normalise_U(U) #去模糊化 U
    return U

def checker_sonar(final_location):
    right=0.0

    checker1=[0,0]
    for i in range(0,97):
        for j in range(0,len(final_location[0])):
            if final_location[i][j]==1:
                checker1[j]+=1
    right+=max(checker1)
    #print(checker1)
    checker2=[0,0]
    for i in range(0,111):
        for j in range(0,len(final_location[0])):
            if final_location[i + 97][j] == 1:
                checker2[j] += 1  # checker分别统计每一类分类正确的个数
    right+=max(checker2)
    #print(checker2)
    print('分类正确的个数是:', right)
    answer = right / 208 * 100
    c=answer
    print("准确率：" + str(answer) + "%")
    return c

if __name__ == '__main__':
    # 加载数据
    data = dataset
    # 随机化数据
    data, order = randomize_data(data)

    # 调用模糊C均值函数
    a=[]
    for ii in np.arange(2,2.1,0.1):
        final_location = fuzzy(data, 2, ii)
    # 还原数据
        final_location = de_randomise_data(final_location, order)
        aa=checker_sonar(final_location)
        a.append(aa)
    #print_matrix(final_location)
    # 准确度分析
    print(a)
    plt.figure(1)
    plt.xlabel('b的取值')
    plt.ylabel('准确率')
    plt.title('b取值不同时聚类的准确率')
    plt.plot(np.arange(2,2.1,0.1),a)
    plt.show()
