import copy
import math
import random
import time

global MAX  # 用于初始化隶属度矩阵U
MAX = 10000.0

global Epsilon  # 结束条件
Epsilon = 0.0000001


import numpy as np
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
            dataset[i][4]=1
        elif dataset[i][4]=='versicolor':
            dataset[i][4]=2
        else:
            dataset[i][4]=3

filename = 'iris.csv'
dataset = load_csv(filename)
for i in range(0, len(dataset[0])-1):
     str_column_to_float(dataset, i)
str_column_to_int(dataset)
dataset=np.array(dataset)



def randomize_data(data):
    order = list(range(0, len(data)))
    random.shuffle(order)
    new_data = [[] for i in range(0, len(data))]
    for index in range(0, len(order)):
        new_data[index] = data[order[index]]
    return new_data, order


def de_randomise_data(data, order):
    new_data = [[] for i in range(0, len(data))]
    for index in range(len(order)):
        new_data[order[index]] = data[index]
    return new_data


def print_matrix(list):
    """
    以可重复的方式打印矩阵
    """
    for i in range(0, len(list)):
        print(list[i])


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

def distance(point, center):
    if len(point) != len(center):
        return -1
    dummy = 0.0
    for i in range(0, len(point)):
        dummy += abs(point[i] - center[i]) ** 2
    return math.sqrt(dummy)


def end_conditon(U, U_old):
    global Epsilon
    for i in range(0, len(U)):
        for j in range(0, len(U[0])):
            if abs(U[i][j] - U_old[i][j]) > Epsilon:
                return False
    return True


def normalise_U(U):
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] != maximum:
                U[i][j] = 0
            else:
                U[i][j] = 1
    return U
"""
    这是主函数，它将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.
    参数是：簇数(cluster_number)和隶属度的因子(m)
"""

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
            #print("结束聚类")
            break
    U = normalise_U(U)    #去模糊化 U
    return U


def checker_iris(final_location):
    right = 0.0
    for k in range(0, 3):
        checker = [0, 0, 0]
        for i in range(0, 50):
            for j in range(0, len(final_location[0])):
                if final_location[i + (50 * k)][j] == 1:  # i+(50*k)表示 j表示第j类
                    checker[j] += 1  # checker分别统计每一类分类正确的个数
        right += max(checker)  # 累加分类正确的个数
        #print(checker)
    print('分类正确的个数是:', right)
    answer = right / 150 * 100
    c=answer
    print("准确率：" + str(answer) + "%")
    return c

def check_sonar(final_location):
    right=0.0

    checker1=[0,0]
    for i in range(0,97):
        for j in range(0,len(final_location[0])):
            if final_location[i][j]==1:
                checker1[j]+=1
    right+=max(checker1)
    checker2=[0,0]
    for i in range(0,111):
        for j in range(0,len(final_location[0])):
            if final_location[i + 97][j] == 1:
                checker2[j] += 1  # checker分别统计每一类分类正确的个数
    right+=max(checker2)
    print('分类正确的个数是:', right)
    answer = right / 208 * 100
    c=answer
    print("准确率：" + str(answer) + "%")
    return c

if __name__ == '__main__':
    # 加载数据
    data = dataset
    a=[]
    # print_matrix(data)
    # 随机化数据
    data, order = randomize_data(data)
    #print_matrix(data)
    # 现在我们有一个名为data的列表，它只是数字
    # 调用模糊C均值函数
    for i in np.arange(1.5,1.6,0.1):
        final_location = fuzzy(data, 3, i)
    # 还原数据
        final_location = de_randomise_data(final_location, order)
        aa=checker_iris(final_location)
        a.append(aa)
        print_matrix(final_location)
    # 准确度分析
    print(a)
    plt.figure(1)
    plt.xlabel('b的取值')
    plt.ylabel('准确率')
    plt.title('b取值不同时聚类的准确率')
    plt.plot(np.arange(1.5,1.6,0.1),a)
    plt.show()

