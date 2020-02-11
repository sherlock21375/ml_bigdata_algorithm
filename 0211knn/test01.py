# -*- coding: utf-8 -*-
# @Time    : 2020/2/10 18:37
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : test01.py
# @Software: PyCharm
'''
    简单使用sklearn的make_blobs生成聚类数据测试
    绘制图形，选择最合适的k值
    使用knn算法进行分类，并将邻居进行连线
'''
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
#生成数据
centers = [[-2,2], [2,2],[0,4]]
X, y =make_blobs(n_samples=100 , centers=centers,random_state= 0, cluster_std=0.60)
plt.figure(figsize=(5,4),dpi=100)
c = np.array(centers)#更改为numpy.ndarray类型
# print(X)
# print(y)
# plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection  import cross_val_score

def choose_k():
    k_range = range(1, 31)
    k_error = []
    # 循环，取k=1到k=31，查看误差效果
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        # cv参数决定数据集划分比例，这里是按照5:1划分训练集和测试集
        scores = cross_val_score(knn, X, y, cv=6, scoring='accuracy')
        k_error.append(1 - scores.mean())

    # 画图，x轴为k值，y值为误差值
    plt.plot(k_range, k_error)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Error')
    plt.show()

def main():
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, cmap='cool')  # 此处c=y指的是按照y的分类进行颜色区分，cmap指色彩映射
    plt.scatter(c[:, 0], c[:, 1], c='orange', s=50, marker='^')
    #模型训练
    k =5
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X,y)
    #模型预测
    X_sample = np.array([[0, 1],[-2,4],[3,1]])#新版本sklearn规定，传入的必须是二维数组，同时我也将其处理为np格式
    y_sample = clf.predict(X_sample)
    print(y_sample)
    # print(X_sample[:,1])
    neighbors = clf.kneighbors(X_sample,return_distance=False)
    # print(neighbors)
    plt.scatter(X_sample[:,0],X_sample[:,1],c=y_sample,marker='x',cmap='cool')#注意，若给定的测试数据集最终预测类型只有两类，导致颜色错误
    for index,element in enumerate(neighbors):#enumrate函数来获取列表中索引的位置
        for i in element:
            # print(index)
            plt.plot([X[i][0], X_sample[index][0]], [X[i][1], X_sample[index][1]],'g--', linewidth=0.6) #预测点与距离最近的k个样本的连线
    # plt.plot([1,2,3], [1,2,3], 'go-', label='line 1', linewidth=2)
    plt.show()

# choose_k()
main()