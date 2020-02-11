# -*- coding: utf-8 -*-
# @Time    : 2020/2/10 22:15
# @Author  : Sherlock June
# @Email   : wangjun980213@163.com
# @File    : test02.py
# @Software: PyCharm
'''
实现knn的回归预测并绘图
'''
import numpy as np
import matplotlib.pyplot as plt
n_dots = 40
X = 5 * np.random.rand(n_dots,1)
y = np.cos(X).ravel()
#拉平，但我们在平时使用的时候flatten()更为合适.在使用过程中flatten()分配了新的内存,
#但ravel()返回的是一个数组的视图.视图是数组的引用(说引用不太恰当,因为原数组和ravel()返回后的数组的地址并不一样),
# print(y)
#添加一些噪声
y += 0.2 * np.random.rand(n_dots) - 0.1
#训练模型
from sklearn.neighbors import KNeighborsRegressor
k = 5
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(X,y)
# plt.scatter(X, y)
# plt.show()
#生成足够密集的点并进行预测
T = np.linspace(0,5,500)[:,np.newaxis]
'''
linspace在指定的间隔内返回均匀间隔的数字
x------array([4, 6, 6, 6, 5])
x1 = x[np.newaxis, :]
x1-----array([[4, 6, 6, 6, 5]])
x2 = x[:, np.newaxis]
x2-----array([[4],
       [6],
       [6],
       [6],
       [5]])
'''
# print(T)
y_pred = knn.predict(T)
score = knn.score(X,y)
print(score)
#画出拟合曲线
plt.figure(figsize=(5 , 4),dpi=144)
plt.scatter(X,y,c ='g', label='data', s=20)
plt.plot(T, y_pred, c='k', label='prediction', lw=1)
plt.axis('tight')#tight:坐标轴数据显示更明细
plt.title("KNeighborsRegressor(k = %i)" % k)
plt.show()