from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

boston = load_boston()
x_data = boston.data # 导入所有特征变量
y_data = boston.target # 导入目标值（房价）
name_data = boston.feature_names #导入特征名
# print(boston.feature_names)
# print(X[0])

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(x_data,y_data,test_size=0.2,random_state=3)

# plt.figure(figsize=(20, 20), dpi=50)
# for i in range(13):
#     plt.subplot(3,5,i+1)
#     plt.scatter(x_data[:,i],y_data,s = 20)
#     plt.title(name_data[i])
# plt.show()

import time
from sklearn.linear_model import LinearRegression

model = LinearRegression()
start = time.time()
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
cv_score = model.score(X_test, y_test)
print('elaspe: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(time.time()-start, train_score, cv_score))

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline

model = polynomial_model(degree=2)
#degree=3时出现过拟合现象
start = time.time()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
cv_score = model.score(X_test, y_test)
print('elaspe: {0:.6f}; train_score: {1:0.6f}; cv_score: {2:.6f}'.format(time.time()-start, train_score, cv_score))



from utils import *

from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plt.figure(figsize=(13, 4), dpi=100)
title = 'Learning Curves (degree={0})'
degrees = [1, 2, 3]

start = time.time()
for i in range(len(degrees)):
    plt.subplot(1, 3, i + 1)
    plot_learning_curve(plt, polynomial_model(degrees[i]), title.format(degrees[i]), x_data, y_data, ylim=(0.01, 1.01), cv=cv)
plt.show()
print('elaspe: {0:.6f}'.format(time.time()-start))
