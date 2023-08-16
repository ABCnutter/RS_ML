import csv
import os
import numpy as np
from sklearn import tree
from osgeo import gdal
import ospybook as pb
from osgeo.gdalconst import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import inv, eig
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import *
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import *
from osgeo import gdal
from osgeo import osr
import os
import joblib
from sklearn.feature_selection import *
from numpy import set_printoptions

Feature_desertification = pd.read_excel('Model Selection.xlsx')  # 读入数据集

print(Feature_desertification.shape)  # 数据集形状
print(Feature_desertification.head(2))  # 显示数据集前两行
print(Feature_desertification.Class.value_counts())  # 显示响应变量Class的取值分布

# 取出数据矩阵x与响应变量y
x = Feature_desertification.iloc[:, :-1]
y = Feature_desertification.iloc[:, -1]

# # 变量间的相关性可视化
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# print(x.corr())  # 求相关系数
# sns.heatmap(x.corr(), cmap='Blues')
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()

# 分层随机抽样，随机抽取50个观测值作为测试集，而其余观测值构成训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=500, random_state=1)

# KNN算法调参
# 数据的标准化
scaler = StandardScaler()
scaler.fit(x_train)

x_train_s = scaler.transform(x_train)
x_test_s = scaler.transform(x_test)
#
# print(np.mean(x_train_s, axis=0))
# print(np.std(x_train_s, axis=0))
# print(np.mean(x_test_s, axis=0))
# print(np.std(x_test_s, axis=0))

# # 寻求最优超参数max_features
# # 将响应变量变为虚拟变量
# y_train_dummy = pd.get_dummies(y_train)
# y_train_dummy = y_train_dummy.iloc[:, 1]
# # 设定10折交叉验证的参数格网和分层随机分组，并将其传入GridSearchCV()
# param_grid_KNN = {'n_neighbors': range(1, 51)}
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# model_KNN = GridSearchCV(KNeighborsClassifier(), param_grid_KNN, cv=kfold)
# model_KNN.fit(x_train_s, y_train)
# print('最优n_neighbors参数为：', model_KNN.best_params_)

# 模型预测
model_KNN = KNeighborsClassifier(n_neighbors=12)
print(model_KNN)
model_KNN.fit(x_train_s, y_train)
print('KNN邻近法的测试集预测准确率为：', model_KNN.score(x_test_s, y_test))
# 预测
pred_KNN = model_KNN.predict(x_test_s)  # 预测测试集
print(pred_KNN)
table_KNN = pd.crosstab(y_test, pred_KNN, rownames=['Actual'], colnames=['Predicted'])  # 混淆矩阵
print('KNN的混淆矩阵：','\n',table_KNN)
print('KNN的Kappa系数：',cohen_kappa_score(y_test, pred_KNN))  # 卡帕系数

# RF算法调优
# 寻求最优超参数max_features
# 设定10折交叉验证的参数格网和分层随机分组，并将其传入GridSearchCV()

# score_lt = []
# n_estimators_values = 0
# for i in range(0,200,5):
#     model_RF = RandomForestClassifier(n_estimators=i+1,random_state=123)
#     model_RF.fit(x_train,y_train)
#     score = model_RF.score(x_test,y_test)
#     score_lt.append(score)
# score_max = max(score_lt)
# print('最大得分：{}'.format(score_max),'子树数量为：{}'.format(score_lt.index(score_max)*5+1))
# # 绘制学习曲线
# x = np.arange(1,201,5)
# plt.subplot(111)
# plt.plot(x, score_lt, 'o-')
# plt.show()


# param_grid_RF = {'max_features': range(1, 7)}
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# model_RF = GridSearchCV(RandomForestClassifier(n_estimators=81,random_state=123), param_grid_RF, cv=kfold)
# model_RF.fit(x_train, y_train)
# print('最优max_features参数为：', model_RF.best_params_)
model_RF = RandomForestClassifier(n_estimators=81, max_features=2, random_state=123)
print(model_RF)
model_RF.fit(x_train, y_train)
print('随机森林的测试集预测准确率为：', model_RF.score(x_test, y_test))

pred_RF = model_RF.predict(x_test)  # 预测测试集
print(pred_RF)
table_RF = pd.crosstab(y_test, pred_RF, rownames=['Actual'], colnames=['Predicted'])  # 混淆矩阵
print('RF的混淆矩阵：','\n',table_RF)
print('RF的Kappa系数：',cohen_kappa_score(y_test, pred_RF))  # 卡帕系数

# # GBM算法调优
# # 寻求最优超参数max_features
# # 将响应变量变为虚拟变量
# y_train_dummy = pd.get_dummies(y_train)
# y_train_dummy = y_train_dummy.iloc[:, 1]
# param_distributions = {'n_estimators':range(1,300),'max_depth':range(1,10),'subsample':np.linspace(0.1,1,10),
#                        'learning_rate':np.linspace(0.1,1,10)}
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# model_GBM = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=123), param_distributions=param_distributions,
#                                cv=kfold,random_state=66)
# # model_GBM = GradientBoostingClassifier(random_state=123)
# # print(model_GBM)
# model_GBM.fit(x_train,y_train)
# print('GBM最优参数为：', model_GBM.best_params_)
# # GBM最优参数为:'subsample': 1.0, 'n_estimators': 169, 'max_depth': 2, 'learning_rate': 0.6
# model_GBM = model_GBM.best_estimator_
model_GBM = GradientBoostingClassifier(subsample=1.0, n_estimators=169, max_depth=2, learning_rate=0.6,random_state=123)
print(model_GBM)
model_GBM.fit(x_train, y_train)
print('GBM的测试集预测准确率为：',model_GBM.score(x_test,y_test))

pred_GBM = model_GBM.predict(x_test)  # 预测测试集
print(pred_GBM)
table_GBM = pd.crosstab(y_test, pred_GBM, rownames=['Actual'], colnames=['Predicted'])  # 混淆矩阵
print('GBM的混淆矩阵：','\n',table_GBM)
print('GBM的Kappa系数：',cohen_kappa_score(y_test, pred_GBM))  # 卡帕系数
