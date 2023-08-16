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

# 变量间的相关性可视化
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(x.corr())  # 求相关系数
sns.heatmap(x.corr(), cmap='Blues')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 分层随机抽样，随机抽取50个观测值作为测试集，而其余观测值构成训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=500, random_state=1)

num_fold = 10
seed = 7
scoring = 'accuracy'

models = {'LR': LogisticRegression(), 'LDA': LinearDiscriminantAnalysis(), 'KNN': KNeighborsClassifier(),
          'CART': DecisionTreeClassifier(), 'NB': GaussianNB(), 'SVM': SVC(), 'AB': AdaBoostClassifier(),
          'GBM': GradientBoostingClassifier(), 'RF': RandomForestClassifier(), 'ET': ExtraTreeClassifier()}

# 评估模型算法
results = []
for key in models:
    kfold = KFold(n_splits=num_fold, shuffle=True, random_state=seed)
    cv_results = cross_val_score(models[key], x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    print('%s : %f (%f)' % (key, cv_results.mean(), cv_results.std()))

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()


# # 随机森林估计
# model = RandomForestClassifier(n_estimators=500, max_features='sqrt', random_state=123)
# model.fit(x_train, y_train)
# print('随机森林的测试集预测准确率为：', model.score(x_test, y_test))
#
#
# # 寻求最优超参数max_features
# # 将响应变量变为虚拟变量
# y_train_dummy = pd.get_dummies(y_train)
# y_train_dummy = y_train_dummy.iloc[:, 1]
# # 设定10折交叉验证的参数格网和分层随机分组，并将其传入GridSearchCV()
# param_grid = {'max_features': range(1, 13)}
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# model = RandomizedSearchCV(RandomForestClassifier(n_estimators=500, random_state=123), param_grid, cv=kfold)
# model.fit(x_train, y_train_dummy)
# print('最优max_features参数为：', model.best_params_)


# max_model_score = 0
# x_1 = 0
# y_1 = 0
# # 利用最优参数再进行随机森林估计
# for i in range(100, 500, 10):
#     for j in range(1, 13):
#         model = RandomForestClassifier(n_estimators=i, max_features=j, random_state=123)
#         # print(model)
#         model.fit(x_train, y_train)
#         if model.score(x_test, y_test) >= max_model_score:
#             max_model_score = model.score(x_test, y_test)
#             x_1 = i
#             y_1 = j
#             print('n_estimators=', x_1, 'max_features=', y_1, '时随机森林的测试集预测准确率为：', max_model_score)
# print('n_estimators=',x_1, 'max_features=',y_1, '时随机森林的测试集预测准确率为：',max_model_score)


# model = RandomForestClassifier(n_estimators=500, max_features=8, random_state=123)
# print(model)
# model.fit(x_train, y_train)
# print('随机森林的测试集预测准确率为：', model.score(x_test, y_test))
#
# pred = model.predict(x_test)  # 预测测试集
# print(pred)
#
# table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])  # 混淆矩阵
# print(table)
#
# print(cohen_kappa_score(y_test, pred))  # 卡帕系数

