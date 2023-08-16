import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import inv, eig
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import plot_partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import *
from osgeo import gdal
import os
import joblib
from sklearn.feature_selection import *
from numpy import set_printoptions

Feature_desertification = pd.read_excel('sum.xlsx')  # 读入数据集

print('数据集的形状为：',Feature_desertification.shape)  # 数据集形状
print(Feature_desertification.head(10),None)  # 显示数据集前两行
print('Class的取值分布为：',Feature_desertification.Class.value_counts())  # 显示响应变量Class的取值分布

# print(Feature_desertification.corr)
# 取出数据矩阵x与响应变量y
x = Feature_desertification.iloc[:, :-1]
y = Feature_desertification.iloc[:, -1]

# 变量间的相关性可视化
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
  # 求相关系数
corr = x.corr()
print(corr)
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots()
    ax = sns.heatmap(corr, mask=mask, vmax=1, vmin=-1, cmap='YlGnBu', annot=True, fmt=".3f",linewidths=.5)
# sns.heatmap(x.corr(), cmap='YlGnBu',annot=True, fmt=".3f",linewidths=.5)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# # SelectKBest法特征选定
# test = SelectKBest(score_func=chi2, k=10)
# fit = test.fit(x, y)
# set_printoptions(precision=3)
# print(fit.scores_)
# # features = fit.transform(x)
# # print(features)
#
# df_feature_importance = pd.DataFrame(fit.scores_)
# df_feature_importance.index = list(Feature_desertification.columns[0:17])
# df_feature_importance.columns = ['importance']
# df_feature_importance2 = df_feature_importance.sort_values(by='importance')  # 按照变量importance升序排列
# print(df_feature_importance2)
# sorted_index = fit.scores_.argsort()
# plt.barh(range(x.shape[1]), fit.scores_[sorted_index])
# plt.yticks(np.arange(x.shape[1]), x.columns[sorted_index])
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Random Forest')
# plt.show()

# 特征重要性特征选定（基于随机森林模型）

# 分层随机抽样，随机抽取50个观测值作为测试集，而其余观测值构成训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=500, random_state=1)

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

# # # 随机森林估计
# # model = RandomForestClassifier(n_estimators=150, max_features='sqrt', random_state=123)
# # model.fit(x_train, y_train)
# # print('随机森林的测试集预测准确率为：', model.score(x_test, y_test))
# # #
# # 寻求最优超参数max_features
# # 将响应变量变为虚拟变量
# y_train_dummy = pd.get_dummies(y_train)
# y_train_dummy = y_train_dummy.iloc[:, 1]
# # 设定10折交叉验证的参数格网和分层随机分组，并将其传入GridSearchCV()
# param_grid = {'max_features': range(1, 10)}
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# model = GridSearchCV(RandomForestClassifier(n_estimators=150, random_state=123), param_grid, cv=kfold)
# model.fit(x_train, y_train_dummy)
# print('最优max_features参数为：', model.best_params_)
#
max_model_score = 0
x_1 = 0
y_1 = 0
# # 利用最优参数再进行随机森林估计
# for i in range(100, 500, 10):
#     for j in range(1, 18):
#         model = RandomForestClassifier(n_estimators=i, max_features=j, random_state=123)
#         # print(model)
#         model.fit(x_train, y_train)
#         if model.score(x_test, y_test) >= max_model_score:
#             max_model_score = model.score(x_test, y_test)
#             x_1 = i
#             y_1 = j
#             print('n_estimators=', x_1, 'max_features=', y_1, '时随机森林的测试集预测准确率为：', max_model_score)
# print('n_estimators=',x_1, 'max_features=',y_1, '时随机森林的测试集预测准确率为：',max_model_score)
# #
#
# 利用最优参数再进行随机森林估计
model = RandomForestClassifier(n_estimators=150, max_features=7, random_state=123)
print(model)
model.fit(x_train, y_train)
print('随机森林的测试集预测准确率为：', model.score(x_test, y_test))

# joblib.dump(model, 'save/model.pkl', compress=3)
# model3 = joblib.load('save/model.pkl')

result = permutation_importance(model, x_train, y_train, n_repeats=10, random_state=123)

# 按照变量次序输出变量重要性
df_feature_importance = pd.DataFrame(model.feature_importances_)
df_feature_importance_2 = pd.DataFrame(result.importances_mean)

df_feature_importance.index = list(Feature_desertification.columns[0:17])
df_feature_importance.columns = ['importance']
df_feature_importance2 = df_feature_importance.sort_values(by='importance')  # 按照变量importance升序排列
print(df_feature_importance2)

sorted_index = model.feature_importances_.argsort()
plt.barh(range(x.shape[1]),model.feature_importances_[sorted_index],color='tab:orange')
plt.yticks(np.arange(x.shape[1]), x.columns[sorted_index])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest')
plt.show()

df_feature_importance_2.index = list(Feature_desertification.columns[0:17])
df_feature_importance_2.columns = ['importance']
df_feature_importance2_2 = df_feature_importance_2.sort_values(by='importance')  # 按照变量importance升序排列
print(df_feature_importance2_2)

sorted_index_2 = result.importances_mean.argsort()
plt.barh(range(x.shape[1]),result.importances_mean[sorted_index_2])
plt.yticks(np.arange(x.shape[1]), x.columns[sorted_index_2])
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest')
plt.show()

pred = model.predict(x_test)
print(pred)

table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])
print(table)

print(cohen_kappa_score(y_test,pred))

'''
# 递归特征消除
model_RFE = LogisticRegression()
rfe = RFE(model_RFE,3)
fit_RFE = rfe.fit(x,y)
print(fit_RFE.ranking_)
'''
