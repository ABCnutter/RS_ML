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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import plot_partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import *
from osgeo import gdal
from osgeo import osr
import os
import joblib
from sklearn.feature_selection import *
from numpy import set_printoptions

Feature_desertification = pd.read_excel('Classification.xlsx')  # 读入数据集

driver = gdal.GetDriverByName('ENVI')

rds = gdal.Open(r'E:\RS_source\NingDong_Data_2\STACK\stack_n\2\2021_stack_2.dat')

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
# sns.heatmap(x.corr(), cmap='Blues')
# plt.title('Correlation Matrix')
# plt.tight_layout()
# plt.show()


# 分层随机抽样，随机抽取50个观测值作为测试集，而其余观测值构成训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=500, random_state=1)

# # 随机森林估计
# model = RandomForestClassifier(n_estimators=150, max_features=3, random_state=123)
# model.fit(x_train, y_train)
# print('随机森林的测试集预测准确率为：', model.score(x_test, y_test))


# # 寻求最优超参数max_features
# # 将响应变量变为虚拟变量
# y_train_dummy = pd.get_dummies(y_train)
# y_train_dummy = y_train_dummy.iloc[:, 1]
# # 设定10折交叉验证的参数格网和分层随机分组，并将其传入GridSearchCV()
# param_grid = {'max_features': range(1, 7)}
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
# model = GridSearchCV(RandomForestClassifier(n_estimators=81, random_state=123), param_grid, cv=kfold)
# model.fit(x_train, y_train_dummy)
# print('最优max_features参数为：', model.best_params_)


# max_model_score = 0
# x_1 = 0
# y_1 = 0
# # 利用最优参数再进行随机森林估计
# for i in range(150, 165, 3):
#     for j in range(1, 15):
#         model = RandomForestClassifier(n_estimators=i, max_features=j, random_state=123)
#         # print(model)
#         model.fit(x_train, y_train)
#         if model.score(x_test, y_test) >= max_model_score:
#             max_model_score = model.score(x_test, y_test)
#             x_1 = i
#             y_1 = j
#             print('n_estimators=', x_1, 'max_features=', y_1, '时随机森林的测试集预测准确率为：', max_model_score)
# print('n_estimators=',x_1, 'max_features=',y_1, '时随机森林的测试集预测准确率为：',max_model_score)


# model = RandomForestClassifier(n_estimators=81, max_features=2, random_state=123)
# print(model)
# model.fit(x_train, y_train)
# print('随机森林的测试集预测准确率为：', model.score(x_test, y_test))

model = GradientBoostingClassifier(subsample=1.0, n_estimators=169, max_depth=2, learning_rate=0.6, random_state=123)
print(model)
model.fit(x_train, y_train)
print('GBM的测试集预测准确率为：', model.score(x_test, y_test))

# df_feature_importance = pd.DataFrame(model.feature_importances_)
#
# df_feature_importance.index = list(Feature_desertification.columns[0:6])
# df_feature_importance.columns = ['importance']
# df_feature_importance2 = df_feature_importance.sort_values(by='importance')  # 按照变量importance升序排列
# print(df_feature_importance2)
#
# sorted_index = model.feature_importances_.argsort()
# plt.barh(range(x.shape[1]),model.feature_importances_[sorted_index])
# plt.yticks(np.arange(x.shape[1]), x.columns[sorted_index])
# plt.xlabel('Feature Importance')
# plt.ylabel('Feature')
# plt.title('Random Forest')
# plt.show()

# # 保存模型
# # model_file = 'finalized_model_joblib_RF.sav'
# # with open(model_file,'wb') as model_f_RF:
# #     joblib.dump(model,model_f_RF)
# # 加载模型
# # with open(model_file,'rb') as model_f_RF:
# #     joblib.load(model_f_RF)

pred = model.predict(x_test)  # 预测测试集
print(pred)

table = pd.crosstab(y_test, pred, rownames=['Actual'], colnames=['Predicted'])  # 混淆矩阵
print(table)

print(cohen_kappa_score(y_test, pred))  # 卡帕系数


Descripition = rds.GetDescription()  # 描述信息
Band_Counts = rds.RasterCount  # 波段数
X_Size = rds.RasterXSize  # X方向的像元个数
Y_Size = rds.RasterYSize  # Y方向的像元个数
Six_Transform = rds.GetGeoTransform()  # 六参数
Projection = rds.GetProjection()  # 投影
# print(Descripition)
print('波段数：', Band_Counts)
print('x方向的像元个数：', X_Size)
print('y方向的像元个数：', Y_Size)
print(Six_Transform)
# print(Projection)

# Metadata = rds.GetMetadata
# print(Metadata)

# 获取遥感影像数据集波段
Band_1 = rds.GetRasterBand(1)
Band_2 = rds.GetRasterBand(2)
Band_3 = rds.GetRasterBand(3)
Band_4 = rds.GetRasterBand(4)
Band_5 = rds.GetRasterBand(5)
Band_6 = rds.GetRasterBand(6)
# Band_7 = rds.GetRasterBand(7)
# Band_8 = rds.GetRasterBand(8)
# Band_9 = rds.GetRasterBand(9)
# Band_10 = rds.GetRasterBand(10)
# Band_11 = rds.GetRasterBand(11)
# Band_12 = rds.GetRasterBand(12)
# Band_13 = rds.GetRasterBand(13)
# Band_14 = rds.GetRasterBand(14)

# 查看波段的基本信息
print(dir(Band_1))

# 获取波段的行列数目
print(Band_1.XSize)
print(Band_1.YSize)

# 获取波段的数据类型
print(Band_1.DataType)

# 获取波段的属性
print(Band_1.GetNoDataValue())
print(Band_1.GetMaximum())
print(Band_1.GetMinimum())
print(Band_1.ComputeRasterMinMax())

# 获取波段中的数据
Arr_1_1 = Band_1.ReadAsArray(0, 0, X_Size, Y_Size)
Arr_2_1 = Band_2.ReadAsArray(0, 0, X_Size, Y_Size)
Arr_3_1 = Band_3.ReadAsArray(0, 0, X_Size, Y_Size)
Arr_4_1 = Band_4.ReadAsArray(0, 0, X_Size, Y_Size)
Arr_5_1 = Band_5.ReadAsArray(0, 0, X_Size, Y_Size)
Arr_6_1 = Band_6.ReadAsArray(0, 0, X_Size, Y_Size)
# Arr_7_1 = Band_7.ReadAsArray(0, 0, X_Size, Y_Size)
# Arr_8_1 = Band_8.ReadAsArray(0, 0, X_Size, Y_Size)
# Arr_9_1 = Band_9.ReadAsArray(0, 0, X_Size, Y_Size)
# Arr_10_1 = Band_10.ReadAsArray(0, 0, X_Size, Y_Size)
# Arr_11_1 = Band_11.ReadAsArray(0, 0, X_Size, Y_Size)
# Arr_12_1 = Band_12.ReadAsArray(0, 0, X_Size, Y_Size)
# Arr_13_1 = Band_13.ReadAsArray(0, 0, X_Size, Y_Size)
# Arr_14_1 = Band_14.ReadAsArray(0, 0, X_Size, Y_Size)

Arr_1_2 = np.reshape(Arr_1_1, (-1, 1))
Arr_2_2 = np.reshape(Arr_2_1, (-1, 1))
Arr_3_2 = np.reshape(Arr_3_1, (-1, 1))
Arr_4_2 = np.reshape(Arr_4_1, (-1, 1))
Arr_5_2 = np.reshape(Arr_5_1, (-1, 1))
Arr_6_2 = np.reshape(Arr_6_1, (-1, 1))
# Arr_7_2 = np.reshape(Arr_7_1, (-1, 1))
# Arr_8_2 = np.reshape(Arr_8_1, (-1, 1))
# Arr_9_2 = np.reshape(Arr_9_1, (-1, 1))
# Arr_10_2 = np.reshape(Arr_10_1, (-1, 1))
# Arr_11_2 = np.reshape(Arr_11_1, (-1, 1))
# Arr_12_2 = np.reshape(Arr_12_1, (-1, 1))
# Arr_13_2 = np.reshape(Arr_13_1, (-1, 1))
# Arr_14_2 = np.reshape(Arr_14_1, (-1, 1))

Arr_sum = np.hstack([Arr_1_2, Arr_2_2, Arr_3_2, Arr_4_2, Arr_5_2, Arr_6_2])
# Arr_7_2,Arr_8_2, Arr_9_2, Arr_10_2, Arr_11_2, Arr_12_2,Arr_13_2,Arr_14_2

# # print(Arr_sum)

data_sum = pd.DataFrame(Arr_sum.astype('float64'), columns=['NDVI', 'BSI', 'Brightness', 'TGSI', 'Mean', 'Wetness'],
                        index=None)

# data_sum = pd.DataFrame(Arr_sum.astype('float64'),
#                         columns=['MSAVI','Albedo ','TGSI','Wetness','Contrast', 'Correlation',
#                                                             'Dissmimilarity','Entropy','Homogeneity','Mean','Second Moment',
#                                                             'Variance','DEM','Slope'],index=None)

data_sum_2 = data_sum.fillna(0)
print(data_sum_2.head(10))
data_len = len(data_sum_2.index)
print(data_len)

pred_2 = []
for i in range(0, data_len, 162):
    pred_1 = model.predict(data_sum_2.iloc[i:i + 162, :])
    pred_2 = np.concatenate([pred_2, pred_1])

pred_3 = np.reshape(pred_2, (X_Size, Y_Size))
print(pred_3.dtype)

# #
# # # with open(r'E:\PYTHONS\NEW\Classification\2017\2017_output_finally_2.csv', 'w', newline='') as csvfile:
# # #     writer = csv.writer(csvfile)
# # #     for row in pred_3:
# # #         writer.writerow(row)
# # #
# # # # # results = []
# # # # # with open("output.csv") as csvfile:
# # # # #     reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
# # # # #     for row in reader:
# # # # #         results.append(row)
# # # # # a = np.reshape(results,(X_Size, Y_Size))
# # # # # print(a.shape)
# # # # # print(a.dtype)
# # # # # # a[a == 0] = np.nan
# # # # # print(a[:2,:3])
# # # # data_csv = pd.read_csv(r'E:\PYTHONS\NEW\Classification\output.csv',header=None)
# # #

dst_filename = r'E:\RS_source\NingDong_Data_2\13_分类结果\class\2021_class_GBM.dat'
tods = driver.Create(dst_filename, X_Size, Y_Size, 1, gdal.GDT_Float64)

tods.SetGeoTransform([619095, 30, 0, 4239045, 0, -30])
# srs = osr.SpatialReference()
# srs.SetUTM(11,1)
# srs.SetWellKnownGeogCS("WGS84")
# tods.SetProjection(srs.ExportToWkt())

tods.WriteRaster(0, 0, X_Size, Y_Size, pred_3.tobytes(), band_list=[1])
tods.FlushCache()
