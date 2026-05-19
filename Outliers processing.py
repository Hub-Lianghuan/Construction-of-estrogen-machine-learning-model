# 加载所需的库
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from pyod.models.hbos import HBOS
# 数据生成
data = pd.read_excel(r"C:\Users\MarsTao\Desktop\model_wu\修改4种重金属（含因子、浓度、风险）.xlsx",sheet_name='Sheet1')
x = []
for i in range(data.shape[0]):
    list = data.iloc[i,3:-2].to_list()#左闭右开
    x.append(list)
# LOF（不看正负）
# 模型参数设置
clf = LOF(n_neighbors=20, contamination=0.1)

# 用训练的模型，预测自己
y_pred = clf.fit_predict(x)  # 返回异常值
X_scores = clf.negative_outlier_factor_  # 返回异常分数LOF
scores = np.array(X_scores).reshape((data.shape[0],1))
scaler_lof = StandardScaler()
lof_scaled = scaler_lof.fit_transform(scores)

# T_F = np.array(y_pred).reshape((data.shape[0],1))
# result_lof = np.concatenate((scores,T_F),axis=1)#axis=0水平拼接，axis=1垂直拼接

# 孤立森林（越小越异常,乘-1）
# 创建孤立森林模型并训练
model = IsolationForest(contamination=0.1)
model.fit(x)

# 使用模型来预测异常值
outliers = model.predict(x)
# 显示异常得分
IS_scores = np.array(model.decision_function(x)).reshape(-1,1)*(-1)
scaler_is = StandardScaler()
is_scaled = scaler_lof.fit_transform(IS_scores)
# print(outliers)
# outliers 的值为 1 表示正常样本，-1 表示异常值

# 最小协方差矩阵行列式（MCD）（不看正负）
def calculate_mahalanobis_score(data):
    # 计算均值向量和协方差矩阵
    mean = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)

    # 计算协方差矩阵的逆矩阵
    inv_cov_matrix = np.linalg.inv(cov_matrix)

    # 计算每个数据点的马氏距离
    mahalanobis_scores = []
    for point in data:
        diff = point - mean
        raw_distance = np.dot(np.dot(diff, inv_cov_matrix), diff.T)
        mahalanobis_distance = np.sqrt(np.maximum(0, raw_distance))
        mahalanobis_scores.append(mahalanobis_distance)
    return mahalanobis_scores

# 计算异常值得分
MCD_scores = calculate_mahalanobis_score(x)
MCD_scores=np.array(MCD_scores).reshape(-1,1)
scaler_MCD = StandardScaler()
mcd_scaled = scaler_lof.fit_transform(MCD_scores)

# KNN（不看正负）
# 创建KNN模型
knn = NearestNeighbors(n_neighbors=5)
# 用数据拟合模型
knn.fit(x)
# 计算每个数据点到其最近的5个邻居的距离
distances, indices = knn.kneighbors(x)
# 计算每个数据点到其最近的5个邻居的平均距离
avg_distances = distances.mean(axis=1)
knn_scores = np.array(avg_distances).reshape(-1,1)
scaler_knn = StandardScaler()
knn_scaled = scaler_lof.fit_transform(knn_scores)

# HBOS（不看正负）


# 初始化HBOS模型
clf = HBOS(n_bins=10, alpha=0.1, tol=0.5, contamination=0.1)

# 训练模型
clf.fit(x)


# 绘制直方图
df = pd.DataFrame(
    np.array(x),
    # columns=['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
)

# for column in df.columns:
#     plt.hist(df[column], bins=30, edgecolor='black')
#     plt.title(f'{column} Histogram')
#     plt.xlabel('Feature Values')
#     plt.ylabel('Frequency')
#     plt.show()

# 预测测试集的异常值得分
y_test_pred = clf.decision_function(x)  # outlier scores
hbos_scores = np.array(y_test_pred).reshape(-1,1)
scaler_hbos = StandardScaler()
hbos_scaled = scaler_lof.fit_transform(hbos_scores)

result_ = np.concatenate((lof_scaled,knn_scaled,is_scaled,hbos_scaled,mcd_scaled),axis=1)
total=np.mean(result_,axis=1).reshape(-1,1)
result = np.concatenate((result_,total),axis=1)

# 列标签
header = ['lof_score','knn_score','if_score','hbos_score','mcd_score','total_score']

# 将列标签保存为字符串，并用逗号分隔
header_str = ','.join(header)

# 先将列标签写入文件
with open(r"C:\Users\MarsTao\Desktop\model_wu\valid.csv", 'w') as f:
    f.write(header_str)
    f.write('\n')
with open(r"C:\Users\MarsTao\Desktop\model_wu\valid.csv", 'w') as f:
    np.savetxt(f, result, delimiter=',')
