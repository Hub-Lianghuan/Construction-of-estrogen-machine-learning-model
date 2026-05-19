import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump
import matplotlib.pyplot as plt
import shap  # 导入 SHAP 库

# 定义误差计算函数
def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def MSE(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def r_2(y_true, y_pred):
    y_mean = np.mean(y_true)
    a, b = 0, 0
    c = np.size(y_true, 0)
    for i in range(c):
        a += (y_true[i] - y_pred[i]) * (y_true[i] - y_pred[i])
        b += (y_true[i] - y_mean) * (y_true[i] - y_mean)
    result = 1 - a / b
    return result

def r_2_Adjusted(r_2, y_true, n_characteristic):
    n = np.size(y_true)
    r_2_Adjusted = 1 - (1 - r_2 ** 2) * n / (n - 1 - n_characteristic)
    return r_2_Adjusted

# 1. 加载数据
data = pd.read_excel(r"C:\Users\LiangHuan\Desktop\特征数据2.xls", sheet_name='1EE2（最终用的这个输入）')
x = data.iloc[:, 4:13]  # 选择特征列
y = data.iloc[:, -2]

# 获取特征名（列名）
column_indices_list = x.columns.tolist()

# 将数据转换为 NumPy 数组
x = np.array(x)
y = np.array(y).reshape(-1, 1)

y = pd.DataFrame(y).fillna(pd.DataFrame(y).median()).values

# 2. 数据预处理：对输入特征进行标准化，目标变量进行归一化
scaler_X = StandardScaler().fit(x)
X_scaled = scaler_X.transform(x)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

# 3. 分割数据集
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=2024)

# 4. 创建随机森林模型并设置已知的超参数
model = RandomForestRegressor(n_estimators=2000, max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features=0.5, random_state=42)

# 5. 创建多输出回归器
multioutput_model = MultiOutputRegressor(model)

# 6. 使用cross_val_score进行10倍交叉验证
cv_scores = cross_val_score(multioutput_model, X_scaled, y_scaled, cv=10, scoring='neg_mean_squared_error')

# 输出交叉验证的平均分数
print(f"Mean CV Score (Negative MSE): {np.mean(cv_scores)}")

# 7. 训练最终模型
multioutput_model.fit(X_train_scaled, y_train_scaled)

# 8. 预测和评估模型
y_train_pred_scaled = multioutput_model.predict(X_train_scaled)
y_test_pred_scaled = multioutput_model.predict(X_test_scaled)

# 9. 计算最终模型误差指标
result_rmse = RMSE(y_test_scaled, y_test_pred_scaled)
result_mse = MSE(y_test_scaled, y_test_pred_scaled)
result_r2_train = r_2(y_train_scaled, y_train_pred_scaled)
result_r2_test = r_2(y_test_scaled, y_test_pred_scaled)
result_r2_adjusted = r_2_Adjusted(result_r2_test, y_train_pred_scaled, x.shape[1])

print(f"Final RMSE: {result_rmse}")
print(f"Final MSE: {result_mse}")
print(f"Final R² Train: {result_r2_train}")
print(f"Final R² Test: {result_r2_test}")
print(f"Final R² Adjusted: {result_r2_adjusted}")

# 10. 保存模型和标准化器
dump(multioutput_model, r'C:\Users\LiangHuan\Desktop\全部\EE2\best_model.joblib')  # 保存训练好的模型
dump(scaler_X, r'C:\Users\LiangHuan\Desktop\全部\EE2\scaler_X.joblib')  # 保存输入特征的标准化器
dump(scaler_y, r'C:\Users\LiangHuan\Desktop\全部\EE2\scaler_y.joblib')  # 保存目标变量的标准化器

print("模型和标准化器已保存！")

# 11. 计算特征重要性（SHAP值）
explainer = shap.TreeExplainer(multioutput_model.estimators_[0])  # 如果是多输出回归器，使用第一个子模型
shap_values = explainer.shap_values(X_train_scaled)

# 计算每个特征的平均SHAP值（特征的重要性）
shap_importance = np.mean(np.abs(shap_values), axis=0)

# 将特征名与重要性组合
feature_importance = pd.DataFrame(shap_importance, index=column_indices_list, columns=["importance"])

# 计算特征占比（每个特征的重要性占总重要性的比例）
total_importance = feature_importance["importance"].sum()
feature_importance["percentage"] = (feature_importance["importance"] / total_importance) * 100

# 按占比排序特征
feature_importance = feature_importance.sort_values("percentage", ascending=False)

# 输出特征占比数据
print("Feature Importance (percentage):")
print(feature_importance)

# 12. 可视化预测图
for i in range(y_scaled.shape[1]):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train_pred_scaled[:, i], y_train_scaled[:, i], color='blue', label='Train Points', s=3)
    plt.scatter(y_test_pred_scaled[:, i], y_test_scaled[:, i], color='red', label='Test Points', s=3)
    plt.plot([0, 1], [0, 1], 'k--', lw=4, label='Ideal Line')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel(f'Predicted Value (normalized) for Target {i + 1}')
    plt.ylabel(f'Experimental Value (normalized) for Target {i + 1}')
    plt.title(f'Prediction for Target {i + 1}')
    plt.legend()
    plt.show()
