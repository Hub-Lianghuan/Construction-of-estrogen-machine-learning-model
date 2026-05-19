import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump
import matplotlib.pyplot as plt
import shap 


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

data = pd.read_excel(r"C:\Users\LiangHuan\Desktop\原始数据合并版本1987.xls", sheet_name='BEQbio(ER-calux）提取s')
x = data.iloc[:, 4:13]  
y = data.iloc[:, -3]


column_indices_list = x.columns.tolist()

x = np.array(x)
y = np.array(y).reshape(-1, 1)

y = pd.DataFrame(y).fillna(pd.DataFrame(y).median()).values

scaler_X = StandardScaler().fit(x)
X_scaled = scaler_X.transform(x)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=2024)

model = RandomForestRegressor(n_estimators=2500, max_depth=20, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', random_state=42)

multioutput_model = MultiOutputRegressor(model)

cv_scores = cross_val_score(multioutput_model, X_scaled, y_scaled, cv=10, scoring='neg_mean_squared_error')

print(f"Mean CV Score (Negative MSE): {np.mean(cv_scores)}")

multioutput_model.fit(X_train_scaled, y_train_scaled)

y_train_pred_scaled = multioutput_model.predict(X_train_scaled)
y_test_pred_scaled = multioutput_model.predict(X_test_scaled)

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

dump(multioutput_model, r'C:\Users\LiangHuan\Desktop\全部\BEQ-calux\best_model.joblib')  
dump(scaler_X, r'C:\Users\LiangHuan\Desktop\全部\BEQ-calux\scaler_X.joblib')  # 
dump(scaler_y, r'C:\Users\LiangHuan\Desktop\全部\BEQ-calux\scaler_y.joblib')  # 

print("模型和标准化器已保存！")

explainer = shap.TreeExplainer(multioutput_model.estimators_[0])  # 
shap_values = explainer.shap_values(X_train_scaled)


shap_importance = np.mean(np.abs(shap_values), axis=0)


feature_importance = pd.DataFrame(shap_importance, index=column_indices_list, columns=["importance"])


total_importance = feature_importance["importance"].sum()
feature_importance["percentage"] = (feature_importance["importance"] / total_importance) * 100


feature_importance = feature_importance.sort_values("percentage", ascending=False)


print("Feature Importance (percentage):")
print(feature_importance)


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
