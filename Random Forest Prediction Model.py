from joblib import load
import numpy as np
import pandas as pd

# 加载训练好的模型和标准化器
best_model = load(r'C:\Users\LiangHuan\Desktop\全部\BEQbio-yes\best_model.joblib')  # 加载训练好的最佳模型
scaler_X = load(r'C:\Users\LiangHuan\Desktop\全部\BEQbio-yes\scaler_X.joblib')  # 加载输入特征的标准化器
scaler_y = load(r'C:\Users\LiangHuan\Desktop\全部\BEQbio-yes\scaler_y.joblib')  # 加载目标变量的标准化器

print("模型和标准化器已加载！")

# 假设你有新的数据需要预测
new_data = pd.read_excel(r"C:\Users\LiangHuan\Desktop\重要数据\yes predicted_results.xlsx", sheet_name="Sheet1")

# 选择特征列（确保与训练时的特征列一致）
X_new = new_data.iloc[:, 3:11]  # 选择特征列，根据你的数据调整列索引

# 对新数据进行标准化（与训练时一致）
X_new_scaled = scaler_X.transform(np.array(X_new))

# 使用训练好的模型进行预测
y_pred_scaled = best_model.predict(X_new_scaled)

# 反归一化预测结果，恢复到原始的数值范围
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# 将预测结果与新数据合并，形成 DataFrame
result_df = new_data.copy()  # 复制新数据，以便附加预测结果
result_df['Predicted_Target'] = y_pred  # 添加预测目标变量列

# 保存结果到 Excel 文件
output_file_path = r"C:\Users\LiangHuan\Desktop\yespredicted_results.xlsx"
result_df.to_excel(output_file_path, index=False)

print(f"预测结果已保存到 Excel 文件: {output_file_path}")
