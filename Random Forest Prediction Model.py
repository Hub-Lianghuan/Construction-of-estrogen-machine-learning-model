from joblib import load
import numpy as np
import pandas as pd

best_model = load(r'C:\Users\LiangHuan\Desktop\全部\BEQbio-yes\best_model.joblib')  
scaler_X = load(r'C:\Users\LiangHuan\Desktop\全部\BEQbio-yes\scaler_X.joblib')  
scaler_y = load(r'C:\Users\LiangHuan\Desktop\全部\BEQbio-yes\scaler_y.joblib')  

print("模型和标准化器已加载！")


new_data = pd.read_excel(r"C:\Users\LiangHuan\Desktop\重要数据\yes predicted_results.xlsx", sheet_name="Sheet1")


X_new = new_data.iloc[:, 3:11]  


X_new_scaled = scaler_X.transform(np.array(X_new))


y_pred_scaled = best_model.predict(X_new_scaled)


y_pred = scaler_y.inverse_transform(y_pred_scaled)


result_df = new_data.copy()  
result_df['Predicted_Target'] = y_pred  


output_file_path = r"C:\Users\LiangHuan\Desktop\yespredicted_results.xlsx"
result_df.to_excel(output_file_path, index=False)

print(f"预测结果已保存到 Excel 文件: {output_file_path}")
