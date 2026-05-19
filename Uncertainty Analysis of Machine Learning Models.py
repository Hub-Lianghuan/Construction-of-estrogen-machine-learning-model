import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

# 设置新罗马字体（Times New Roman）用于英文显示
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
# 添加数学字体设置，确保科学计数法使用新罗马字体
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX字体，与新罗马相似
plt.rcParams['mathtext.rm'] = 'Times New Roman'  # 设置数学字体中的罗马体
# 1. 加载数据
file_path = r'C:\Users\LiangHuan\Desktop\原始数据、特征数据\原始数据合并版本1987.xls'

try:
    # 尝试不同引擎读取xls文件
    try:
        df = pd.read_excel(file_path, sheet_name='BEQbio(ER-calux）提取s', engine='xlrd')
    except:
        df = pd.read_excel(file_path, sheet_name='BEQbio(ER-calux）提取s', engine='openpyxl')

    print(f"Data loaded successfully, shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")

except Exception as e:
    print(f"Failed to load data, error: {e}")
    exit()

# 2. 准备数据
X = df.iloc[:, 4:14]  # 第4列到13列（索引3到12）
y = df.iloc[:, -2]  # 倒数第二列

print(f"Feature data shape: {X.shape}")
print(f"Target variable shape: {y.shape}")

# 检查是否有缺失值
if X.isnull().any().any() or y.isnull().any():
    print("Missing values detected, processing...")
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

# 使用全部数据进行训练和测试
X_train, y_train = X, y
X_test, y_test = X, y  # 使用全部数据作为测试集

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# 4. 训练随机森林模型
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
rf_model.fit(X_train, y_train)


# 5. 使用自助法（Bootstrap）计算置信区间
def calculate_confidence_intervals(X_data, y_data, model, n_bootstrap=100, alpha=0.05):
    """使用自助法计算预测值的置信区间"""
    predictions = []

    for i in range(n_bootstrap):
        # 自助采样
        X_resample, y_resample = resample(X_data, y_data, random_state=i)

        # 训练新模型
        temp_model = RandomForestRegressor(
            n_estimators=50,  # 减少树的数量以加快计算
            random_state=i,
            n_jobs=-1
        )
        temp_model.fit(X_resample, y_resample)

        # 在原始测试集上预测
        pred = temp_model.predict(X_test)
        predictions.append(pred)

    # 转换为数组
    predictions = np.array(predictions)  # 形状：(n_bootstrap, n_samples)

    # 计算均值和置信区间
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)

    # 计算置信区间（使用正态分布近似）
    z_score = 1.96  # 95%置信水平的z值
    lower_bound = mean_predictions - z_score * std_predictions
    upper_bound = mean_predictions + z_score * std_predictions

    return mean_predictions, lower_bound, upper_bound, predictions


# 6. 计算置信区间
print("Calculating confidence intervals...")
mean_preds, lower_bounds, upper_bounds, all_predictions = calculate_confidence_intervals(
    X_train, y_train, rf_model, n_bootstrap=100
)

# 7. 绘制置信区间图
plt.figure(figsize=(14, 8))

# 对测试集样本按真实值排序以便更好地可视化
sorted_indices = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_indices]
mean_preds_sorted = mean_preds[sorted_indices]
lower_bounds_sorted = lower_bounds[sorted_indices]
upper_bounds_sorted = upper_bounds[sorted_indices]

# 创建样本索引
x_index = np.arange(len(y_test_sorted))

# 绘制置信区间带
plt.fill_between(
    x_index,
    lower_bounds_sorted,
    upper_bounds_sorted,
    alpha=0.3,
    color='steelblue',
    label='95% Confidence Interval'
)

# 绘制预测均值线
plt.plot(x_index, mean_preds_sorted, 'b-', linewidth=2, label='Predicted Mean')

# 绘制真实值
plt.scatter(x_index, y_test_sorted, color='red', alpha=0.6, s=10, label='True Values', zorder=5)

# 添加标签和标题
plt.xlabel('Test Samples (Sorted by True Values)', fontsize=20,fontweight='bold')
plt.ylabel('Predicted Values', fontsize=20,fontweight='bold')

plt.legend(loc='best', fontsize=20)

# 去掉网格
plt.grid(False)

# 设置x轴刻度为整百显示
n_samples = len(y_test_sorted)
if n_samples > 100:
    tick_interval = 200
    xticks = np.arange(0, n_samples, tick_interval)
else:
    xticks = np.arange(0, n_samples, max(1, n_samples // 10))

plt.xticks(xticks, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')

# 设置标签字体
plt.xlabel('Test Samples (Sorted by True Values)', fontsize=20, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=20, fontweight='bold')


# 设置图框粗细
ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

plt.grid(False)

# 自定义科学计数法格式化函数
def scientific_format(x, pos):
    if x == 0:
        return '0'
    exponent = int(np.floor(np.log10(abs(x))))
    coefficient = x / (10**exponent)
    if abs(coefficient) < 1:
        coefficient *= 10
        exponent -= 1
    if exponent == 0:
        return f'{coefficient:.1f}'
    else:
        return f'{coefficient:.1f}×10$^{{{exponent}}}$'

# 应用自定义格式化器到y轴
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(scientific_format))

plt.tight_layout()

# 8. 保存图像
output_path = r'C:\Users\LiangHuan\Desktop\confidence_interval_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Confidence interval plot saved to: {output_path}")

# 显示图像
plt.show()

# 9. 计算不确定性评价指标
print("\n" + "=" * 60)
print("Model Uncertainty Analysis Metrics")
print("=" * 60)

# 转换为 numpy 数组，避免 pandas 索引影响计算
y_true = y_test.values.astype(float)
y_pred = mean_preds.astype(float)
lower = lower_bounds.astype(float)
upper = upper_bounds.astype(float)

# 预测区间宽度
interval_width = upper - lower

# 1. PICP: Prediction Interval Coverage Probability
# 真实值落入预测区间的比例
within_interval = (y_true >= lower) & (y_true <= upper)
picp = np.mean(within_interval)

# 2. MPIW: Mean Prediction Interval Width
# 平均预测区间宽度
mpiw = np.mean(interval_width)

# 3. PINAW: Prediction Interval Normalized Average Width
# 归一化平均预测区间宽度
y_range = np.max(y_true) - np.min(y_true)

if y_range == 0:
    pinaw = np.nan
else:
    pinaw = mpiw / y_range

# 4. MIS: Mean Interval Score
# 95% 预测区间对应 alpha = 0.05
alpha = 0.05

lower_penalty = (2 / alpha) * (lower - y_true) * (y_true < lower)
upper_penalty = (2 / alpha) * (y_true - upper) * (y_true > upper)

interval_score = interval_width + lower_penalty + upper_penalty
mis = np.mean(interval_score)

# 5. 基础预测误差指标，可选，但建议保留
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

if ss_tot == 0:
    r2 = np.nan
else:
    r2 = 1 - ss_res / ss_tot

# 打印结果
print(f"Number of samples: {len(y_true)}")
print(f"R2: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

print("-" * 60)
print(f"PICP_95: {picp:.4f} ({picp * 100:.2f}%)")
print(f"MPIW: {mpiw:.4f}")
print(f"PINAW: {pinaw:.4f}")
print(f"MIS: {mis:.4f}")

print("-" * 60)
print(f"Minimum interval width: {np.min(interval_width):.4f}")
print(f"Maximum interval width: {np.max(interval_width):.4f}")
print(f"Predicted mean range: [{np.min(y_pred):.4f}, {np.max(y_pred):.4f}]")

# 10. 可选：创建第二个图，显示预测分布
if len(X_test) <= 20:  # 如果测试样本不多，可以显示每个样本的预测分布
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i, idx in enumerate(sorted_indices[:20]):
        ax = axes[i]
        ax.hist(all_predictions[:, idx], bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(x=y_test.values[idx], color='red', linestyle='--', linewidth=2, label='True Value')
        ax.axvline(x=mean_preds[idx], color='blue', linestyle='-', linewidth=2, label='Predicted Mean')
        ax.set_title(f'Sample {idx}')
        # 为第二个图的x轴也应用科学计数法
        ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
        ax.grid(False)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Random Forest Model Prediction Distribution (First 20 Samples)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存第二个图
    dist_output_path = r'C:\Users\LiangHuan\Desktop\prediction_distribution_plot.png'
    plt.savefig(dist_output_path, dpi=300, bbox_inches='tight')
    print(f"Prediction distribution plot saved to: {dist_output_path}")
    plt.show()

# 11. 保存逐样本预测结果到 Excel
results_df = pd.DataFrame({
    'True_Values': y_true,
    'Predicted_Mean': y_pred,
    'Prediction_Lower_95': lower,
    'Prediction_Upper_95': upper,
    'Prediction_Interval_Width': interval_width,
    'Within_95_Interval': within_interval,
    'Interval_Score': interval_score,
    'Absolute_Error': np.abs(y_true - y_pred),
    'Squared_Error': (y_true - y_pred) ** 2
})

results_output_path = r'C:\Users\LiangHuan\Desktop\prediction_results_uncertainty.xlsx'
results_df.to_excel(results_output_path, index=False)
print(f"Prediction results saved to: {results_output_path}")

results_output_path = r'C:\Users\LiangHuan\Desktop\prediction_results_confidence_intervals.xlsx'
results_df.to_excel(results_output_path, index=False)
print(f"Prediction results saved to: {results_output_path}")