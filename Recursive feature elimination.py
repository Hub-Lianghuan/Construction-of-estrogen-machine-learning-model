"""
RFE特征选择分析 - 简化版（添加训练集和测试集对比）
作者：LiangHuan
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# 配置
FILE_PATH = r'C:\Users\LiangHuan\Desktop\原始数据、特征数据\特征数据2.xls'
SHEET_NAME = 'BEQ-bio(E-SCREEN)'

# 1. 读取数据
print("正在读取数据...")
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
print(f"数据形状: {df.shape}")
print(f"列名: {list(df.columns)}")

# 提取特征和目标变量
# 特征: 第5列到第24列（索引4到23）
# 目标: 倒数第二列
X = df.iloc[:, 4:24]  # 第5-24列
y = df.iloc[:, -2]  # 倒数第二列

print(f"\n特征数量: {X.shape[1]}")
print(f"特征名称: {list(X.columns)}")
print(f"目标变量: {y.name}")

# 处理缺失值
X = X.fillna(X.mean())
y = y.fillna(y.mean())

# 2. 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n数据分割情况:")
print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")

# 3. RFECV自动确定最优特征数量
print("\n" + "=" * 60)
print("RFECV自动确定最优特征数量...")

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

rfecv = RFECV(
    estimator=rf_model,
    step=1,
    cv=5,
    scoring='r2',
    min_features_to_select=3,
    n_jobs=-1
)

rfecv.fit(X_train, y_train)

optimal_n_features = rfecv.n_features_
selected_features_auto = X.columns[rfecv.support_]

print(f"✓ RFECV推荐的最优特征数量: {optimal_n_features}")
print(f"✓ 最优交叉验证R²分数: {rfecv.cv_results_['mean_test_score'].max():.4f}")
print(f"\n自动选中的特征 ({len(selected_features_auto)}个):")
for i, feat in enumerate(selected_features_auto, 1):
    print(f"  {i}. {feat}")

# 4. 手动选择特征数量
print("\n" + "=" * 60)
print("手动特征选择")
print("=" * 60)

# 这里可以修改特征数量
manual_n_features = optimal_n_features  # 使用自动推荐的数量
# manual_n_features = 10  # 如果想手动设置，取消注释并修改数字

print(f"使用特征数量: {manual_n_features}")

# 使用RFE进行特征选择
rfe = RFE(
    estimator=rf_model,
    n_features_to_select=manual_n_features,
    step=1
)

rfe.fit(X_train, y_train)
selected_features_manual = X.columns[rfe.support_]

print(f"\n手动选中的特征 ({len(selected_features_manual)}个):")
for i, feat in enumerate(selected_features_manual, 1):
    print(f"  {i}. {feat}")

# 5. 评估性能（添加训练集和测试集对比）
print("\n" + "=" * 60)
print("性能评估（训练集 vs 测试集）")
print("=" * 60)


# 定义计算性能的函数
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """计算模型在训练集和测试集上的性能"""
    # 训练集预测
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)

    # 测试集预测
    y_test_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    return {
        'train_r2': train_r2,
        'train_mse': train_mse,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'y_test_pred': y_test_pred
    }


# 5.1 使用所有特征训练模型
print("\n1. 使用所有特征:")
rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
rf_full.fit(X_train, y_train)
full_results = evaluate_model(rf_full, X_train, y_train, X_test, y_test)

print(f"   训练集 R²: {full_results['train_r2']:.4f}, MSE: {full_results['train_mse']:.4f}")
print(f"   测试集 R²: {full_results['test_r2']:.4f}, MSE: {full_results['test_mse']:.4f}")
print(f"   训练集-测试集R²差异: {full_results['train_r2'] - full_results['test_r2']:.4f}")

# 5.2 使用自动选择的特征训练模型
print("\n2. 使用自动选择的特征:")
X_train_auto = X_train[selected_features_auto]
X_test_auto = X_test[selected_features_auto]
rf_auto = RandomForestRegressor(n_estimators=100, random_state=42)
rf_auto.fit(X_train_auto, y_train)
auto_results = evaluate_model(rf_auto, X_train_auto, y_train, X_test_auto, y_test)

print(f"   训练集 R²: {auto_results['train_r2']:.4f}, MSE: {auto_results['train_mse']:.4f}")
print(f"   测试集 R²: {auto_results['test_r2']:.4f}, MSE: {auto_results['test_mse']:.4f}")
print(f"   训练集-测试集R²差异: {auto_results['train_r2'] - auto_results['test_r2']:.4f}")

# 5.3 使用手动选择的特征训练模型
print("\n3. 使用手动选择的特征:")
X_train_manual = X_train[selected_features_manual]
X_test_manual = X_test[selected_features_manual]
rf_manual = RandomForestRegressor(n_estimators=100, random_state=42)
rf_manual.fit(X_train_manual, y_train)
manual_results = evaluate_model(rf_manual, X_train_manual, y_train, X_test_manual, y_test)

print(f"   训练集 R²: {manual_results['train_r2']:.4f}, MSE: {manual_results['train_mse']:.4f}")
print(f"   测试集 R²: {manual_results['test_r2']:.4f}, MSE: {manual_results['test_mse']:.4f}")
print(f"   训练集-测试集R²差异: {manual_results['train_r2'] - manual_results['test_r2']:.4f}")

# 5.4 性能对比表格
print("\n" + "=" * 60)
print("性能对比表格")
print("=" * 60)

print(f"{'方法':<20} {'特征数':<10} {'训练集R²':<10} {'训练集MSE':<12} {'测试集R²':<10} {'测试集MSE':<12}")
print("-" * 80)
print(
    f"{'所有特征':<20} {X.shape[1]:<10} {full_results['train_r2']:<10.4f} {full_results['train_mse']:<12.4f} {full_results['test_r2']:<10.4f} {full_results['test_mse']:<12.4f}")
print(
    f"{'自动选择':<20} {len(selected_features_auto):<10} {auto_results['train_r2']:<10.4f} {auto_results['train_mse']:<12.4f} {auto_results['test_r2']:<10.4f} {auto_results['test_mse']:<12.4f}")
print(
    f"{'手动选择':<20} {manual_n_features:<10} {manual_results['train_r2']:<10.4f} {manual_results['train_mse']:<12.4f} {manual_results['test_r2']:<10.4f} {manual_results['test_mse']:<12.4f}")

# 6. 特征排名
print("\n" + "=" * 60)
print("特征排名")
print("=" * 60)

ranking_df = pd.DataFrame({
    '特征名称': X.columns,
    '排名': rfe.ranking_,
    '是否选中': rfe.support_
}).sort_values('排名')

print(ranking_df.to_string(index=False))

# 7. 可视化（添加更多对比）
plt.figure(figsize=(16, 12))

# 子图1: 特征数量与R²关系
plt.subplot(2, 3, 1)
n_features_range = range(1, len(rfecv.cv_results_['mean_test_score']) + 1)
plt.plot(n_features_range, rfecv.cv_results_['mean_test_score'], 'b-', linewidth=2)
plt.fill_between(n_features_range,
                 rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
                 rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
                 alpha=0.2)
plt.axvline(x=optimal_n_features, color='r', linestyle='--',
            label=f'最优特征数: {optimal_n_features}')
plt.xlabel('特征数量')
plt.ylabel('交叉验证R^2分数')
plt.title(
    f'特征数量与模型性能关系\n最优特征数: {optimal_n_features}, R^2={rfecv.cv_results_["mean_test_score"].max():.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2: R²对比（训练集 vs 测试集）
plt.subplot(2, 3, 2)
methods = ['所有特征', '自动选择', '手动选择']
train_r2 = [full_results['train_r2'], auto_results['train_r2'], manual_results['train_r2']]
test_r2 = [full_results['test_r2'], auto_results['test_r2'], manual_results['test_r2']]

x = np.arange(len(methods))
width = 0.35

bars1 = plt.bar(x - width / 2, train_r2, width, label='训练集R²', color='skyblue', edgecolor='black')
bars2 = plt.bar(x + width / 2, test_r2, width, label='测试集R²', color='lightgreen', edgecolor='black')

plt.ylabel('R^2分数')
plt.title('训练集 vs 测试集 R²对比')
plt.xticks(x, methods)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 在柱子上添加数值
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 子图3: MSE对比（训练集 vs 测试集）
plt.subplot(2, 3, 3)
train_mse = [full_results['train_mse'], auto_results['train_mse'], manual_results['train_mse']]
test_mse = [full_results['test_mse'], auto_results['test_mse'], manual_results['test_mse']]

bars3 = plt.bar(x - width / 2, train_mse, width, label='训练集MSE', color='lightcoral', edgecolor='black')
bars4 = plt.bar(x + width / 2, test_mse, width, label='测试集MSE', color='orange', edgecolor='black')

plt.ylabel('MSE')
plt.title('训练集 vs 测试集 MSE对比')
plt.xticks(x, methods)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# 在柱子上添加数值
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# 子图4: 实际vs预测（手动选择 - 测试集）
plt.subplot(2, 3, 4)
plt.scatter(y_test, manual_results['y_test_pred'], alpha=0.6, color='blue')
min_val = min(y_test.min(), manual_results['y_test_pred'].min())
max_val = max(y_test.max(), manual_results['y_test_pred'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想线')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title(f'实际值 vs 预测值 (手动选择-测试集)\nR^2={manual_results["test_r2"]:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图5: 特征数量对比
plt.subplot(2, 3, 5)
sizes = [X.shape[1], len(selected_features_auto), manual_n_features]
labels = [f'原始特征\n{sizes[0]}个', f'自动选择\n{sizes[1]}个', f'手动选择\n{sizes[2]}个']
colors = ['lightgray', 'lightblue', 'lightcoral']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('特征数量对比')

# 子图6: 训练集-测试集R²差异
plt.subplot(2, 3, 6)
r2_diff = [full_results['train_r2'] - full_results['test_r2'],
           auto_results['train_r2'] - auto_results['test_r2'],
           manual_results['train_r2'] - manual_results['test_r2']]

bars_diff = plt.bar(methods, r2_diff, color=['skyblue', 'lightgreen', 'orange'], edgecolor='black')
plt.ylabel('R²差异 (训练集-测试集)')
plt.title('过拟合程度评估\n(差异越小越好)')
plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')

# 在柱子上添加数值
for bar, diff in zip(bars_diff, r2_diff):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + (0.01 if height >= 0 else -0.03),
             f'{diff:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)

plt.suptitle('RFE特征选择分析结果 (训练集 vs 测试集对比)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# 8. 保存结果（添加训练集和测试集指标）
print("\n" + "=" * 60)
print("保存结果")
print("=" * 60)

# 保存选中的特征
selected_df = pd.DataFrame({
    '序号': range(1, len(selected_features_manual) + 1),
    '特征名称': selected_features_manual,
    '排名': ranking_df[ranking_df['是否选中']]['排名'].values
})

# 保存性能对比（包含训练集和测试集）
performance_df = pd.DataFrame({
    '方法': ['所有特征', '自动选择', '手动选择'],
    '特征数量': [X.shape[1], len(selected_features_auto), manual_n_features],
    '训练集_R2': [full_results['train_r2'], auto_results['train_r2'], manual_results['train_r2']],
    '训练集_MSE': [full_results['train_mse'], auto_results['train_mse'], manual_results['train_mse']],
    '测试集_R2': [full_results['test_r2'], auto_results['test_r2'], manual_results['test_r2']],
    '测试集_MSE': [full_results['test_mse'], auto_results['test_mse'], manual_results['test_mse']],
    'R2差异(训练-测试)': [full_results['train_r2'] - full_results['test_r2'],
                          auto_results['train_r2'] - auto_results['test_r2'],
                          manual_results['train_r2'] - manual_results['test_r2']]
})

# 保存到Excel
output_path = r'C:\Users\LiangHuan\Desktop\原始数据、特征数据\RFE分析结果_完整版.xlsx'
with pd.ExcelWriter(output_path) as writer:
    selected_df.to_excel(writer, sheet_name='选中特征', index=False)
    ranking_df.to_excel(writer, sheet_name='全部特征排名', index=False)
    performance_df.to_excel(writer, sheet_name='性能对比', index=False)

    # 添加额外的信息表
    info_df = pd.DataFrame({
        '项目': ['数据集', '总特征数', '最优特征数(自动)', '手动选择特征数',
                 '训练集样本数', '测试集样本数', '随机种子'],
        '值': [SHEET_NAME, X.shape[1], optimal_n_features, manual_n_features,
               X_train.shape[0], X_test.shape[0], 42]
    })
    info_df.to_excel(writer, sheet_name='分析信息', index=False)

print(f"✓ 结果已保存到: {output_path}")

# 9. 总结分析
print("\n" + "=" * 60)
print("分析完成！详细总结")
print("=" * 60)
print(f"1. 数据信息:")
print(f"   - 原始特征数量: {X.shape[1]}")
print(f"   - RFECV推荐最优特征数量: {optimal_n_features}")
print(f"   - 实际使用特征数量: {manual_n_features}")
print(f"   - 特征减少比例: {(1 - manual_n_features / X.shape[1]) * 100:.1f}%")

print(f"\n2. 模型性能总结:")
print(f"   - 测试集R²分数 (所有特征): {full_results['test_r2']:.4f}")
print(f"   - 测试集R²分数 (手动选择): {manual_results['test_r2']:.4f}")
print(f"   - R²变化: {manual_results['test_r2'] - full_results['test_r2']:+.4f}")

print(f"\n3. 过拟合评估:")
print(f"   - 所有特征: 训练集-测试集R²差异 = {full_results['train_r2'] - full_results['test_r2']:.4f}")
print(f"   - 手动选择: 训练集-测试集R²差异 = {manual_results['train_r2'] - manual_results['test_r2']:.4f}")

if manual_results['test_r2'] >= full_results['test_r2']:
    print(f"\n✓ 结论: 使用更少的特征获得了相似或更好的性能")
else:
    r2_loss = full_results['test_r2'] - manual_results['test_r2']
    if r2_loss < 0.05:
        print(f"\n⚠ 结论: 特征减少后测试集R²略有下降 ({r2_loss:.4f})，但降幅不大")
    else:
        print(f"\n⚠ 结论: 特征减少后测试集R²下降明显 ({r2_loss:.4f})")

print(f"\n4. 选中的前5个特征:")
for i in range(min(5, len(selected_features_manual))):
    print(f"   {i + 1}. {selected_features_manual[i]}")