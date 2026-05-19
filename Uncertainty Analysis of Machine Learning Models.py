import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['mathtext.rm'] = 'Times New Roman'  

file_path = r'C:\Users\LiangHuan\Desktop\原始数据、特征数据\原始数据合并版本1987.xls'

try:

    try:
        df = pd.read_excel(file_path, sheet_name='BEQbio(ER-calux）提取s', engine='xlrd')
    except:
        df = pd.read_excel(file_path, sheet_name='BEQbio(ER-calux）提取s', engine='openpyxl')

    print(f"Data loaded successfully, shape: {df.shape}")
    print(f"Column names: {df.columns.tolist()}")

except Exception as e:
    print(f"Failed to load data, error: {e}")
    exit()


X = df.iloc[:, 4:14] 
y = df.iloc[:, -2]  

print(f"Feature data shape: {X.shape}")
print(f"Target variable shape: {y.shape}")


if X.isnull().any().any() or y.isnull().any():
    print("Missing values detected, processing...")
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())


X_train, y_train = X, y
X_test, y_test = X, y  

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    oob_score=True
)
rf_model.fit(X_train, y_train)



def calculate_confidence_intervals(X_data, y_data, model, n_bootstrap=100, alpha=0.05):

    predictions = []

    for i in range(n_bootstrap):

        X_resample, y_resample = resample(X_data, y_data, random_state=i)


        temp_model = RandomForestRegressor(
            n_estimators=50, 
            random_state=i,
            n_jobs=-1
        )
        temp_model.fit(X_resample, y_resample)


        pred = temp_model.predict(X_test)
        predictions.append(pred)


    predictions = np.array(predictions)  


    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)


    z_score = 1.96  
    lower_bound = mean_predictions - z_score * std_predictions
    upper_bound = mean_predictions + z_score * std_predictions

    return mean_predictions, lower_bound, upper_bound, predictions



print("Calculating confidence intervals...")
mean_preds, lower_bounds, upper_bounds, all_predictions = calculate_confidence_intervals(
    X_train, y_train, rf_model, n_bootstrap=100
)


plt.figure(figsize=(14, 8))


sorted_indices = np.argsort(y_test.values)
y_test_sorted = y_test.values[sorted_indices]
mean_preds_sorted = mean_preds[sorted_indices]
lower_bounds_sorted = lower_bounds[sorted_indices]
upper_bounds_sorted = upper_bounds[sorted_indices]


x_index = np.arange(len(y_test_sorted))


plt.fill_between(
    x_index,
    lower_bounds_sorted,
    upper_bounds_sorted,
    alpha=0.3,
    color='steelblue',
    label='95% Confidence Interval'
)


plt.plot(x_index, mean_preds_sorted, 'b-', linewidth=2, label='Predicted Mean')


plt.scatter(x_index, y_test_sorted, color='red', alpha=0.6, s=10, label='True Values', zorder=5)


plt.xlabel('Test Samples (Sorted by True Values)', fontsize=20,fontweight='bold')
plt.ylabel('Predicted Values', fontsize=20,fontweight='bold')

plt.legend(loc='best', fontsize=20)


plt.grid(False)


n_samples = len(y_test_sorted)
if n_samples > 100:
    tick_interval = 200
    xticks = np.arange(0, n_samples, tick_interval)
else:
    xticks = np.arange(0, n_samples, max(1, n_samples // 10))

plt.xticks(xticks, fontsize=20, fontweight='bold')
plt.yticks(fontsize=20, fontweight='bold')


plt.xlabel('Test Samples (Sorted by True Values)', fontsize=20, fontweight='bold')
plt.ylabel('Predicted Values', fontsize=20, fontweight='bold')



ax = plt.gca()
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)

plt.grid(False)


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

from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(scientific_format))

plt.tight_layout()


output_path = r'C:\Users\LiangHuan\Desktop\confidence_interval_plot.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Confidence interval plot saved to: {output_path}")

plt.show()


print("\n" + "=" * 60)
print("Model Uncertainty Analysis Metrics")
print("=" * 60)


y_true = y_test.values.astype(float)
y_pred = mean_preds.astype(float)
lower = lower_bounds.astype(float)
upper = upper_bounds.astype(float)


interval_width = upper - lower

# 1. PICP: Prediction Interval Coverage Probability

within_interval = (y_true >= lower) & (y_true <= upper)
picp = np.mean(within_interval)

# 2. MPIW: Mean Prediction Interval Width

mpiw = np.mean(interval_width)

# 3. PINAW: Prediction Interval Normalized Average Width

y_range = np.max(y_true) - np.min(y_true)

if y_range == 0:
    pinaw = np.nan
else:
    pinaw = mpiw / y_range


alpha = 0.05

lower_penalty = (2 / alpha) * (lower - y_true) * (y_true < lower)
upper_penalty = (2 / alpha) * (y_true - upper) * (y_true > upper)

interval_score = interval_width + lower_penalty + upper_penalty
mis = np.mean(interval_score)


mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

if ss_tot == 0:
    r2 = np.nan
else:
    r2 = 1 - ss_res / ss_tot


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

if len(X_test) <= 20:  
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i, idx in enumerate(sorted_indices[:20]):
        ax = axes[i]
        ax.hist(all_predictions[:, idx], bins=15, alpha=0.7, edgecolor='black')
        ax.axvline(x=y_test.values[idx], color='red', linestyle='--', linewidth=2, label='True Value')
        ax.axvline(x=mean_preds[idx], color='blue', linestyle='-', linewidth=2, label='Predicted Mean')
        ax.set_title(f'Sample {idx}')

        ax.xaxis.set_major_formatter(FuncFormatter(scientific_format))
        ax.grid(False)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Random Forest Model Prediction Distribution (First 20 Samples)', fontsize=16, fontweight='bold')
    plt.tight_layout()


    dist_output_path = r'C:\Users\LiangHuan\Desktop\prediction_distribution_plot.png'
    plt.savefig(dist_output_path, dpi=300, bbox_inches='tight')
    print(f"Prediction distribution plot saved to: {dist_output_path}")
    plt.show()


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
