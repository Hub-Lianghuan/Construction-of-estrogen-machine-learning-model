import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

# =========================
# Basic plot settings
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1. File path
# =========================
file_path = r"C:\Users\LiangHuan\Desktop\原始数据、特征数据\原始数据合并版本1987.xls"

target_configs = [
    {
        "target_name": "E1浓度(ng/L)",
        "sheet_name": "E1提取001多重",
        "feature_slice": (3, 13),
        "target_col_index": -2
    },
    {
        "target_name": "E2浓度(ng/L)",
        "sheet_name": "E2提取00",
        "feature_slice": (3, 13),
        "target_col_index": -2
    },
    {
        "target_name": "EE2浓度(ng/L)",
        "sheet_name": "EE2提取00",
        "feature_slice": (3, 12),
        "target_col_index": -3
    },
    {
        "target_name": "BEQbio(YES)",
        "sheet_name": "BEQbio(YES)提取s",
        "feature_slice": (4, 14),
        "target_col_index": -2
    },
    {
        "target_name": "BEQbio(ER-calux）",
        "sheet_name": "BEQbio(ER-calux）提取s",
        "feature_slice": (4, 14),
        "target_col_index": -2
    },
    {
        "target_name": "BEQbio(E-Screen）",
        "sheet_name": "BEQbio(E-Screen）提取s",
        "feature_slice": (4, 14),
        "target_col_index": -2
    }
]


# =========================
# 3. Read Excel safely
# =========================
def read_excel_safely(file_path, sheet_name):
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="xlrd")
    except Exception:
        df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    return df


# =========================
# 4. Single-target sensitivity analysis
# =========================
def run_single_target_sensitivity(
    file_path,
    config,
    perturbation_rate=0.10,
    n_estimators=100,
    random_state=42,
    eps=1e-12
):
    target_name = config["target_name"]
    sheet_name = config["sheet_name"]
    feature_start, feature_end = config["feature_slice"]
    target_col_index = config["target_col_index"]

    print("\n" + "=" * 80)
    print(f"Running sensitivity analysis for target: {target_name}")
    print("=" * 80)

    df = read_excel_safely(file_path, sheet_name)

    print(f"Sheet: {sheet_name}")
    print(f"Data shape: {df.shape}")
    print(f"All columns: {df.columns.tolist()}")

    feature_cols = df.columns[feature_start:feature_end].tolist()
    target_col = df.columns[target_col_index]

    print(f"Feature columns used for {target_name}: {feature_cols}")
    print(f"Target column used for {target_name}: {target_col}")


    data = df[feature_cols + [target_col]].copy()

    for col in feature_cols + [target_col]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=[target_col])

    X = data[feature_cols].copy()
    X = X.fillna(X.mean())

    y = data[target_col].copy()
    y = y.fillna(y.mean())

    print(f"Final X shape: {X.shape}")
    print(f"Final y shape: {y.shape}")

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        oob_score=True
    )
    model.fit(X, y)

    y_base = model.predict(X)

    results = []

    for feature in feature_cols:
        for perturb_label, rate in [
            ("+10%", perturbation_rate),
            ("-10%", -perturbation_rate)
        ]:
            X_perturbed = X.copy()

            X_perturbed[feature] = X_perturbed[feature] * (1 + rate)

            if X[feature].min() >= 0:
                X_perturbed[feature] = X_perturbed[feature].clip(lower=0)

            y_perturbed = model.predict(X_perturbed)

            delta = y_perturbed - y_base

            mean_change = np.mean(delta)
            mean_abs_change = np.mean(np.abs(delta))
            mean_relative_change = np.mean(
                np.abs(delta) / (np.abs(y_base) + eps)
            )

            sensitivity_index = mean_relative_change / abs(rate)

            positive_ratio = np.mean(delta > 0)

            if mean_change > 0:
                direction = "Positive"
            elif mean_change < 0:
                direction = "Negative"
            else:
                direction = "Neutral"

            results.append({
                "Target": target_name,
                "Sheet": sheet_name,
                "Feature": feature,
                "Perturbation": perturb_label,
                "Mean_Change": mean_change,
                "Mean_Abs_Change": mean_abs_change,
                "Mean_Relative_Change": mean_relative_change,
                "Sensitivity_Index": sensitivity_index,
                "Positive_Ratio": positive_ratio,
                "Direction": direction,
                "N_Samples": len(X)
            })

    result_df = pd.DataFrame(results)
    return result_df


# =========================
# 5. Run all targets
# =========================
all_results = []

for config in target_configs:
    one_result = run_single_target_sensitivity(
        file_path=file_path,
        config=config,
        perturbation_rate=0.10,
        n_estimators=100,
        random_state=42
    )
    all_results.append(one_result)

all_results_df = pd.concat(all_results, ignore_index=True)


# =========================
# 6. Merge +10% and -10% results

summary_df = (
    all_results_df
    .groupby(["Target", "Feature"], as_index=False)
    .agg(
        SI_Mean=("Sensitivity_Index", "mean"),
        Mean_Relative_Change_Mean=("Mean_Relative_Change", "mean"),
        Mean_Abs_Change_Mean=("Mean_Abs_Change", "mean"),
        N_Samples=("N_Samples", "first")
    )
)

plus_df = all_results_df[all_results_df["Perturbation"] == "+10%"].copy()

plus_direction_df = plus_df[[
    "Target",
    "Feature",
    "Mean_Change",
    "Positive_Ratio",
    "Direction"
]].rename(columns={
    "Mean_Change": "Mean_Change_Plus10",
    "Positive_Ratio": "Positive_Ratio_Plus10",
    "Direction": "Direction_Plus10"
})

summary_df = summary_df.merge(
    plus_direction_df,
    on=["Target", "Feature"],
    how="left"
)


# =========================
# 7. Normalize SI within each target
# =========================
def normalize_within_target(s):
    max_value = s.max()
    if pd.isna(max_value) or max_value == 0:
        return s * np.nan
    return s / max_value

summary_df["Normalized_SI"] = (
    summary_df
    .groupby("Target")["SI_Mean"]
    .transform(normalize_within_target)
)


# =========================
# 8. Create heatmap matrix
# =========================
heatmap_df = summary_df.pivot(
    index="Target",
    columns="Feature",
    values="Normalized_SI"
)

target_order = [config["target_name"] for config in target_configs]
heatmap_df = heatmap_df.reindex(target_order)


# =========================
# 9. Save Excel results
# =========================
output_excel_path = r"C:\Users\LiangHuan\Desktop\feature_sensitivity_all_targets.xlsx"

with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
    all_results_df.to_excel(writer, sheet_name="All_Perturbation_Results", index=False)
    summary_df.to_excel(writer, sheet_name="Summary_For_Heatmap", index=False)
    heatmap_df.to_excel(writer, sheet_name="Heatmap_Matrix")

    top3_df = (
        summary_df
        .sort_values(["Target", "SI_Mean"], ascending=[True, False])
        .groupby("Target")
        .head(3)
    )
    top3_df.to_excel(writer, sheet_name="Top3_Features", index=False)

print(f"All sensitivity results saved to: {output_excel_path}")


# =========================
# 10. Plot heatmap
# =========================
fig_width = max(10, 0.7 * len(heatmap_df.columns))
fig_height = max(5, 0.7 * len(heatmap_df.index))

plt.figure(figsize=(fig_width, fig_height))

heatmap_values = heatmap_df.values.astype(float)
masked_values = np.ma.masked_invalid(heatmap_values)

cmap = plt.cm.YlOrRd.copy()
cmap.set_bad(color="white")

ax = plt.gca()

im = ax.imshow(
    masked_values,
    aspect="auto",
    cmap=cmap,
    vmin=0,
    vmax=1
)

ax.set_xticks(np.arange(len(heatmap_df.columns)))
ax.set_yticks(np.arange(len(heatmap_df.index)))

ax.set_xticklabels(
    heatmap_df.columns,
    rotation=45,
    ha="right",
    fontsize=12,
    fontweight="bold"
)

ax.set_yticklabels(
    heatmap_df.index,
    fontsize=12,
    fontweight="bold"
)

for i in range(heatmap_values.shape[0]):
    for j in range(heatmap_values.shape[1]):
        value = heatmap_values[i, j]
        if not np.isnan(value):
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10
            )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label(
    "Normalized Sensitivity Index",
    fontsize=14,
    fontweight="bold"
)
cbar.ax.tick_params(labelsize=12)

for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(1.5)

plt.xlabel("Feature Variables", fontsize=14, fontweight="bold")
plt.ylabel("Target Variables", fontsize=14, fontweight="bold")

plt.tight_layout()

heatmap_output_path = r"C:\Users\LiangHuan\Desktop\feature_sensitivity_heatmap.png"
plt.savefig(heatmap_output_path, dpi=300, bbox_inches="tight")
print(f"Feature sensitivity heatmap saved to: {heatmap_output_path}")

plt.show()
