from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(filename: str, title: str, cells: list):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nb["cells"] = [md(f"# {title}\n\n**Quy trình:** Data → Understand → Clean → Represent → Train → Validate → Test → Persist → Deploy")]
    nb["cells"].extend(cells)
    nbf.write(nb, ROOT / filename)


COMMON_IMPORTS = r"""
from pathlib import Path
import json, os, re, warnings
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-asg02")
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import display
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
ROOT = Path.cwd()
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "artifacts").mkdir(exist_ok=True)
(ROOT / "figures").mkdir(exist_ok=True)
RANDOM_STATE = 42
"""


diabetes_cells = [
    md("""
    ## 1. Bài toán và nguồn dữ liệu

    - **Dataset:** Diabetes Prediction Dataset — https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset/data
    - **Mục tiêu:** phân loại nhị phân nguy cơ tiểu đường (`diabetes`: 0/1).
    - **Một quan sát:** một hồ sơ người bệnh tại một thời điểm.
    - **Đặc trưng triển khai:** `age`, `bmi`, `hbA1c_level`, `blood_glucose_level`, `gender`, `smoking_history`.
    - Các cột `year`, `location`, nhóm chủng tộc, `hypertension`, `heart_disease` được loại khỏi mô hình để đầu vào huấn luyện khớp chính xác với biểu mẫu/API đã triển khai. Đây là đánh đổi: giảm thông tin nhưng tránh training-serving skew và hạn chế sử dụng thuộc tính nhạy cảm.

    Notebook dùng validation để so sánh 5 mô hình. Test được giữ kín đến sau khi chọn mô hình.
    """),
    code(COMMON_IMPORTS + r"""
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
"""),
    md("## 2. Kiểm tra dữ liệu bắt buộc"),
    code(r"""
df_raw = pd.read_csv(ROOT / "diabetes_dataset.csv")
print("shape:", df_raw.shape)
display(df_raw.head().T)
df_raw.info()
display(df_raw.describe(include="number").T[["count", "mean", "std", "min", "50%", "max"]].round(2))
display(df_raw.describe(exclude="number").T[["count", "unique", "top", "freq"]])
print("\nMissing values:\n", df_raw.isna().sum())
print("\nDuplicated records:", df_raw.duplicated().sum())
print("\nTarget distribution:\n", df_raw["diabetes"].value_counts())
"""),
    md("""
    ## 3. Làm sạch, chia dữ liệu và chống leakage

    Bản ghi trùng hoàn toàn được loại **trước khi chia** để cùng một hồ sơ không xuất hiện ở cả train và test. Sau đó chia 70/15/15 có stratification. Imputer, scaler và one-hot encoder chỉ được `fit` trên train bên trong `Pipeline`.
    """),
    code(r"""
features = ["age", "bmi", "hbA1c_level", "blood_glucose_level", "gender", "smoking_history"]
target = "diabetes"

df = df_raw[features + [target]].copy()
df["gender"] = df["gender"].astype("string").str.strip()
df["smoking_history"] = df["smoking_history"].astype("string").str.strip()
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Removed {before-len(df):,} exact duplicates; cleaned shape = {df.shape}")

X, y = df[features], df[target].astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
print("train/validation/test:", X_train.shape, X_val.shape, X_test.shape)
print("positive rates:", y_train.mean().round(4), y_val.mean().round(4), y_test.mean().round(4))
"""),
    md("## 4. EDA: quan sát → diễn giải → hàm ý ML"),
    code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="diabetes", ax=axes[0], color="#4C78A8")
axes[0].set_title("Target distribution")
sns.boxplot(data=df.sample(min(30000, len(df)), random_state=RANDOM_STATE), x="diabetes", y="hbA1c_level", ax=axes[1])
axes[1].set_title("HbA1c by target")
sns.boxplot(data=df.sample(min(30000, len(df)), random_state=RANDOM_STATE), x="diabetes", y="blood_glucose_level", ax=axes[2])
axes[2].set_title("Glucose by target")
plt.tight_layout()
plt.savefig(ROOT / "figures/diabetes_eda.png", dpi=160, bbox_inches="tight")
plt.show()

print("Observation: lớp 1 ít hơn rõ rệt; HbA1c và glucose của lớp 1 cao hơn.")
print("Interpretation: accuracy đơn lẻ có thể che giấu false negative.")
print("ML implication: dùng stratification, class_weight và ưu tiên Recall/F1 của lớp 1.")
"""),
    md("""
    ## 5. Biểu diễn dữ liệu

    Một hồ sơ thô gồm 4 số + 2 biến phân loại. Sau điền khuyết, chuẩn hóa số và one-hot encoding, mỗi hồ sơ thành vector $x_i\\in\\mathbb{R}^d$; một batch có dạng $X\\in\\mathbb{R}^{B\\times d}$. `handle_unknown='ignore'` giúp API nhận hạng mục hợp lệ mới mà không làm sai số chiều.
    """),
    code(r"""
numeric_features = ["age", "bmi", "hbA1c_level", "blood_glucose_level"]
categorical_features = ["gender", "smoking_history"]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_features),
])

representation = clone(preprocessor).fit_transform(X_train)
print("Original record:")
display(X_train.head(1))
print("Raw feature shape:", X_train.shape)
print("Final numerical matrix:", representation.shape, representation.dtype)
print("One numerical vector (rounded):", np.round(representation[0], 3))
"""),
    md("## 6. So sánh 5 mô hình trên validation"),
    code(r"""
models = {
    "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
    "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1),
}

def score_binary(model, X_eval, y_eval):
    pred = model.predict(X_eval)
    score = model.predict_proba(X_eval)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_eval)
    return {
        "Accuracy": accuracy_score(y_eval, pred),
        "Precision_1": precision_score(y_eval, pred, zero_division=0),
        "Recall_1": recall_score(y_eval, pred, zero_division=0),
        "F1_1": f1_score(y_eval, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_eval, score),
    }

rows, fitted = [], {}
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(estimator))])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    rows.append({"Model": name, **score_binary(pipe, X_val, y_val)})

comparison = pd.DataFrame(rows).sort_values(["F1_1", "Recall_1"], ascending=False).reset_index(drop=True)
display(comparison.style.format({c: "{:.4f}" for c in comparison.columns if c != "Model"}))
comparison.to_csv(ROOT / "artifacts/diabetes_validation_metrics.csv", index=False)
selected_name = comparison.loc[0, "Model"]
print("Selected by validation F1_1:", selected_name)
"""),
    md("## 7. Refit train+validation, test đúng một lần và lưu pipeline"),
    code(r"""
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
final_pipeline = Pipeline([
    ("preprocess", clone(preprocessor)),
    ("model", clone(models[selected_name])),
])
final_pipeline.fit(X_trainval, y_trainval)
test_metrics = score_binary(final_pipeline, X_test, y_test)
print("FINAL TEST METRICS")
display(pd.DataFrame([test_metrics]).style.format("{:.4f}"))

test_pred = final_pipeline.predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_predictions(y_test, test_pred, ax=axes[0], colorbar=False)
axes[0].set_title("Diabetes — test confusion matrix")
if hasattr(final_pipeline, "predict_proba"):
    RocCurveDisplay.from_predictions(y_test, final_pipeline.predict_proba(X_test)[:, 1], ax=axes[1])
else:
    RocCurveDisplay.from_predictions(y_test, final_pipeline.decision_function(X_test), ax=axes[1])
axes[1].set_title("Diabetes — test ROC")
plt.tight_layout()
plt.savefig(ROOT / "figures/diabetes_test_evaluation.png", dpi=160, bbox_inches="tight")
plt.show()

tn, fp, fn, tp = confusion_matrix(y_test, test_pred).ravel()
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}. FN là ca bệnh bị bỏ sót nên cần theo dõi đặc biệt.")

joblib.dump(final_pipeline, ROOT / "models/diabetes_pipeline.joblib", compress=3)
metadata = {
    "application": "diabetes", "selected_model": selected_name, "raw_features": features,
    "split": "70/15/15 stratified; model selection on validation; one final test evaluation",
    "selection_metric": "F1 positive class", "test_metrics": test_metrics,
    "final_input_shape": [int(len(X_trainval)), int(final_pipeline.named_steps["preprocess"].transform(X_trainval.head(1)).shape[1])],
    "random_state": RANDOM_STATE,
}
(ROOT / "models/diabetes_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved models/diabetes_pipeline.joblib and metadata.")
"""),
]


housing_cells = [
    md("""
    ## 1. Bài toán và nguồn dữ liệu

    - **Dataset:** Vietnam Housing Dataset (Hà Nội) — https://www.kaggle.com/code/kerneler/starter-vietnam-housing-dataset-5742ed64-7/input
    - **Mục tiêu:** hồi quy giá bán nhà Hà Nội theo đơn vị tỷ VNĐ.
    - **Một quan sát:** một tin đăng bất động sản.
    - **Đặc trưng triển khai:** diện tích, số phòng ngủ, số tầng, quận, loại hình nhà, tình trạng pháp lý.
    - `Giá/m2` chỉ dùng để tạo nhãn `Price`, sau đó bị loại khỏi X để ngăn target leakage.

    Notebook dùng validation để so sánh 5 mô hình; test chỉ được dùng một lần sau lựa chọn.
    """),
    code(COMMON_IMPORTS + r"""
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit
"""),
    md("## 2. Kiểm tra dữ liệu bắt buộc"),
    code(r"""
df_raw = pd.read_csv(ROOT / "VN_housing_dataset.csv")
print("shape:", df_raw.shape)
display(df_raw.head().T)
df_raw.info()
display(df_raw.describe(include="number").T[["count", "mean", "std", "min", "50%", "max"]].round(2))
display(df_raw.describe(exclude="number").T[["count", "unique", "top", "freq"]])
print("\nMissing values:\n", df_raw.isna().sum())
print("\nDuplicated records:", df_raw.duplicated().sum())
"""),
    md("""
    ## 3. Làm sạch có quy tắc nghiệp vụ

    Dữ liệu Việt Nam dùng dấu phẩy thập phân và ba đơn vị `triệu/m²`, `tỷ/m²`, `đ/m²`. Hàm dưới đây quy đổi tất cả về **triệu VNĐ/m²** trước khi tính `Price`. Các ngưỡng cố định (không học từ test) loại bản ghi không thể tin cậy: diện tích 10–1000 m², đơn giá 0.1–1000 triệu/m², tổng giá 0.1–300 tỷ. Không gọi đây là IQR vì ngưỡng là quy tắc nghiệp vụ.
    """),
    code(r"""
def first_decimal(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+(?:[\.,]\d+)?", str(value))
    return float(match.group(0).replace(",", ".")) if match else np.nan

def price_per_m2_to_million(value):
    if pd.isna(value):
        return np.nan
    text = str(value).lower().strip()
    match = re.search(r"\d+(?:[\.,]\d+)?", text)
    if not match:
        return np.nan
    token = match.group(0)
    if "triệu" in text:
        return float(token.replace(",", "."))
    if "tỷ" in text:
        return float(token.replace(",", ".")) * 1000.0
    # Chuỗi dạng 90.476 đ/m² dùng dấu chấm phân cách hàng nghìn.
    return float(token.replace(".", "")) / 1_000_000.0

df = df_raw.copy()
df["Diện tích"] = df["Diện tích"].map(first_decimal)
df["Số phòng ngủ"] = df["Số phòng ngủ"].map(first_decimal)
df["Số tầng"] = df["Số tầng"].map(first_decimal)
df["Đơn giá (triệu/m²)"] = df["Giá/m2"].map(price_per_m2_to_million)
df["Price"] = df["Diện tích"] * df["Đơn giá (triệu/m²)"] / 1000.0

features = ["Diện tích", "Số phòng ngủ", "Số tầng", "Quận", "Loại hình nhà ở", "Giấy tờ pháp lý"]
target = "Price"
valid = (
    df["Diện tích"].between(10, 1000)
    & df["Đơn giá (triệu/m²)"].between(0.1, 1000)
    & df["Price"].between(0.1, 300)
)
df = df.loc[valid, features + [target]].copy()
for col in ["Quận", "Loại hình nhà ở", "Giấy tờ pháp lý"]:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Valid rows before/after exact de-duplication: {before:,} / {len(df):,}")
display(df.head())
"""),
    md("""
    ## 4. Chia 70/15/15 theo nhóm đặc trưng

    Vì nhiều tin có cùng sáu đặc trưng triển khai, `GroupShuffleSplit` giữ toàn bộ hồ sơ có cùng vector X trong cùng một tập. Nhờ vậy test không chứa bản sao chính xác của một profile đã thấy ở train. Imputer/encoder/scaler vẫn chỉ fit trên train.
    """),
    code(r"""
X, y = df[features], df[target]
group_key = X.astype("string").fillna("<NA>").agg("|".join, axis=1)
outer = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=RANDOM_STATE)
train_idx, temp_idx = next(outer.split(X, y, groups=group_key))
X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]
temp_groups = group_key.iloc[temp_idx]
inner = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=RANDOM_STATE)
val_rel, test_rel = next(inner.split(X_temp, y_temp, groups=temp_groups))
X_val, y_val = X_temp.iloc[val_rel], y_temp.iloc[val_rel]
X_test, y_test = X_temp.iloc[test_rel], y_temp.iloc[test_rel]

def keys(frame):
    return set(frame.astype("string").fillna("<NA>").agg("|".join, axis=1))
print("train/validation/test:", X_train.shape, X_val.shape, X_test.shape)
print("Exact profile overlap train-test:", len(keys(X_train) & keys(X_test)))
"""),
    md("## 5. EDA: quan sát → diễn giải → hàm ý ML"),
    code(r"""
sample = df.sample(min(25000, len(df)), random_state=RANDOM_STATE)
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
sns.histplot(sample["Price"], bins=60, ax=axes[0, 0], color="#4C78A8")
axes[0, 0].set_title("Target: house price (billion VND)")
sns.scatterplot(data=sample, x="Diện tích", y="Price", alpha=.25, ax=axes[0, 1])
axes[0, 1].set_xlim(0, 300); axes[0, 1].set_ylim(0, 60)
axes[0, 1].set_title("Area vs price (zoomed)")
district_order = sample.groupby("Quận")["Price"].median().nlargest(10).index
sns.boxplot(data=sample[sample["Quận"].isin(district_order)], y="Quận", x="Price", ax=axes[1, 0])
axes[1, 0].set_xlim(0, 50); axes[1, 0].set_title("Top districts by median price")
sns.heatmap(df[["Diện tích", "Số phòng ngủ", "Số tầng", "Price"]].corr(), annot=True, cmap="vlag", vmin=-1, vmax=1, ax=axes[1, 1])
axes[1, 1].set_title("Numeric correlation")
plt.tight_layout()
plt.savefig(ROOT / "figures/housing_eda.png", dpi=160, bbox_inches="tight")
plt.show()

print("Observation: giá lệch phải; diện tích và quận liên quan rõ đến giá.")
print("Interpretation: một số giao dịch cao giá vẫn còn hợp lệ sau lọc nghiệp vụ.")
print("ML implication: ưu tiên MAE, đồng thời báo RMSE để phạt sai số lớn.")
"""),
    md("""
    ## 6. Biểu diễn dữ liệu

    Ba biến số được điền trung vị và chuẩn hóa; ba biến phân loại được điền `Unknown` rồi one-hot. Vector cuối có $d$ chiều, batch có dạng $B\times d$.
    """),
    code(r"""
numeric_features = ["Diện tích", "Số phòng ngủ", "Số tầng"]
categorical_features = ["Quận", "Loại hình nhà ở", "Giấy tờ pháp lý"]
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric_features),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), categorical_features),
])
representation = clone(preprocessor).fit_transform(X_train)
display(X_train.head(1))
print("Raw matrix:", X_train.shape)
print("Final numerical matrix:", representation.shape, representation.dtype)
print("One vector (rounded):", np.round(representation[0], 3))
"""),
    md("## 7. So sánh 5 mô hình trên validation"),
    code(r"""
models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=10.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=16, min_samples_leaf=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=2, max_features=.8, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": HistGradientBoostingRegressor(max_iter=180, learning_rate=.07, l2_regularization=1.0, random_state=RANDOM_STATE),
}

def score_regression(model, X_eval, y_eval):
    pred = model.predict(X_eval)
    mse = mean_squared_error(y_eval, pred)
    return {"MAE": mean_absolute_error(y_eval, pred), "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y_eval, pred)}

rows = []
for name, estimator in models.items():
    pipe = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(estimator))])
    pipe.fit(X_train, y_train)
    rows.append({"Model": name, **score_regression(pipe, X_val, y_val)})
comparison = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
display(comparison.style.format({"MAE":"{:.4f}", "MSE":"{:.4f}", "RMSE":"{:.4f}", "R2":"{:.4f}"}))
comparison.to_csv(ROOT / "artifacts/housing_validation_metrics.csv", index=False)
selected_name = comparison.loc[0, "Model"]
print("Selected by validation MAE:", selected_name)

# Khoảng bất định vận hành: bách phân vị 90% của |residual| trên validation.
# Đây là heuristic được hiệu chỉnh trên validation, không phải confidence interval thống kê.
interval_pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(models[selected_name]))])
interval_pipeline.fit(X_train, y_train)
validation_abs_error_q90 = float(np.quantile(np.abs(y_val.to_numpy() - interval_pipeline.predict(X_val)), 0.90))
print("Validation absolute-error q90 (billion VND):", round(validation_abs_error_q90, 4))
"""),
    md("## 8. Refit train+validation, test đúng một lần và lưu pipeline"),
    code(r"""
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
final_pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(models[selected_name]))])
final_pipeline.fit(X_trainval, y_trainval)
test_metrics = score_regression(final_pipeline, X_test, y_test)
print("FINAL TEST METRICS")
display(pd.DataFrame([test_metrics]).style.format("{:.4f}"))

test_pred = final_pipeline.predict(X_test)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(y_test, test_pred, alpha=.25)
lims = [0, min(100, max(float(y_test.max()), float(test_pred.max())))]
axes[0].plot(lims, lims, "--", color="black"); axes[0].set(xlim=lims, ylim=lims, xlabel="Actual", ylabel="Predicted", title="Actual vs predicted (zoomed)")
residual = y_test.to_numpy() - test_pred
sns.histplot(residual, bins=50, ax=axes[1]); axes[1].set_title("Test residuals")
plt.tight_layout()
plt.savefig(ROOT / "figures/housing_test_evaluation.png", dpi=160, bbox_inches="tight")
plt.show()

joblib.dump(final_pipeline, ROOT / "models/housing_pipeline.joblib", compress=3)
metadata = {
    "application": "housing", "selected_model": selected_name, "raw_features": features,
    "target_unit": "billion VND", "split": "group-aware approx. 70/15/15; selection on validation; one final test evaluation",
    "selection_metric": "MAE", "test_metrics": test_metrics,
    "validation_absolute_error_q90": validation_abs_error_q90,
    "uncertainty_note": "Heuristic prediction range based on validation absolute-error q90; not a statistical confidence interval",
    "final_input_shape": [int(len(X_trainval)), int(final_pipeline.named_steps["preprocess"].transform(X_trainval.head(1)).shape[1])],
    "random_state": RANDOM_STATE,
}
(ROOT / "models/housing_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved models/housing_pipeline.joblib and metadata.")
"""),
]


# Notebook e-commerce có pipeline NLP riêng trong tools/build_ecommerce_notebook.py.

write_notebook("ASG_02_1.ipynb", "Ứng dụng 1 — Dự đoán tiểu đường", diabetes_cells)
write_notebook("ASG_02_2.ipynb", "Ứng dụng 2 — Dự đoán giá nhà Hà Nội", housing_cells)
subprocess.run(
    [sys.executable, str(ROOT / "tools" / "build_ecommerce_notebook.py")],
    cwd=ROOT,
    check=True,
)
print("Generated three notebooks in", ROOT)
