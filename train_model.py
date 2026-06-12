"""
train_model.py
================
训练并对比两个模型来预测新加坡年度 CO2 总排放 (Mt)。

方法论(答辩可直接引用):
1. 目标 Y = co2(总排放 Mt)。但 co2 是强趋势非平稳序列,直接预测绝对值会
   导致外推失效(测试段 R2 为负)。因此模型实际学习目标改为【差分 Δco2】
   (每年相对上一年的变化量),预测时用【前一年真实值 + 预测变化量】做 one-step-ahead 还原。
   这是处理非平稳时间序列的标准做法,可消除趋势、稳定外推。
2. 特征严格排除 co2 的组成部分(oil/gas/coal/fossil_fuel_co2),防止目标泄漏。
   同时把特征也做差分(预测 Δco2 用各驱动变量的 Δ),口径一致。
3. 两个模型:
   - Benchmark: LinearRegression(简单可解释,基准下限)
   - 进阶: XGBoost(捕捉非线性,TimeSeriesSplit 交叉验证调参,不用随机K折)
4. 评估:在【还原后的绝对 co2】上计算 MAE/RMSE/R2,保证指标对应用户看到的量纲。
5. 产出: best_co2_model.joblib / benchmark_model.joblib / model_metrics.json

运行: python train_model.py
"""

import json
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBRegressor

RANDOM_STATE = 42

# 原始驱动特征(差分前)。排除 co2 组成部分防泄漏。
BASE_FEATURES = [
    "population",
    "gdp",
    "primary_energy_consumption",
    "energy_per_gdp",
    "energy_per_capita",
]
# 模型实际输入 = 各驱动变量的差分 + year(保留时间信号)
FEATURES = ["year"] + [f"d_{c}" for c in BASE_FEATURES]
TARGET = "co2"
DTARGET = "d_co2"


def load_data(path="cleaned_sg_co2_data.csv"):
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
    df = df[df["primary_energy_consumption"] > 0].reset_index(drop=True)
    return df


def make_diff_frame(df):
    """构造差分特征与差分目标。第一行因无前值被丢弃。"""
    d = df.copy()
    for c in BASE_FEATURES:
        d[f"d_{c}"] = d[c].diff()
    d[DTARGET] = d[TARGET].diff()
    d = d.dropna().reset_index(drop=True)
    return d


def evaluate_levels(y_true_level, y_pred_level):
    return {
        "MAE": float(mean_absolute_error(y_true_level, y_pred_level)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_level, y_pred_level))),
        "R2": float(r2_score(y_true_level, y_pred_level)),
    }


def reconstruct(prev_level_start, d_preds):
    """把预测的 Δ 序列累加还原成绝对水平。prev_level_start = 测试集前一年的真实co2。"""
    levels = []
    cur = prev_level_start
    for d in d_preds:
        cur = cur + d
        levels.append(cur)
    return np.array(levels)


def main():
    df = load_data()
    print(f"原始可用年份: {df['year'].min()}-{df['year'].max()} ({len(df)}行)")

    d = make_diff_frame(df)
    print(f"差分后可用: {d['year'].min()}-{d['year'].max()} ({len(d)}行)")

    n_test = 8
    train = d.iloc[:-n_test]
    test = d.iloc[-n_test:]
    X_train, y_train = train[FEATURES], train[DTARGET]
    X_test, y_test = test[FEATURES], test[DTARGET]
    print(f"训练集: {train['year'].min()}-{train['year'].max()} ({len(train)}行)")
    print(f"测试集: {test['year'].min()}-{test['year'].max()} ({len(test)}行)\n")

    # 测试集起点前一年的真实绝对 co2(用于累加还原)
    first_test_year = int(test["year"].iloc[0])
    prev_level = float(df.loc[df["year"] == first_test_year - 1, "co2"].iloc[0])
    # 测试集每年的真实绝对 co2(算指标用)
    true_levels = df[df["year"].isin(test["year"])].sort_values("year")["co2"].values
    # one-step-ahead: 每个测试年用其"前一年真实co2"作为还原基准
    prev_levels = df.set_index("year").loc[[y-1 for y in test["year"]], "co2"].values

    # ---------- 1. Benchmark: 线性回归 ----------
    benchmark = LinearRegression().fit(X_train, y_train)
    bench_dpred = benchmark.predict(X_test)
    bench_levels = prev_levels + bench_dpred  # one-step-ahead 还原
    bench_metrics = evaluate_levels(true_levels, bench_levels)
    print("[Benchmark] LinearRegression (还原后绝对值):", bench_metrics)

    # ---------- 2. XGBoost + 时间序列CV调参 ----------
    tscv = TimeSeriesSplit(n_splits=4)
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [2, 3],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }
    xgb = XGBRegressor(random_state=RANDOM_STATE, objective="reg:squarederror")
    grid = GridSearchCV(xgb, param_grid, cv=tscv,
                        scoring="neg_mean_absolute_error", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_xgb = grid.best_estimator_
    xgb_dpred = best_xgb.predict(X_test)
    xgb_levels = prev_levels + xgb_dpred  # one-step-ahead 还原
    xgb_metrics = evaluate_levels(true_levels, xgb_levels)
    print("[Advanced] XGBoost 最优参数:", grid.best_params_)
    print("[Advanced] XGBoost (还原后绝对值):", xgb_metrics)

    # ---------- 3. 选优(测试集 MAE 越小越好) ----------
    if xgb_metrics["MAE"] <= bench_metrics["MAE"]:
        best_name, best_test_levels = "XGBoost", xgb_levels
    else:
        best_name, best_test_levels = "LinearRegression", bench_levels
    print(f"\n>>> 选中模型: {best_name}")

    # ---------- 4. 用全部差分数据重训选中模型 ----------
    X_all, y_all = d[FEATURES], d[DTARGET]
    if best_name == "XGBoost":
        final_model = XGBRegressor(random_state=RANDOM_STATE,
                                   objective="reg:squarederror",
                                   **grid.best_params_).fit(X_all, y_all)
    else:
        final_model = LinearRegression().fit(X_all, y_all)
    final_benchmark = LinearRegression().fit(X_all, y_all)

    # 残差标准差(基于还原后绝对值),用于真实置信区间
    resid = true_levels - best_test_levels
    resid_std = float(np.std(resid, ddof=1))

    # ---------- 5. 保存 ----------
    bundle = {
        "model": final_model,
        "base_features": BASE_FEATURES,
        "features": FEATURES,
        "name": best_name,
        "target": TARGET,
        "diff_target": True,
    }
    joblib.dump(bundle, "best_co2_model.joblib")
    joblib.dump({
        "model": final_benchmark,
        "base_features": BASE_FEATURES,
        "features": FEATURES,
        "name": "LinearRegression",
        "target": TARGET,
        "diff_target": True,
    }, "benchmark_model.joblib")

    metrics_out = {
        "approach": "Predict yearly change (diff), one-step-ahead reconstruction to absolute level",
        "base_features": BASE_FEATURES,
        "model_input_features": FEATURES,
        "target": TARGET,
        "excluded_for_leakage": ["coal_co2", "oil_co2", "gas_co2", "fossil_fuel_co2"],
        "train_period": [int(train["year"].min()), int(train["year"].max())],
        "test_period": [int(test["year"].min()), int(test["year"].max())],
        "benchmark": {"model": "LinearRegression", "metrics": bench_metrics},
        "advanced": {"model": "XGBoost", "best_params": grid.best_params_, "metrics": xgb_metrics},
        "selected_model": best_name,
        "residual_std": resid_std,
    }
    with open("model_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print("已保存: best_co2_model.joblib, benchmark_model.joblib, model_metrics.json")


if __name__ == "__main__":
    main()
