from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.compose import TransformedTargetRegressor


TARGETS = ["im_long", "mm_long", "im_short", "mm_short"]


@dataclass(frozen=True)
class PreparedData:
    wide: pd.DataFrame
    long: pd.DataFrame
    feature_cols: list[str]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "turnover rate": "turnover_rate",
        "Market Cap": "market_cap",
    }
    df = df.rename(columns=rename_map)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def _safe_log(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x > 0)
    return np.log(x)


def _winsorize_numeric(df: pd.DataFrame, cols: Iterable[str], alpha: float) -> pd.DataFrame:
    if alpha <= 0:
        return df
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        lo = s.quantile(alpha)
        hi = s.quantile(1 - alpha)
        out[col] = s.clip(lower=lo, upper=hi)
    return out


def load_and_prepare(
    input_path: Path,
    *,
    target_scale: Literal["decimal", "pct_points"] = "decimal",
    winsor_alpha: float = 0.01,
) -> PreparedData:
    df = pd.read_csv(input_path)
    df = _normalize_columns(df)

    expected = {"ticker", "price", "adt", "turnover_rate", "market_cap", "n_observations"}
    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    for target in TARGETS:
        if target not in df.columns:
            raise ValueError(f"Missing target column: {target}")

    df["ticker"] = df["ticker"].astype(str)

    numeric_cols = [
        "price",
        "adt",
        "turnover_rate",
        "market_cap",
        "var_10d_pct",
        "volatility_annual_pct",
        "mean_return_annual_pct",
        "n_observations",
        *TARGETS,
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Targets are stored as percentage points (30 -> 30%).
    if target_scale == "decimal":
        for t in TARGETS:
            df[t] = df[t] / 100.0

    df["log_price"] = _safe_log(df["price"])
    df["log_adt"] = _safe_log(df["adt"])
    df["log_market_cap"] = _safe_log(df["market_cap"])
    df["log_turnover_rate"] = _safe_log(df["turnover_rate"].where(df["turnover_rate"] > 0))

    engineered = [
        "log_price",
        "log_adt",
        "log_market_cap",
        "log_turnover_rate",
        "var_10d_pct",
        "volatility_annual_pct",
        "mean_return_annual_pct",
        "n_observations",
    ]
    df = _winsorize_numeric(df, engineered, winsor_alpha)

    df["missing_risk_metrics"] = (
        df[["var_10d_pct", "volatility_annual_pct", "mean_return_annual_pct"]].isna().any(axis=1)
    ).astype(int)

    feature_cols = engineered + ["missing_risk_metrics"]

    long = df.melt(
        id_vars=["ticker", *feature_cols],
        value_vars=TARGETS,
        var_name="target",
        value_name="y",
    )
    long["margin_kind"] = long["target"].str.extract(r"^(im|mm)_", expand=False)
    long["side"] = long["target"].str.extract(r"_(long|short)$", expand=False)

    return PreparedData(wide=df, long=long, feature_cols=feature_cols)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _save_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))


def plot_distributions(df_wide: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    targets = [t for t in TARGETS if t in df_wide.columns]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, t in zip(axes, targets, strict=False):
        s = df_wide[t].dropna()
        if len(s) == 0:
            ax.set_visible(False)
            continue

        # Plot a trimmed histogram so extreme short-margin outliers (e.g. 4000%+)
        # don't collapse the main mass into a single bin.
        clip_q = 0.99
        hi = float(s.quantile(clip_q))
        n_hi = int((s > hi).sum())
        s_clip = s.clip(upper=hi)

        sns.histplot(s_clip, bins=40, ax=ax)
        ax.set_title(t)
        ax.set_xlabel("margin (%)")
        ax.text(
            0.98,
            0.98,
            f"clip@p{int(clip_q*100)}={hi:.2f}\n>clip={n_hi}\nmax={float(s.max()):.2f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9),
        )

    fig.suptitle("Margin targets distribution (trimmed; values are %)")
    fig.savefig(out_dir / "targets_distribution.png", dpi=160)
    plt.close(fig)

    # Extra diagnostic: log-scale histograms for short margins only.
    short_targets = [t for t in ["im_short", "mm_short"] if t in df_wide.columns]
    if short_targets:
        fig, axes = plt.subplots(1, len(short_targets), figsize=(10, 4), constrained_layout=True)
        if len(short_targets) == 1:
            axes = [axes]
        for ax, t in zip(axes, short_targets, strict=False):
            s = df_wide[t].dropna()
            if len(s) == 0:
                ax.set_visible(False)
                continue
            sns.histplot(np.log10(s), bins=40, ax=ax)
            ax.set_title(f"{t} (log10)")
            ax.set_xlabel("log10(margin %)")
        fig.suptitle("Short margin distribution (log scale)")
        fig.savefig(out_dir / "short_targets_distribution_log10.png", dpi=160)
        plt.close(fig)


def plot_scatter_lowess(df_wide: pd.DataFrame, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    x_cols = ["var_10d_pct", "volatility_annual_pct", "log_adt", "log_market_cap", "log_price"]
    targets = [t for t in TARGETS if t in df_wide.columns]
    for t in targets:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        axes = axes.ravel()
        s_all = df_wide[t].dropna()
        clip_q = 0.99
        y_hi = float(s_all.quantile(clip_q)) if len(s_all) else float("nan")
        for ax, x in zip(axes, x_cols, strict=False):
            d = df_wide[[x, t]].dropna()
            if len(d) == 0:
                ax.set_visible(False)
                continue
            if len(d) > 0 and np.isfinite(y_hi):
                d = d.copy()
                d[t] = d[t].clip(upper=y_hi)
            sns.regplot(
                data=d,
                x=x,
                y=t,
                lowess=True,
                scatter_kws={"s": 8, "alpha": 0.5},
                line_kws={"color": "crimson"},
                ax=ax,
            )
            ax.set_title(f"{t} vs {x} (y clipped @p{int(clip_q*100)})")
            ax.set_ylabel("margin (%)")
        for ax in axes[len(x_cols) :]:
            ax.set_visible(False)
        fig.savefig(out_dir / f"scatter_lowess_{t}.png", dpi=160)
        plt.close(fig)


def plot_spearman_heatmap(df_wide: pd.DataFrame, feature_cols: list[str], out_dir: Path) -> None:
    _ensure_dir(out_dir)
    cols = [c for c in feature_cols if c in df_wide.columns] + [t for t in TARGETS if t in df_wide.columns]
    corr = df_wide[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
    ax.set_title("Spearman correlation (features + targets)")
    fig.savefig(out_dir / "spearman_heatmap.png", dpi=160)
    plt.close(fig)


def fit_elasticnet_separate(
    data: PreparedData, out_dir: Path, *, seed: int = 42
) -> pd.DataFrame:
    _ensure_dir(out_dir)
    df = data.wide
    metrics: list[dict[str, object]] = []

    numeric_features = data.feature_cols
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        ],
        remainder="drop",
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    for target in TARGETS:
        d = df[numeric_features + [target]].dropna(subset=[target]).copy()
        if len(d) < 50:
            continue

        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0],
            alphas=None,
            cv=cv,
            random_state=seed,
            max_iter=50_000,
        )
        pipe = Pipeline([("pre", pre), ("model", model)])

        y = d[target].to_numpy()
        y_pred = cross_val_predict(pipe, d[numeric_features], y, cv=cv)
        row = {"model": "elasticnet_separate", "target": target, **_metric_row(y, y_pred)}
        metrics.append(row)

        pipe.fit(d[numeric_features], y)
        coef = pipe.named_steps["model"].coef_
        coef_df = pd.DataFrame({"feature": numeric_features, "coef": coef})
        coef_df.to_csv(out_dir / f"elasticnet_coef_{target}.csv", index=False)

    metrics_df = pd.DataFrame(metrics).sort_values(["model", "target"])
    metrics_df.to_csv(out_dir / "metrics_elasticnet_separate.csv", index=False)
    return metrics_df


def fit_tree_separate(
    data: PreparedData,
    out_dir: Path,
    *,
    max_depth: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    _ensure_dir(out_dir)
    df = data.wide
    metrics: list[dict[str, object]] = []

    numeric_features = data.feature_cols
    pre = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
        ],
        remainder="drop",
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    for target in TARGETS:
        d = df[numeric_features + [target]].dropna(subset=[target]).copy()
        if len(d) < 50:
            continue

        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=max(25, int(0.01 * len(d))),
            random_state=seed,
        )
        pipe = Pipeline([("pre", pre), ("model", tree)])

        y = d[target].to_numpy()
        y_pred = cross_val_predict(pipe, d[numeric_features], y, cv=cv)
        metrics.append({"model": "tree_separate", "target": target, **_metric_row(y, y_pred)})

        pipe.fit(d[numeric_features], y)
        rules = export_text(
            pipe.named_steps["model"],
            feature_names=list(numeric_features),
            decimals=3,
        )
        (out_dir / f"tree_rules_{target}.txt").write_text(rules)

    metrics_df = pd.DataFrame(metrics).sort_values(["model", "target"])
    metrics_df.to_csv(out_dir / "metrics_tree_separate.csv", index=False)
    return metrics_df


def fit_elasticnet_shared(
    data: PreparedData,
    out_dir: Path,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    _ensure_dir(out_dir)
    df = data.long.dropna(subset=["y"]).copy()
    if len(df) < 200:
        return pd.DataFrame()

    base_features = data.feature_cols

    df = df.copy()
    df["is_im"] = (df["margin_kind"] == "im").astype(int)
    df["is_short"] = (df["side"] == "short").astype(int)
    df["is_im_short"] = (df["is_im"] * df["is_short"]).astype(int)

    # Shared model with target-specific slopes via interactions:
    # y ~ base + (base×is_im) + (base×is_short) + (base×is_im_short) + indicators
    for f in base_features:
        df[f"{f}__im"] = df[f] * df["is_im"]
        df[f"{f}__short"] = df[f] * df["is_short"]
        df[f"{f}__im_short"] = df[f] * df["is_im_short"]

    design_cols = (
        list(base_features)
        + [f"{f}__im" for f in base_features]
        + [f"{f}__short" for f in base_features]
        + [f"{f}__im_short" for f in base_features]
        + ["is_im", "is_short", "is_im_short"]
    )

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                design_cols,
            )
        ],
        remainder="drop",
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    base = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0],
        cv=cv,
        random_state=seed,
        max_iter=50_000,
    )
    # Short margins can include very large values (e.g. hard-to-borrow names),
    # so fit in log space to balance the shared objective.
    model = TransformedTargetRegressor(
        regressor=base,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    pipe = Pipeline([("pre", pre), ("model", model)])

    X = df[design_cols]
    y = df["y"].to_numpy()

    y_pred = cross_val_predict(pipe, X, y, cv=cv)
    overall = {"model": "elasticnet_shared", "target": "overall", **_metric_row(y, y_pred)}

    metrics = [overall]
    for target in TARGETS:
        m = df["target"] == target
        if m.sum() == 0:
            continue
        metrics.append(
            {"model": "elasticnet_shared", "target": target, **_metric_row(y[m], y_pred[m])}
        )

    metrics_df = pd.DataFrame(metrics).sort_values(["model", "target"])
    metrics_df.to_csv(out_dir / "metrics_elasticnet_shared.csv", index=False)

    pipe.fit(X, y)
    feature_names = list(design_cols)
    coefs = pipe.named_steps["model"].regressor_.coef_
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs})
    coef_df.to_csv(out_dir / "elasticnet_coef_shared.csv", index=False)

    return metrics_df


def run_all(
    *,
    input_path: Path,
    out_dir: Path,
    target_scale: Literal["decimal", "pct_points"],
    winsor_alpha: float,
    tree_max_depth: int,
    seed: int,
) -> None:
    _ensure_dir(out_dir)
    prepared = load_and_prepare(
        input_path, target_scale=target_scale, winsor_alpha=winsor_alpha
    )

    summary = {
        "input_path": str(input_path),
        "n_rows_wide": int(len(prepared.wide)),
        "n_rows_long": int(len(prepared.long)),
        "target_scale": target_scale,
        "winsor_alpha": winsor_alpha,
        "tree_max_depth": tree_max_depth,
        "seed": seed,
        "missing_counts": prepared.wide.isna().sum().sort_values(ascending=False).head(50).to_dict(),
    }
    _save_json(out_dir / "run_summary.json", summary)

    plot_dir = out_dir / "plots"
    df_plot = prepared.wide.copy()
    # Keep plots in percent points (30 means 30%) regardless of modeling scale.
    if target_scale == "decimal":
        for t in TARGETS:
            df_plot[t] = df_plot[t] * 100.0
    plot_distributions(df_plot, plot_dir)
    plot_scatter_lowess(df_plot, plot_dir)
    plot_spearman_heatmap(df_plot, prepared.feature_cols, plot_dir)

    model_dir = out_dir / "models"
    _ensure_dir(model_dir)
    fit_elasticnet_separate(prepared, model_dir, seed=seed)
    fit_tree_separate(prepared, model_dir, max_depth=tree_max_depth, seed=seed)
    fit_elasticnet_shared(prepared, model_dir, seed=seed)
