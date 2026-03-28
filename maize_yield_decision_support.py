"""
Environment-specific maize yield decision-support pipeline.

Builds an Option B-style model for plant breeding:
- Predicts yield by line within each environment (location-year).
- Uses phenotype history, environmental data, and genetic markers.
- Adds explicit genotype-by-environment interaction features.
- Evaluates with grouped CV that respects environment boundaries.
- Produces ranked candidate lines per environment for advancement.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
DEFAULT_REAL_DATA_ROOT = Path(r"C:\Users\kensm\OneDrive\Desktop\Real Hackathon DataSets")


@dataclass
class DataBundle:
    phenotype: pd.DataFrame
    environment: pd.DataFrame
    genomic: pd.DataFrame


@dataclass
class BuildResult:
    model_df: pd.DataFrame
    feature_df: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    id_cols: List[str]
    feature_cols: List[str]
    categorical_features: List[str]
    numeric_features: List[str]


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = [re.sub(r"[^0-9A-Za-z_]+", "_", str(c).strip()).upper() for c in df.columns]
    used: Dict[str, int] = {}
    unique: List[str] = []
    for col in cleaned:
        if col not in used:
            used[col] = 0
            unique.append(col)
        else:
            used[col] += 1
            unique.append(f"{col}_{used[col]}")
    out = df.copy()
    out.columns = unique
    return out


def read_csv_clean(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return clean_columns(df)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_cohort_from_genomic_path(genomic_path: str | Path) -> Optional[str]:
    name = Path(genomic_path).name.upper()
    match = re.search(r"\b(C[12])\b", name)
    if match:
        return match.group(1)
    if name.startswith("C1"):
        return "C1"
    if name.startswith("C2"):
        return "C2"
    return None


def resolve_phenotype_path(
    phenotype_path: Optional[str],
    genomic_path: str | Path,
) -> str:
    if phenotype_path:
        return phenotype_path

    cohort = infer_cohort_from_genomic_path(genomic_path) or "C1"
    phenotype_filename = f"{cohort}_Phenotype_Data_V2.csv"
    candidates = [
        Path.cwd() / phenotype_filename,
        DEFAULT_REAL_DATA_ROOT / phenotype_filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not auto-resolve phenotype file. "
        f"Expected '{phenotype_filename}' in current directory or "
        f"'{DEFAULT_REAL_DATA_ROOT}'. "
        "Please pass --phenotype-path explicitly."
    )


def normalize_key(value: object) -> str:
    if pd.isna(value):
        return ""
    txt = str(value).strip().upper()
    txt = re.sub(r"[^A-Z0-9]", "", txt)
    txt = re.sub(r"^PID", "", txt)
    txt = txt.lstrip("0") or "0"
    return txt


def infer_column(
    columns: Sequence[str], candidates: Sequence[str], required: bool = True
) -> Optional[str]:
    colset = {c.upper(): c for c in columns}
    for cand in candidates:
        if cand.upper() in colset:
            return colset[cand.upper()]
    if required:
        raise ValueError(f"Could not infer required column from candidates: {candidates}")
    return None


def is_mostly_numeric(series: pd.Series, min_success_rate: float = 0.8) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False
    converted = pd.to_numeric(non_null, errors="coerce")
    success_rate = converted.notna().mean()
    return bool(success_rate >= min_success_rate)


def infer_line_key(
    phenotype: pd.DataFrame,
    genomic: pd.DataFrame,
    preferred_line_key: Optional[str] = None,
    preferred_genomic_key: Optional[str] = None,
) -> Tuple[str, str]:
    genomic_key = preferred_genomic_key or genomic.columns[0]
    if preferred_line_key:
        return preferred_line_key, genomic_key

    line_candidates = [
        "LINE_UNIQUE_ID",
        "LINE",
        "GERMPLASM_ID",
        "GERMPLASM_ID_TESTER",
        "GENOTYPE_ID",
    ]
    best_col = None
    best_overlap = -1

    geno_vals = set(genomic[genomic_key].map(normalize_key))
    geno_vals.discard("")
    if not geno_vals:
        raise ValueError("Genomic key column appears empty after normalization.")

    for cand in line_candidates:
        if cand not in phenotype.columns:
            continue
        pvals = set(phenotype[cand].map(normalize_key))
        pvals.discard("")
        if not pvals:
            continue
        overlap = len(pvals & geno_vals)
        if overlap > best_overlap:
            best_overlap = overlap
            best_col = cand

    if best_col is None or best_overlap <= 0:
        raise ValueError(
            "Could not infer phenotype↔genomic line key overlap. "
            "Pass --line-key and/or --genomic-key explicitly."
        )
    return best_col, genomic_key


def preprocess_genomics(
    genomic_df: pd.DataFrame,
    genomic_key_col: str,
    max_genetic_components: int,
) -> pd.DataFrame:
    g = genomic_df.copy()
    g = g.rename(columns={genomic_key_col: "GENOTYPE_ID"})
    marker_cols = [c for c in g.columns if c != "GENOTYPE_ID"]
    if not marker_cols:
        raise ValueError("No marker columns found in genomic data.")

    X_markers = g[marker_cols].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="most_frequent")
    X_imp = imputer.fit_transform(X_markers)

    n_samples, n_features = X_imp.shape
    n_comp = min(max_genetic_components, n_samples - 1, n_features)
    if n_comp < 1:
        raise ValueError("Not enough genomic samples/features for PCA.")

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X_imp)
    pca = PCA(n_components=n_comp, random_state=RANDOM_STATE)
    comps = pca.fit_transform(X_scaled)

    pc_cols = [f"GPC_{i+1}" for i in range(n_comp)]
    gpcs = pd.DataFrame(comps, columns=pc_cols, index=g.index)
    out = pd.concat([g[["GENOTYPE_ID"]], gpcs], axis=1)
    out["NORM_GENOTYPE_ID"] = out["GENOTYPE_ID"].map(normalize_key)
    return out


def summarize_environment_features(env_df: pd.DataFrame) -> pd.DataFrame:
    e = env_df.copy()
    numeric_cols = [c for c in e.columns if c not in {"YEAR", "LOC"}]
    for col in numeric_cols:
        e[col] = pd.to_numeric(e[col], errors="coerce")

    month_groups = {
        "PRCP_MEAN": r"_PRCP$",
        "TAVG_MEAN": r"_TAVG$",
        "DP01_SUM": r"_DP01$",
        "DP10_SUM": r"_DP10$",
        "HTDD_SUM": r"_HTDD$",
        "CLDD_SUM": r"_CLDD$",
    }
    for out_col, pattern in month_groups.items():
        cols = [c for c in numeric_cols if re.search(pattern, c)]
        if cols:
            if out_col.endswith("_MEAN"):
                e[out_col] = e[cols].mean(axis=1)
            else:
                e[out_col] = e[cols].sum(axis=1)

    e["ENV_ID"] = e["LOC"].astype(str) + "_" + e["YEAR"].astype(str)
    return e


def build_dataset(
    data: DataBundle,
    target_col: str,
    max_genetic_components: int,
    line_key: Optional[str],
    genomic_key: Optional[str],
    max_gxe_env_features: int,
) -> BuildResult:
    pheno = data.phenotype.copy()
    env = summarize_environment_features(data.environment.copy())
    geno = data.genomic.copy()

    target_col = target_col.upper()
    if target_col not in pheno.columns:
        raise ValueError(f"Target column '{target_col}' not found in phenotype data.")

    year_col = infer_column(pheno.columns, ["YEAR", "YEAR_X", "YEAR_Y"])
    loc_col = infer_column(pheno.columns, ["LOC", "LOCATION"])
    pheno = pheno.rename(columns={year_col: "YEAR", loc_col: "LOC"})

    env_year_col = infer_column(env.columns, ["YEAR"])
    env_loc_col = infer_column(env.columns, ["LOC"])
    env = env.rename(columns={env_year_col: "YEAR", env_loc_col: "LOC"})

    line_key_final, genomic_key_final = infer_line_key(pheno, geno, line_key, genomic_key)
    pheno["LINE_KEY"] = pheno[line_key_final]
    pheno["NORM_LINE_KEY"] = pheno["LINE_KEY"].map(normalize_key)

    geno_pca = preprocess_genomics(geno, genomic_key_final, max_genetic_components)
    geno_merge = geno_pca.drop(columns=["GENOTYPE_ID"]).rename(
        columns={"NORM_GENOTYPE_ID": "NORM_LINE_KEY"}
    )

    merged = pheno.merge(env, on=["YEAR", "LOC"], how="left", suffixes=("", "_ENV"))
    merged = merged.merge(geno_merge, on="NORM_LINE_KEY", how="left")

    merged["ENV_ID"] = merged["LOC"].astype(str) + "_" + merged["YEAR"].astype(str)
    merged[target_col] = pd.to_numeric(merged[target_col], errors="coerce")

    # Past traits from phenotype (numerics other than target and IDs)
    id_like = {
        target_col,
        "YEAR",
        "LOC",
        "ENV_ID",
        "LINE_KEY",
        "NORM_LINE_KEY",
        "PROJECTS_X",
        "PROJECTS_Y",
        "SHORTHAND_X",
        "SHORTHAND_Y",
        "FILE_LIST",
        "CROSS",
    }
    phenotype_numeric = [c for c in pheno.columns if c not in id_like and is_mostly_numeric(pheno[c])]

    env_numeric = [
        c
        for c in env.columns
        if c not in {"YEAR", "LOC", "ENV_ID"}
        and is_mostly_numeric(env[c])
    ]
    genetic_pc_cols = [c for c in merged.columns if c.startswith("GPC_")]

    # Determine top environmental features (variance) to cross with genetic PCs.
    if env_numeric:
        env_var = merged[env_numeric].apply(pd.to_numeric, errors="coerce").var().sort_values(ascending=False)
        top_env_cols = env_var.head(max_gxe_env_features).index.tolist()
    else:
        top_env_cols = []

    interaction_data: Dict[str, pd.Series] = {}
    for gcol in genetic_pc_cols:
        gvals = pd.to_numeric(merged[gcol], errors="coerce")
        for ecol in top_env_cols:
            evals = pd.to_numeric(merged[ecol], errors="coerce")
            interaction_data[f"{gcol}_X_{ecol}"] = gvals * evals
    if interaction_data:
        merged = pd.concat([merged, pd.DataFrame(interaction_data, index=merged.index)], axis=1)

    engineered_cols = [c for c in merged.columns if "_X_" in c]
    numeric_features = sorted(set(phenotype_numeric + env_numeric + genetic_pc_cols + engineered_cols + ["YEAR"]))
    numeric_features = [c for c in numeric_features if c in merged.columns]

    categorical_features = ["LOC", "ENV_ID"]

    model_cols = unique_preserve_order(
        ["LINE_KEY", "YEAR", "LOC", "ENV_ID", target_col] + numeric_features + categorical_features
    )
    model_df = merged[model_cols].copy()
    model_df = model_df.dropna(subset=[target_col]).reset_index(drop=True)

    feature_cols = unique_preserve_order(numeric_features + categorical_features)
    feature_df = model_df[feature_cols].copy()
    for col in numeric_features:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
    target = model_df[target_col].astype(float)
    groups = model_df["ENV_ID"].astype(str)

    return BuildResult(
        model_df=model_df,
        feature_df=feature_df,
        target=target,
        groups=groups,
        id_cols=["LINE_KEY", "YEAR", "LOC", "ENV_ID"],
        feature_cols=feature_cols,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
    )


def build_preprocessor(
    numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> ColumnTransformer:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, list(numeric_features)),
            ("cat", cat_pipe, list(categorical_features)),
        ]
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def grouped_cv_predict(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    preprocessor: ColumnTransformer,
    model,
    n_splits: int,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    unique_groups = groups.nunique()
    folds = min(n_splits, unique_groups)
    if folds < 2:
        raise ValueError("Need at least 2 unique environments for grouped CV.")

    gkf = GroupKFold(n_splits=folds)
    oof_pred = np.full(shape=(len(y),), fill_value=np.nan, dtype=float)
    fold_metrics: List[Dict[str, float]] = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred = pipe.predict(X.iloc[te_idx])
        oof_pred[te_idx] = pred

        m = regression_metrics(y.iloc[te_idx].to_numpy(), pred)
        m["fold"] = fold
        m["n_train"] = int(len(tr_idx))
        m["n_test"] = int(len(te_idx))
        fold_metrics.append(m)

    return oof_pred, fold_metrics


def get_feature_names_from_pipeline(
    preprocessor: ColumnTransformer, numeric_features: Sequence[str], categorical_features: Sequence[str]
) -> List[str]:
    names = list(numeric_features)
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    return names + cat_names


def fit_final_and_explain(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor: ColumnTransformer,
    model,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    permutation_repeats: int,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame]:
    pipe = Pipeline([("prep", preprocessor), ("model", model)])
    pipe.fit(X, y)

    feature_names = get_feature_names_from_pipeline(
        pipe.named_steps["prep"], numeric_features, categorical_features
    )
    raw_importance = pipe.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance_gain": raw_importance})
    fi = fi.sort_values("importance_gain", ascending=False).reset_index(drop=True)

    # Permutation importance gives model-agnostic signal and is easier to explain to breeders.
    perm = permutation_importance(
        pipe,
        X,
        y,
        n_repeats=permutation_repeats,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )
    pi = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_perm_mean": perm.importances_mean,
            "importance_perm_std": perm.importances_std,
        }
    ).sort_values("importance_perm_mean", ascending=False)
    pi = pi.reset_index(drop=True)

    return pipe, fi, pi


def evaluate_by_environment(
    df: pd.DataFrame,
    actual_col: str,
    pred_col: str,
    env_col: str = "ENV_ID",
) -> pd.DataFrame:
    rows = []
    for env_id, chunk in df.groupby(env_col):
        valid = chunk[[actual_col, pred_col]].dropna()
        if len(valid) < 2:
            continue
        m = regression_metrics(valid[actual_col].to_numpy(), valid[pred_col].to_numpy())
        m["env_id"] = env_id
        m["n"] = int(len(valid))
        rows.append(m)
    if not rows:
        return pd.DataFrame(columns=["env_id", "n", "rmse", "mae", "r2"])
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def rank_candidates(df: pd.DataFrame, pred_col: str, top_k: int) -> pd.DataFrame:
    ranked_chunks = []
    for env_id, chunk in df.groupby("ENV_ID"):
        c = chunk.sort_values(pred_col, ascending=False).copy()
        c["RANK_IN_ENV"] = np.arange(1, len(c) + 1)
        ranked_chunks.append(c.head(top_k))
    ranked = pd.concat(ranked_chunks, axis=0).reset_index(drop=True)
    cols = ["ENV_ID", "LOC", "YEAR", "LINE_KEY", pred_col, "RANK_IN_ENV"]
    keep = [c for c in cols if c in ranked.columns]
    return ranked[keep]


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    resolved_phenotype_path = resolve_phenotype_path(args.phenotype_path, args.genomic_path)

    data = DataBundle(
        phenotype=read_csv_clean(resolved_phenotype_path),
        environment=read_csv_clean(args.environment_path),
        genomic=read_csv_clean(args.genomic_path),
    )

    build = build_dataset(
        data=data,
        target_col=args.target_col,
        max_genetic_components=args.max_genetic_components,
        line_key=args.line_key,
        genomic_key=args.genomic_key,
        max_gxe_env_features=args.max_gxe_env_features,
    )

    X = build.feature_df
    y = build.target
    groups = build.groups

    prep_base = build_preprocessor(build.numeric_features, build.categorical_features)
    prep_final = build_preprocessor(build.numeric_features, build.categorical_features)

    baseline_model = Ridge(alpha=args.ridge_alpha, random_state=RANDOM_STATE)
    final_model = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
    )

    baseline_oof, baseline_folds = grouped_cv_predict(
        X=X,
        y=y,
        groups=groups,
        preprocessor=prep_base,
        model=baseline_model,
        n_splits=args.cv_splits,
    )
    final_oof, final_folds = grouped_cv_predict(
        X=X,
        y=y,
        groups=groups,
        preprocessor=prep_final,
        model=final_model,
        n_splits=args.cv_splits,
    )

    baseline_metrics = regression_metrics(y.to_numpy(), baseline_oof)
    final_metrics = regression_metrics(y.to_numpy(), final_oof)

    trained_final, feature_importance, permutation_df = fit_final_and_explain(
        X=X,
        y=y,
        preprocessor=build_preprocessor(build.numeric_features, build.categorical_features),
        model=final_model,
        numeric_features=build.numeric_features,
        categorical_features=build.categorical_features,
        permutation_repeats=args.permutation_repeats,
    )

    fitted_predictions = trained_final.predict(X)

    prediction_df = build.model_df[build.id_cols].copy()
    prediction_df["OBSERVED_YIELD"] = y
    prediction_df["PREDICTED_YIELD_CV_BASELINE"] = baseline_oof
    prediction_df["PREDICTED_YIELD_CV_FINAL"] = final_oof
    prediction_df["PREDICTED_YIELD_FULL_MODEL"] = fitted_predictions

    env_eval = evaluate_by_environment(
        prediction_df,
        actual_col="OBSERVED_YIELD",
        pred_col="PREDICTED_YIELD_CV_FINAL",
        env_col="ENV_ID",
    )
    ranked = rank_candidates(prediction_df, pred_col="PREDICTED_YIELD_FULL_MODEL", top_k=args.top_k)

    # Save artifacts
    prediction_df.to_csv(output_dir / "prediction_results.csv", index=False)
    ranked.to_csv(output_dir / "ranked_candidates_per_environment.csv", index=False)
    feature_importance.to_csv(output_dir / "feature_importance_model_gain.csv", index=False)
    permutation_df.to_csv(output_dir / "feature_importance_permutation.csv", index=False)
    env_eval.to_csv(output_dir / "environment_level_metrics.csv", index=False)
    pd.DataFrame(baseline_folds).to_csv(output_dir / "baseline_fold_metrics.csv", index=False)
    pd.DataFrame(final_folds).to_csv(output_dir / "final_fold_metrics.csv", index=False)

    summary = {
        "decision_support_statement": (
            "A model that predicts maize yield using genetic and environmental data "
            "to help select the best lines under limited resources."
        ),
        "strategy": "Option B - environment-specific prediction with genotype, environment, and past traits.",
        "why_environment_specific": [
            "More realistic for breeding decisions because selections are made per target environment.",
            "Captures local adaptation by allowing environment-specific ranking outcomes.",
            "Improves targeted recommendations for each location-year combination.",
            "Better handles genotype-by-environment effects via explicit GxE interaction features and grouped validation.",
        ],
        "dataset_shape": {"n_rows": int(len(prediction_df)), "n_features": int(X.shape[1])},
        "target_column": args.target_col.upper(),
        "phenotype_path_used": str(resolved_phenotype_path),
        "environment_path_used": str(args.environment_path),
        "genomic_path_used": str(args.genomic_path),
        "baseline_grouped_cv_metrics": baseline_metrics,
        "final_grouped_cv_metrics": final_metrics,
        "model_comparison": {
            "rmse_improvement": baseline_metrics["rmse"] - final_metrics["rmse"],
            "mae_improvement": baseline_metrics["mae"] - final_metrics["mae"],
            "r2_improvement": final_metrics["r2"] - baseline_metrics["r2"],
        },
        "outputs": {
            "prediction_results": str(output_dir / "prediction_results.csv"),
            "ranked_candidates_per_environment": str(output_dir / "ranked_candidates_per_environment.csv"),
            "feature_importance_model_gain": str(output_dir / "feature_importance_model_gain.csv"),
            "feature_importance_permutation": str(output_dir / "feature_importance_permutation.csv"),
            "environment_level_metrics": str(output_dir / "environment_level_metrics.csv"),
        },
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Environment-specific maize yield prediction pipeline for decision support."
    )
    parser.add_argument(
        "--phenotype-path",
        default=None,
        help=(
            "Path to phenotype CSV. If omitted, auto-uses cohort-specific "
            "Phenotype_Data_V2 file (e.g., C1_Phenotype_Data_V2.csv for C1 genotypes)."
        ),
    )
    parser.add_argument("--environment-path", required=True, help="Path to environment CSV.")
    parser.add_argument("--genomic-path", required=True, help="Path to genomic marker CSV.")
    parser.add_argument("--target-col", default="YLD_BE", help="Yield target column in phenotype data.")
    parser.add_argument(
        "--line-key",
        default=None,
        help="Phenotype column to join with genomic IDs (optional; auto-infer by overlap if omitted).",
    )
    parser.add_argument(
        "--genomic-key",
        default=None,
        help="Genomic ID column (optional; defaults to first genomic CSV column).",
    )
    parser.add_argument("--max-genetic-components", type=int, default=30, help="Max PCA components for markers.")
    parser.add_argument(
        "--max-gxe-env-features",
        type=int,
        default=10,
        help="Number of top environmental covariates used in GxE interaction features.",
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="GroupKFold splits by environment.")
    parser.add_argument("--ridge-alpha", type=float, default=1.0, help="Baseline ridge alpha.")
    parser.add_argument("--n-estimators", type=int, default=600, help="Gradient boosting estimators.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Gradient boosting learning rate.")
    parser.add_argument("--max-depth", type=int, default=3, help="Gradient boosting tree depth.")
    parser.add_argument("--subsample", type=float, default=0.8, help="Gradient boosting subsample fraction.")
    parser.add_argument("--permutation-repeats", type=int, default=5, help="Permutation repeats.")
    parser.add_argument("--top-k", type=int, default=10, help="Top candidate lines per environment.")
    parser.add_argument("--output-dir", default="outputs", help="Output folder for artifacts.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
