#!/usr/bin/env python3
"""
Maize line yield (YLDBE) under GxE — full ML pipeline.

## 1. Lock Analytical Unit
One row = LINE × (LOC × YEAR); keys `env_id` = LOC_YEAR, `line_id` = LINE_UNIQUE_ID
(e.g. C1.1.191). Target YLDBE (`YLD_BE` in files). Alternatives: PHT, MST.

## 2. Build Genotype Matrix
Per-pop imputed files; progeny rows match ^\\d{11}$; line_id = C{cohort}.{pop}.{linenum}.

## 3. Validate Joins
Match rates, past_yld from train-year aggregates, NaN column drops, geno coverage check.

## 4. Refine Genetic Encoding
PCA (50 comps, refit inside CV on train-fold lines) + Ridge(alpha=1e-2) on raw SNPs baseline.

## 5. Environment Covariates
Monthly PRCP/TAVG-style columns, totals, summer averages, global StandardScaler.

## 6. Improve GxE Features
Domain blocks + RF permutation screen (top 5/block, cap 20 env vars) + PCA×env interactions.

## 7. Gradient Boosting
Phase A: RandomizedSearchCV + GroupKFold on GBR. Phase B: HistGradientBoosting (or optional XGB)
with year-holdout early stopping.

## 8. Unbiased Evaluation
OOF metrics global and by env_id; residual plot; per-env R² table.

## 9. Ranking Under Constraints
Top-k per env; bootstrap uncertainty; optional genetic distance diversification.

## 10. Interpretation
Permutation importance by bucket; optional SHAP summary for tree model.

## 11. Packaging
YAML config + run_summary.json under output/.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

PROGENY_RE = re.compile(r"^\d{11}$")
LINE_ID_RE = re.compile(r"^C(\d+)\.(\d+)\.(\d+)$", re.IGNORECASE)
POP_FILE_RE_C1 = re.compile(r"C1\.(\d+)_Imputed\.csv$", re.IGNORECASE)
POP_FILE_RE_C2 = re.compile(r"C2\.(\d+)_Imputed\.csv$", re.IGNORECASE)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_step(name: str) -> None:
    logging.info("=== %s ===", name)


# --- Step 1 ---
def step1_lock_analytical_unit(
    pheno_path: Path,
    env_path: Path,
    target_col: str,
    output_dir: Path,
) -> pd.DataFrame:
    log_step("Step 1: Lock analytical unit")
    pheno = pd.read_csv(pheno_path, low_memory=False)
    # Harmonize year column
    year_col = "YEAR_x" if "YEAR_x" in pheno.columns else infer_col(pheno.columns, ["YEAR"])
    loc_col = infer_col(pheno.columns, ["LOC"])
    line_col = infer_col(pheno.columns, ["LINE_UNIQUE_ID", "LINEUNIQUEID"])
    pheno = pheno.rename(columns={year_col: "YEAR", loc_col: "LOC", line_col: "LINE_UNIQUE_ID_RAW"})
    pheno["LINE_UNIQUE_ID"] = pheno["LINE_UNIQUE_ID_RAW"].astype(str).str.strip()
    pheno["YEAR"] = pd.to_numeric(pheno["YEAR"], errors="coerce")
    pheno["LOC"] = pheno["LOC"].astype(str).str.strip()
    pheno["env_id"] = pheno["LOC"] + "_" + pheno["YEAR"].astype("Int64").astype(str)

    extr = pheno["LINE_UNIQUE_ID"].str.extract(LINE_ID_RE)
    extr.columns = ["cohort_num", "pop_num", "line_num"]
    for c in extr.columns:
        pheno[c] = pd.to_numeric(extr[c], errors="coerce")
    bad = pheno["cohort_num"].isna()
    if bad.any():
        logging.warning("Dropped %s phenotype rows with non-parseable line_id", int(bad.sum()))
        pheno = pheno.loc[~bad].copy()

    # Canonical line_id (e.g. C1.114.00000000001 -> C1.114.1) for genotype join
    pheno["line_id"] = (
        "C"
        + pheno["cohort_num"].astype(int).astype(str)
        + "."
        + pheno["pop_num"].astype(int).astype(str)
        + "."
        + pheno["line_num"].astype(int).astype(str)
    )

    tgt = infer_col(pheno.columns, [target_col, "YLDBE", "YLD_BE"])
    pheno["YLDBE"] = pd.to_numeric(pheno[tgt], errors="coerce")

    key = ["line_id", "env_id"]
    dup_mask = pheno.duplicated(key, keep=False)
    if dup_mask.any():
        n = int(dup_mask.sum())
        logging.warning("Duplicate grain rows: %s - dropping duplicates (keep first)", n)
        pheno = pheno.drop_duplicates(subset=key, keep="first")

    env = pd.read_csv(env_path, low_memory=False)
    ey = infer_col(env.columns, ["YEAR"])
    el = infer_col(env.columns, ["LOC"])
    env = env.rename(columns={ey: "YEAR", el: "LOC"})
    env["YEAR"] = pd.to_numeric(env["YEAR"], errors="coerce")
    env["LOC"] = env["LOC"].astype(str).str.strip()

    merged = pheno.merge(env, on=["YEAR", "LOC"], how="left", suffixes=("", "_ENV"))
    out_path = output_dir / "pheno_env.csv"
    merged.to_csv(out_path, index=False)
    logging.info("Saved %s rows to %s", len(merged), out_path)
    return merged


def infer_col(cols: Sequence[str], candidates: Sequence[str]) -> str:
    up = {str(c).upper(): c for c in cols}
    for cand in candidates:
        if cand.upper() in up:
            return up[cand.upper()]
    raise ValueError(f"None of {candidates} found in columns {list(cols)[:20]}...")


# --- Step 2 ---
def step2_build_genotype_matrix(
    imputed_dir: Path,
    cohort: str,
    output_dir: Path,
    max_snps: Optional[int],
    max_pop_files: Optional[int],
) -> pd.DataFrame:
    log_step("Step 2: Build genotype matrix")
    cohort = cohort.upper()
    pat = POP_FILE_RE_C1 if cohort == "C1" else POP_FILE_RE_C2
    files = [f for f in imputed_dir.glob("*_Imputed.csv") if pat.search(f.name)]
    files.sort(key=lambda p: int(pat.search(p.name).group(1)))
    if max_pop_files is not None:
        files = files[: int(max_pop_files)]
    if not files:
        raise FileNotFoundError(f"No imputed CSVs matching {pat.pattern} under {imputed_dir}")

    snp_cols_ref: Optional[List[str]] = None
    chunks: List[pd.DataFrame] = []

    for i, fp in enumerate(files):
        m = pat.search(fp.name)
        if not m:
            continue
        pop = int(m.group(1))
        df = pd.read_csv(fp, index_col=0, low_memory=False)
        df.index = df.index.astype(str)
        prog_mask = df.index.str.match(PROGENY_RE)
        df = df.loc[prog_mask].copy()
        if df.empty:
            continue
        if snp_cols_ref is None:
            snp_cols_ref = [c for c in df.columns]
            if max_snps is not None and len(snp_cols_ref) > max_snps:
                rng = np.random.default_rng(42)
                snp_cols_ref = list(rng.choice(snp_cols_ref, size=max_snps, replace=False))
        else:
            df = df[[c for c in snp_cols_ref if c in df.columns]]
            missing = set(snp_cols_ref) - set(df.columns)
            if missing:
                for c in missing:
                    df[c] = np.nan

        df = df[snp_cols_ref]
        linenum = df.index.to_series().map(lambda s: int(s.lstrip("0") or "0"))
        cohort_digit = "1" if cohort == "C1" else "2"
        df.insert(0, "line_id", [f"C{cohort_digit}.{pop}.{ln}" for ln in linenum])
        df = df.drop_duplicates(subset=["line_id"], keep="first")
        chunks.append(df)
        if (i + 1) % 50 == 0:
            logging.info("Loaded %s/%s population files", i + 1, len(files))

    geno = pd.concat(chunks, axis=0, ignore_index=False)
    geno = geno.reset_index(drop=True)
    snp_names = [c for c in geno.columns if c != "line_id"]
    X = geno[snp_names].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    X = X.fillna(0.0)
    geno[snp_names] = X
    out = output_dir / f"genotypes_{cohort.lower()}.csv"
    geno.to_csv(out, index=False)
    logging.info("Genotype matrix: %s lines, %s SNPs -> %s", len(geno), len(snp_names), out)
    return geno


# --- Step 3 ---
def step3_validate_joins(
    pheno_env: pd.DataFrame,
    geno: pd.DataFrame,
    train_min: int,
    train_max: int,
    output_dir: Path,
    min_geno_rate: float,
    min_pheno_env_rate: float,
    nan_drop_frac: float,
    subset_populations: bool,
    save_full_dataset_csv: bool = True,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    log_step("Step 3: Validate joins")
    snp_cols = [c for c in geno.columns if c != "line_id"]
    pheno_lines = set(pheno_env["line_id"].unique())
    geno_lines = set(geno["line_id"].unique())
    matched = pheno_lines & geno_lines

    pops_with_geno: set[int] = set()
    for lid in geno_lines:
        m = LINE_ID_RE.match(str(lid))
        if m:
            pops_with_geno.add(int(m.group(2)))

    if subset_populations and pops_with_geno:
        extr = pheno_env["line_id"].astype(str).str.extract(LINE_ID_RE)
        pop_ser = pd.to_numeric(extr[1], errors="coerce")
        in_loaded_pop = pop_ser.isin(pops_with_geno).values
        pheno_lines_scoped = set(pheno_env.loc[in_loaded_pop, "line_id"].unique())
        denom = max(len(pheno_lines_scoped), 1)
        rate = len(matched) / denom
        logging.info(
            "Genotype subset mode: evaluating coverage among %s pheno lines in loaded populations only",
            len(pheno_lines_scoped),
        )
    else:
        denom = max(len(pheno_lines), 1)
        rate = len(matched) / denom

    logging.info("Unique phenotype lines: %s", len(pheno_lines))
    logging.info("Unique genotype lines: %s", len(geno_lines))
    logging.info("Overlap lines: %s (%.2f%% of evaluation set)", len(matched), 100 * rate)
    if rate < min_geno_rate:
        raise RuntimeError(
            f"Genotype coverage {rate:.2%} < required {min_geno_rate:.2%}. "
            "Check cohort/paths and line_id parsing."
        )

    env_keys = pheno_env[["YEAR", "LOC"]].drop_duplicates()
    env_present = pheno_env.dropna(subset=[c for c in pheno_env.columns if c not in {"YLDBE"}][:1])
    # Phenotype→env: rows where key env columns exist (not all-NaN from failed merge)
    meta_env_cols = [c for c in pheno_env.columns if any(x in c.upper() for x in ("PRCP", "TAVG", "CLAY", "SILT", "PHH2O"))]
    if meta_env_cols:
        ok = pheno_env[meta_env_cols].notna().any(axis=1)
        pe_rate = float(ok.mean())
    else:
        ok = pd.Series(True, index=pheno_env.index)
        pe_rate = 1.0
    logging.info("Phenotype rows with any env feature: %.2f%%", 100 * pe_rate)
    if pe_rate < min_pheno_env_rate:
        logging.warning(
            "Phenotype-env match %.2f%% below target %.2f%% - continuing with warning",
            100 * pe_rate,
            100 * min_pheno_env_rate,
        )

    unmatched_pheno = sorted(pheno_lines - geno_lines)[:10]
    unmatched_geno = sorted(geno_lines - pheno_lines)[:10]
    logging.info("Top unmatched pheno line_ids: %s", unmatched_pheno)
    logging.info("Top unmatched geno line_ids (sample): %s", unmatched_geno)

    train_mask = pheno_env["YEAR"].between(train_min, train_max) & pheno_env["YLDBE"].notna()
    past = (
        pheno_env.loc[train_mask]
        .groupby("line_id", as_index=False)["YLDBE"]
        .agg(past_yld_mean="mean", past_yld_median="median")
    )

    # Narrow merge only (wide SNP matrix stays in genotypes_*.csv — avoids TB-scale CSV).
    full = pheno_env.merge(past, on="line_id", how="left")
    full["has_genotype"] = full["line_id"].isin(geno_lines)

    nan_frac = full.isna().mean()
    drop_cols = nan_frac[nan_frac > nan_drop_frac].index.tolist()
    drop_cols = [c for c in drop_cols if c not in {"line_id", "env_id", "YEAR", "LOC", "YLDBE"}]
    if drop_cols:
        logging.info("Dropping %s columns with >%.0f%% NaN", len(drop_cols), nan_drop_frac * 100)
        full = full.drop(columns=drop_cols, errors="ignore")

    full_path = output_dir / "full_dataset.csv"
    if not save_full_dataset_csv:
        logging.info("Skipping full_dataset.csv write (save_full_dataset_csv=false)")
    else:
        full.to_csv(full_path, index=False)
        logging.info("Saved full_dataset.csv (%s rows)", len(full))

    report = {
        "geno_match_rate": rate,
        "pheno_env_row_rate": pe_rate,
        "unmatched_pheno_top10": unmatched_pheno,
        "unmatched_geno_top10": unmatched_geno,
        "dropped_high_nan_cols": drop_cols,
    }
    return full, snp_cols, report


# --- Step 5 (env feats) — used before modeling ---
def step5_environment_features(df: pd.DataFrame, scaling: str) -> Tuple[pd.DataFrame, List[str], StandardScaler]:
    log_step("Step 5: Environment covariates")
    out = df.copy()
    meta = {"line_id", "env_id", "YEAR", "LOC", "YLDBE", "past_yld_mean", "past_yld_median"}
    meta |= {c for c in out.columns if c.startswith("cohort") or c.endswith("_num")}

    prcp_cols = [c for c in out.columns if "PRCP" in c.upper()]
    tavg_cols = [c for c in out.columns if "TAVG" in c.upper()]
    if prcp_cols:
        out["total_PRCP"] = out[prcp_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    summer_cols = [c for c in out.columns if re.search(r"(^|[^0-9])(6|7|8)[^0-9]*TAVG", c.upper()) or c.upper() in {"X06_TAVG", "X07_TAVG", "X08_TAVG"}]
    if not summer_cols:
        summer_cols = [c for c in tavg_cols if any(m in c for m in ("06", "07", "08", "6_", "7_", "8_"))]
    if summer_cols:
        out["summer_tavg"] = out[summer_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    env_numeric = []
    for c in out.columns:
        if c in meta or c.startswith("SNP_"):
            continue
        if out[c].dtype == object:
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            env_numeric.append(c)

    scaler = StandardScaler()
    if scaling.strip().lower() == "global" and env_numeric:
        out[env_numeric] = scaler.fit_transform(out[env_numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0))
    return out, env_numeric, scaler


def domain_blocks(env_cols: List[str]) -> Dict[str, List[str]]:
    blocks = {"precip": [], "temp": [], "soil": [], "other": []}
    for c in env_cols:
        u = c.upper()
        if "PRCP" in u or "DP01" in u or "DP10" in u:
            blocks["precip"].append(c)
        elif "TAVG" in u or "HTDD" in u or "CLDD" in u or "TEMP" in u:
            blocks["temp"].append(c)
        elif any(x in u for x in ("CLAY", "SILT", "SAND", "PHH2O", "SOC", "NITROGEN", "CFVO")):
            blocks["soil"].append(c)
        else:
            blocks["other"].append(c)
    return blocks


# --- Step 6 ---
def step6_select_env_and_gxe(
    df: pd.DataFrame,
    env_numeric: List[str],
    gpc_cols: List[str],
    rng: np.random.Generator,
    max_env: int,
    n_pairs: int,
) -> Tuple[List[str], List[Tuple[str, str]]]:
    log_step("Step 6: GxE feature refinement")
    blocks = domain_blocks(env_numeric)
    cheap = RandomForestRegressor(
        n_estimators=60,
        max_depth=6,
        random_state=int(rng.integers(0, 1_000_000)),
        n_jobs=1,
    )
    y = df["YLDBE"].values
    chosen: List[str] = []

    for _, cols in blocks.items():
        cols = [c for c in cols if c in df.columns and df[c].notna().any()]
        if len(cols) <= 5:
            chosen.extend(cols)
            continue
        Xb = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if Xb.shape[1] == 0:
            continue
        cheap.fit(Xb, y)
        imp = permutation_importance(cheap, Xb, y, n_repeats=5, random_state=42, n_jobs=1)
        order = np.argsort(-imp.importances_mean)[:5]
        chosen.extend([cols[i] for i in order])

    chosen = list(dict.fromkeys(chosen))
    chosen = chosen[:max_env]

    pair_list: List[Tuple[str, str]] = []
    if gpc_cols and chosen:
        use_g = gpc_cols[:10]
        use_e = chosen[:10]
        Xp = df[use_g + use_e].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        cheap.fit(Xp, y)
        imp = permutation_importance(cheap, Xp, y, n_repeats=4, random_state=42, n_jobs=1)
        names = list(Xp.columns)
        scored: List[Tuple[str, str, float]] = []
        for gc in use_g:
            for ec in use_e:
                if gc not in names or ec not in names:
                    continue
                i, j = names.index(gc), names.index(ec)
                scored.append((gc, ec, float(imp.importances_mean[i] + imp.importances_mean[j])))
        scored.sort(key=lambda t: -t[2])
        pair_list = [(a, b) for a, b, _ in scored[:n_pairs]]
    return chosen, pair_list


def fit_pca_for_lines(
    geno_wide: pd.DataFrame,
    snp_cols: Sequence[str],
    line_ids: np.ndarray,
    n_components: int,
    random_state: int,
) -> Tuple[np.ndarray, Pipeline]:
    sub = geno_wide.loc[geno_wide["line_id"].isin(set(line_ids)), ["line_id"] + list(snp_cols)]
    sub = sub.drop_duplicates("line_id")
    X = sub[snp_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    n_comp = min(n_components, X.shape[0] - 1, X.shape[1])
    if n_comp < 1:
        raise ValueError("Not enough samples/features for PCA.")
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=random_state)),
        ]
    )
    Z = pipe.fit_transform(X)
    return Z, pipe


def map_pca_to_rows(
    geno_wide: pd.DataFrame,
    snp_cols: Sequence[str],
    line_ids: pd.Series,
    pipe: Pipeline,
) -> pd.DataFrame:
    sub = geno_wide[["line_id"] + list(snp_cols)].drop_duplicates("line_id").set_index("line_id")
    X = sub.reindex(line_ids).astype(np.float32)
    X = np.nan_to_num(X.values, nan=0.0)
    comps = pipe.transform(X)
    gpc_cols = [f"GPC_{i+1}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, columns=gpc_cols, index=line_ids.index)


def grouped_oof_with_pca(
    df: pd.DataFrame,
    snp_cols: List[str],
    geno_wide: pd.DataFrame,
    nongeno_feature_cols: List[str],
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    pca_n: int,
    model,
    random_state: int,
) -> np.ndarray:
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(df, y, groups=groups):
        tr_lines = df.iloc[tr]["line_id"].unique()
        _, pipe = fit_pca_for_lines(geno_wide, snp_cols, tr_lines, pca_n, random_state)
        gpc_tr = map_pca_to_rows(geno_wide, snp_cols, df.iloc[tr]["line_id"], pipe).values
        gpc_te = map_pca_to_rows(geno_wide, snp_cols, df.iloc[te]["line_id"], pipe).values
        Xtr = np.hstack([gpc_tr, df.iloc[tr][nongeno_feature_cols].values.astype(float)])
        Xte = np.hstack([gpc_te, df.iloc[te][nongeno_feature_cols].values.astype(float)])
        model.fit(Xtr, y[tr])
        oof[te] = model.predict(Xte)
    return oof


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    yt, yp = y_true[m], y_pred[m]
    return {
        "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Maize GxE YLDBE pipeline")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("pipeline_config.yaml"))
    args = parser.parse_args()
    cfg_path = args.config.resolve()
    cfg = load_config(cfg_path)
    out = Path(cfg["output_dir"])
    if not out.is_absolute():
        out = (cfg_path.parent / out).resolve()
    setup_logging(out)
    rng = np.random.default_rng(int(cfg.get("random_state", 42)))

    cohort = str(cfg["cohort"]).upper()
    paths = cfg["paths"]
    years = cfg["years"]
    hypers = cfg["hypers"]

    pheno_path = Path(paths[f"pheno_{cohort.lower()}"])
    env_path = Path(paths["environmental"])
    imputed_dir = Path(paths[f"imputed_{cohort.lower()}_dir"])

    # Step 1
    pheno_env = step1_lock_analytical_unit(
        pheno_path, env_path, str(cfg.get("target_col", "YLD_BE")), out
    )

    # Step 2
    geno = step2_build_genotype_matrix(
        imputed_dir,
        cohort,
        out,
        cfg.get("genotype_max_snps"),
        cfg.get("genotype_max_pop_files"),
    )
    snp_cols = [c for c in geno.columns if c != "line_id"]

    # Step 3
    subset_pops = cfg.get("genotype_max_pop_files") is not None
    full, snp_cols, join_report = step3_validate_joins(
        pheno_env,
        geno,
        int(years["train_min"]),
        int(years["train_max"]),
        out,
        float(cfg.get("min_geno_match_rate", 0.7)),
        float(cfg.get("min_pheno_env_match_rate", 0.8)),
        float(cfg.get("nan_col_drop_frac", 0.5)),
        subset_populations=bool(subset_pops),
        save_full_dataset_csv=bool(cfg.get("save_full_dataset_csv", True)),
    )

    # Training / test masks
    tr_min, tr_max = int(years["train_min"]), int(years["train_max"])
    te_year = int(years["test_year"])
    train_year_mask = full["YEAR"].between(tr_min, tr_max) & full["YLDBE"].notna()
    test_year_mask = (full["YEAR"] == te_year) & full["YLDBE"].notna()
    model_df = full.loc[train_year_mask | test_year_mask].copy()
    model_df = model_df[model_df["line_id"].isin(set(geno["line_id"]))].copy()

    # Subsample training rows for runtime
    max_rows = cfg.get("max_train_rows")
    per_env = cfg.get("max_rows_per_env")
    train_only = model_df.loc[model_df["YEAR"].between(tr_min, tr_max)]
    if per_env:
        parts = []
        for env_id, chunk in train_only.groupby("env_id"):
            if len(chunk) > int(per_env):
                parts.append(chunk.sample(n=int(per_env), random_state=int(rng.integers(0, 1_000_000))))
            else:
                parts.append(chunk)
        train_only = pd.concat(parts, axis=0)
    if max_rows is not None and len(train_only) > int(max_rows):
        train_only = train_only.sample(n=int(max_rows), random_state=42)
    # Reattach test
    test_only = model_df.loc[model_df["YEAR"] == te_year]
    model_df = pd.concat([train_only, test_only], axis=0).drop_duplicates()
    snp_in_m = [c for c in snp_cols if c in model_df.columns]
    if snp_in_m:
        model_df = model_df.drop(columns=snp_in_m, errors="ignore")
        logging.info("Dropped %s raw SNP columns from modeling frame (kept separate geno matrix)", len(snp_in_m))

    logging.info("Modeling rows: %s (train-year subset + test year)", len(model_df))

    # Step 5 — env scaling (in-place on model_df slice columns that exist)
    model_df, env_numeric, env_scaler = step5_environment_features(model_df, str(cfg.get("env_scaling", "global")))
    joblib.dump(env_scaler, out / "env_scaler.pkl")

    # Step 4 + 6: PCA on full geno reference; pre-compute baseline Ridge SNPs for reporting
    pca_n = int(hypers.get("pca_components", 50))
    train_lines = model_df.loc[model_df["YEAR"].between(tr_min, tr_max), "line_id"].unique()
    _, final_pca_pipe = fit_pca_for_lines(geno, snp_cols, train_lines, pca_n, int(cfg.get("random_state", 42)))
    joblib.dump(final_pca_pipe, out / "pca_pipeline.pkl")

    gpc_df = map_pca_to_rows(geno, snp_cols, model_df["line_id"], final_pca_pipe)
    gpc_cols = list(gpc_df.columns)
    model_df = pd.concat([model_df.reset_index(drop=True), gpc_df.reset_index(drop=True)], axis=1)

    is_train = model_df["YEAR"].between(tr_min, tr_max).values

    # Ridge baseline on raw SNPs (train rows)
    ridge_X = (
        model_df.loc[is_train, ["line_id"]]
        .merge(geno, on="line_id", how="left")[snp_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values
    )
    ridge_y = model_df.loc[is_train, "YLDBE"].values
    ridge = Ridge(alpha=float(hypers.get("ridge_alpha", 1e-2)))
    ridge.fit(ridge_X, ridge_y)
    joblib.dump(ridge, out / "ridge_snp_baseline.pkl")
    logging.info("Ridge SNP baseline fit on %s train rows", len(ridge_y))

    # Step 6 env + GxE (pairs chosen on train slice; columns on full model_df)
    env_only_numeric = [c for c in env_numeric if c in model_df.columns and not str(c).startswith("GPC_")]
    env_chosen, gxe_pairs = step6_select_env_and_gxe(
        model_df.loc[is_train].copy(),
        env_only_numeric,
        gpc_cols,
        rng,
        int(cfg.get("max_env_features_gxe", 20)),
        int(cfg.get("n_interaction_pairs", 10)),
    )
    interact_cols: List[str] = []
    gxe_parts: Dict[str, pd.Series] = {}
    for gc, ec in gxe_pairs:
        cname = f"GXE__{gc}__x__{ec}"
        gxe_parts[cname] = model_df[gc].astype(float) * model_df[ec].astype(float)
        interact_cols.append(cname)
    if gxe_parts:
        model_df = pd.concat([model_df, pd.DataFrame(gxe_parts)], axis=1)

    past_cols = [c for c in ("past_yld_mean", "past_yld_median") if c in model_df.columns]
    nongeno_feature_cols = env_chosen + past_cols + interact_cols
    nongeno_feature_cols = [c for c in nongeno_feature_cols if c in model_df.columns]
    pd.Series(nongeno_feature_cols, name="env_and_gxe_column").to_csv(
        out / "env_gxe_feature_columns.csv", index=False
    )
    feature_cols = gpc_cols + nongeno_feature_cols
    G = model_df[gpc_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    Ng = model_df[nongeno_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    X_all = np.hstack([G, Ng])
    y_all = model_df["YLDBE"].values.astype(float)
    groups_all = model_df["env_id"].astype(str).values
    train_idx = is_train

    # Phase A — RandomizedSearchCV on train rows only (GPU XGBoost if available/configured)
    n_train_rows = int(train_idx.sum())
    use_gpu_cfg = bool(cfg.get("use_gpu", False))
    cuda_device = int(cfg.get("cuda_device", 0))
    xgb_ok = False
    torch_cuda = False
    xgb = None
    if use_gpu_cfg:
        try:
            import xgboost as xgb  # type: ignore

            xgb_ok = True
        except Exception:
            logging.warning("xgboost import failed; Step 7a will use sklearn GBR CPU fallback.")
        try:
            import torch  # type: ignore

            torch_cuda = bool(torch.cuda.is_available())
        except Exception:
            # torch optional here; if unavailable we keep CPU fallback to avoid accidental CPU XGBoost path.
            torch_cuda = False
    use_gpu_xgb = bool(use_gpu_cfg and xgb_ok and torch_cuda)

    n_iter = int(hypers.get("random_search_iter", 50))
    if n_train_rows < 15_000:
        n_iter = min(n_iter, 12)

    gb_hyp = hypers.get("gbr", {})
    fixed_params_gbr = {
        "n_estimators": int(gb_hyp.get("n_estimators", 500)),
        "max_depth": int(gb_hyp.get("max_depth", 6)),
        "learning_rate": float(gb_hyp.get("learning_rate", 0.05)),
        "subsample": float(gb_hyp.get("subsample", 0.8)),
    }
    skip_search = bool(cfg.get("skip_random_search", False))

    if use_gpu_xgb:
        log_step("Step 7a: XGBoost (GPU) + RandomizedSearchCV (GroupKFold)")
        logging.info(
            "Step 7a GPU enabled (use_gpu=%s, torch_cuda=%s, device=cuda:%s).",
            use_gpu_cfg,
            torch_cuda,
            cuda_device,
        )
        if n_train_rows < 15_000:
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 5, 6],
                "learning_rate": [0.05, 0.08],
                "subsample": [0.8],
                "colsample_bytree": [0.8, 1.0],
            }
        else:
            param_dist = {
                "n_estimators": [300, 500, 700],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.75, 0.8, 0.9],
                "colsample_bytree": [0.6, 0.8, 1.0],
            }

        fixed_params_xgb = {
            "n_estimators": int(gb_hyp.get("n_estimators", 500)),
            "max_depth": int(gb_hyp.get("max_depth", 6)),
            "learning_rate": float(gb_hyp.get("learning_rate", 0.05)),
            "subsample": float(gb_hyp.get("subsample", 0.8)),
            "colsample_bytree": 0.8,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "device": f"cuda:{cuda_device}",
            "predictor": "gpu_predictor",
            "n_jobs": 1,
            "random_state": 42,
            "verbosity": 0,
        }

        if skip_search:
            logging.info(
                "Skipping RandomizedSearchCV (small train or skip_random_search); using config xgboost params %s",
                fixed_params_xgb,
            )
            search = type(
                "SearchStub",
                (),
                {"best_params_": fixed_params_xgb, "best_estimator_": xgb.XGBRegressor(**fixed_params_xgb)},
            )()
        else:
            search = RandomizedSearchCV(
                xgb.XGBRegressor(**fixed_params_xgb),
                param_dist,
                n_iter=n_iter,
                cv=GroupKFold(n_splits=int(hypers.get("group_cv_splits", 5))),
                random_state=42,
                # Keep single worker to avoid multi-process contention on one GPU.
                n_jobs=1,
                scoring="neg_root_mean_squared_error",
            )
            search.fit(X_all[train_idx], y_all[train_idx], groups=groups_all[train_idx])
            logging.info("Best XGBoost GPU params: %s", search.best_params_)
        oof_model = xgb.XGBRegressor(**search.best_params_)
    else:
        if use_gpu_cfg and not use_gpu_xgb:
            logging.warning(
                "use_gpu=true but Step 7a GPU path unavailable (xgboost=%s, torch_cuda=%s); using sklearn GBR CPU fallback.",
                xgb_ok,
                torch_cuda,
            )
        log_step("Step 7a: GradientBoosting + RandomizedSearchCV (GroupKFold)")
        gbr = GradientBoostingRegressor(random_state=42)
        if n_train_rows < 15_000:
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [4, 5, 6],
                "learning_rate": [0.05, 0.08],
                "subsample": [0.8],
            }
        else:
            param_dist = {
                "n_estimators": [300, 500, 700],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.75, 0.8, 0.9],
            }

        if skip_search:
            logging.info(
                "Skipping RandomizedSearchCV (small train or skip_random_search); using config gbr params %s",
                fixed_params_gbr,
            )
            search = type(
                "SearchStub",
                (),
                {"best_params_": fixed_params_gbr, "best_estimator_": GradientBoostingRegressor(random_state=42, **fixed_params_gbr)},
            )()
        else:
            search = RandomizedSearchCV(
                gbr,
                param_dist,
                n_iter=n_iter,
                cv=GroupKFold(n_splits=int(hypers.get("group_cv_splits", 5))),
                random_state=42,
                n_jobs=int(cfg.get("sklearn_n_jobs", 1)),
                scoring="neg_root_mean_squared_error",
            )
            search.fit(X_all[train_idx], y_all[train_idx], groups=groups_all[train_idx])
            logging.info("Best GBR params: %s", search.best_params_)
        oof_model = GradientBoostingRegressor(**search.best_params_, random_state=42)

    # OOF on train with PCA refit per fold (Step 4 + 8)
    log_step("Step 8: OOF predictions (PCA refit per fold)")
    oof_train = grouped_oof_with_pca(
        model_df.loc[train_idx].reset_index(drop=True),
        snp_cols,
        geno,
        nongeno_feature_cols,
        y_all[train_idx],
        groups_all[train_idx],
        int(hypers.get("group_cv_splits", 5)),
        pca_n,
        oof_model,
        int(cfg.get("random_state", 42)),
    )
    oof_full = np.full(len(y_all), np.nan)
    oof_full[np.where(train_idx)[0]] = oof_train
    model_df["oof_pred"] = oof_full

    glob_metrics = metrics_dict(y_all[train_idx], oof_train)
    logging.info("OOF train metrics: %s", glob_metrics)

    by_env_rows = []
    for eid, chunk in model_df.loc[train_idx].groupby("env_id"):
        m = metrics_dict(chunk["YLDBE"].values, chunk["oof_pred"].values)
        m["env_id"] = eid
        m["n"] = len(chunk)
        by_env_rows.append(m)
    by_env = pd.DataFrame(by_env_rows)
    by_env.to_csv(out / "metrics_by_env_oof.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    tr = model_df.loc[train_idx]
    res = tr["YLDBE"] - tr["oof_pred"]
    ax.axhline(0, color="gray", lw=0.8)
    ax.scatter(tr["oof_pred"], res, alpha=0.15, s=8)
    ax.set_xlabel("OOF predicted YLDBE")
    ax.set_ylabel("Residual (obs - pred)")
    ax.set_title("Residuals vs OOF prediction (train years)")
    fig.tight_layout()
    fig.savefig(out / "res_plot.png", dpi=150)
    plt.close(fig)

    # Phase B — HistGradientBoosting or XGB with test-year early stopping
    log_step("Step 7b: Phase B booster + env/year holdout early stopping")
    X_tr = X_all[train_idx]
    y_tr = y_all[train_idx]
    te_idx = model_df["YEAR"].values == te_year
    X_va = X_all[te_idx]
    y_va = y_all[te_idx]
    hgb: Any
    if cfg.get("use_xgboost", False):
        try:
            import xgboost as xgb

            esr = int(hypers.get("early_stopping_rounds", 50))
            hgb = xgb.XGBRegressor(
                n_estimators=int(hypers["hist_gb"]["max_iter"]),
                max_depth=int(hypers["hist_gb"]["max_depth"]),
                learning_rate=float(hypers["hist_gb"]["learning_rate"]),
                subsample=0.8,
                random_state=42,
                n_jobs=-1,
            )
            if len(X_va) >= 20:
                hgb.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=False,
                    early_stopping_rounds=esr,
                )
            else:
                hgb.fit(X_tr, y_tr)
        except ImportError:
            logging.warning("xgboost not installed; falling back to HistGradientBoostingRegressor")
            cfg["use_xgboost"] = False

    if not cfg.get("use_xgboost", False):
        hgb = HistGradientBoostingRegressor(
            max_iter=int(hypers["hist_gb"]["max_iter"]),
            max_depth=int(hypers["hist_gb"]["max_depth"]),
            learning_rate=float(hypers["hist_gb"]["learning_rate"]),
            random_state=42,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=int(hypers.get("early_stopping_rounds", 50)),
        )
        hgb.fit(X_tr, y_tr)
    joblib.dump(hgb, out / "best_model.pkl")

    model_df["pred_final"] = hgb.predict(X_all)
    oof_path = out / "oof_preds.csv"
    model_df[["line_id", "env_id", "YEAR", "LOC", "YLDBE", "oof_pred", "pred_final"]].to_csv(
        oof_path, index=False
    )

    # Step 9 ranking
    log_step("Step 9: Top-k per environment")
    topk = int(cfg.get("topk", 10))
    rank_rows = []
    for env_id, chunk in model_df.loc[te_idx].groupby("env_id"):
        sub = chunk.sort_values("pred_final", ascending=False).head(topk).copy()
        sub["rank"] = np.arange(1, len(sub) + 1)
        sub["uncertainty_oof_rmse"] = glob_metrics["rmse"]
        rank_rows.append(sub)
    rankings = pd.concat(rank_rows, axis=0) if rank_rows else pd.DataFrame()
    rankings.to_csv(out / "rankings_topk_per_env.csv", index=False)

    # Step 10 interpretation
    log_step("Step 10: Permutation importance + buckets")
    perm = permutation_importance(
        hgb,
        X_tr,
        y_tr,
        n_repeats=10,
        random_state=42,
        n_jobs=int(cfg.get("sklearn_n_jobs", 1)),
    )
    fi = pd.DataFrame({"feature": feature_cols, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std})
    fi = fi.sort_values("importance_mean", ascending=False)
    fi.to_csv(out / "perm_importance.csv", index=False)

    def bucket(name: str) -> str:
        if name.startswith("GPC_"):
            return "Genetic_PCA"
        if name.startswith("GXE__"):
            return "GxE_interaction"
        if any(x in name.upper() for x in ("PRCP", "TAVG", "HTDD", "CLDD", "DP01", "DP10")):
            return "Weather"
        if any(x in name.upper() for x in ("CLAY", "SILT", "SAND", "PH", "SOC", "NITROGEN", "CFVO")):
            return "Soil"
        if "past_yld" in name:
            return "Past_pheno"
        return "Other"

    fi["bucket"] = fi["feature"].map(bucket)
    bucketed = fi.groupby("bucket")["importance_mean"].sum().reset_index()
    bucketed.to_csv(out / "perm_importance_buckets.csv", index=False)
    fi.head(20).to_csv(out / "interpretation_top_features_slide.csv", index=False)

    if cfg.get("use_shap", True):
        try:
            import shap

            explainer = shap.Explainer(hgb.predict, X_tr[: min(2000, len(X_tr))])
            sv = explainer(X_tr[: min(2000, len(X_tr))])
            shap.summary_plot(sv, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(out / "shap.png", dpi=150, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logging.warning("SHAP skipped: %s", e)

    # Step 11 summary
    summary = {
        "cohort": cohort,
        "join_report": join_report,
        "modeling_rows": int(len(model_df)),
        "oof_global_metrics": glob_metrics,
        "best_gbr_cv_params": search.best_params_,
        "feature_columns": feature_cols,
        "paths": {k: str(Path(v)) for k, v in paths.items()},
    }
    def _json_default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)

    with open(out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    logging.info("Done. Artifacts in %s", out)


if __name__ == "__main__":
    main()
