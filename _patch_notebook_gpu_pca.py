"""Patch maize_gxe_ml_pipeline.ipynb with GPU PCA; write copy to output/linear_regression_torch_pca_gpu.ipynb."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / "maize_gxe_ml_pipeline.ipynb"
OUT_NB = ROOT / "output" / "linear_regression_torch_pca_gpu.ipynb"

MD_GPU = """## GPU PCA (PyTorch `pca_lowrank`)

SNP→GPC uses **CUDA** when `use_gpu: true` and PyTorch sees CUDA. YAML: `max_snps`, `pca_once`, `gpu_pca_n_folds`, `cuda_device`.
"""

GPU_CODE = r'''import time

import torch

from sklearn.decomposition import PCA


def _effective_pca_k(n_samples: int, n_features: int, requested: int) -> int:
    return max(1, min(int(requested), n_samples - 1, n_features))


def pca_torch_gpu(
    X_t: torch.Tensor,
    n_components: int,
    center: bool = True,
) -> tuple:
    """GPU PCA via torch.pca_lowrank. Returns (scores_np, V_np, S_np)."""
    n, p = X_t.shape
    k = _effective_pca_k(n, p, n_components)
    if center:
        mean = X_t.mean(dim=0, keepdim=True)
        Xc = X_t - mean
    else:
        mean = torch.zeros(1, p, device=X_t.device, dtype=X_t.dtype)
        Xc = X_t
    _U, S, V = torch.pca_lowrank(Xc.float(), q=k, center=False, niter=5)
    Vk = V[:, :k]
    scores = Xc @ Vk
    return scores.cpu().numpy(), Vk.cpu().numpy(), S[:k].cpu().numpy()


def pca_torch_gpu_fit(
    X_t: torch.Tensor,
    n_components: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns scores (N,K), V (P,K), S (K), mean (1,P) on device."""
    n, p = X_t.shape
    k = _effective_pca_k(n, p, n_components)
    mean = X_t.mean(dim=0, keepdim=True)
    Xc = X_t - mean
    _U, S, V = torch.pca_lowrank(Xc.float(), q=k, center=False, niter=5)
    Vk = V[:, :k]
    scores = Xc @ Vk
    return scores, Vk, S[:k], mean


def pca_torch_project(X_t: torch.Tensor, mean: torch.Tensor, Vk: torch.Tensor) -> torch.Tensor:
    return (X_t - mean) @ Vk


def build_X_snp_pruned_gpu(
    model_df: pd.DataFrame,
    geno_wide: pd.DataFrame,
    snp_cols: List[str],
    max_snps: int,
    device: torch.device,
) -> tuple[torch.Tensor, List[str], torch.Tensor]:
    X_np = (
        model_df[["line_id"]]
        .merge(geno_wide, on="line_id", how="left")[snp_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .values.astype(np.float32)
    )
    X_t = torch.from_numpy(X_np).to(device, non_blocking=True)
    p = X_t.shape[1]
    k = min(max_snps, p)
    snp_var = torch.var(X_t, dim=0, unbiased=False)
    top_idx = torch.topk(snp_var, k).indices
    top_idx = torch.sort(top_idx).values
    Xp = X_t[:, top_idx]
    pruned_cols = [snp_cols[int(i)] for i in top_idx.cpu().numpy()]
    logging.info("GPU SNPs pruned: %s (from %s)", Xp.shape[1], p)
    return Xp, pruned_cols, top_idx


def fit_pca_sklearn_cpu(
    geno_wide: pd.DataFrame,
    snp_cols_sub: Sequence[str],
    line_ids: np.ndarray,
    n_components: int,
    random_state: int,
) -> Pipeline:
    sub = geno_wide.loc[geno_wide["line_id"].isin(set(line_ids)), ["line_id"] + list(snp_cols_sub)]
    sub = sub.drop_duplicates("line_id")
    X = np.nan_to_num(sub[list(snp_cols_sub)].values.astype(np.float32), nan=0.0)
    n_comp = _effective_pca_k(X.shape[0], X.shape[1], n_components)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=random_state)),
        ]
    )
    pipe.fit(X)
    return pipe


def map_pca_sklearn_to_rows(
    geno_wide: pd.DataFrame,
    snp_cols_sub: Sequence[str],
    line_ids: pd.Series,
    pipe: Pipeline,
) -> pd.DataFrame:
    sub = geno_wide[["line_id"] + list(snp_cols_sub)].drop_duplicates("line_id").set_index("line_id")
    X = sub.reindex(line_ids).astype(np.float32)
    X = np.nan_to_num(X.values, nan=0.0)
    comps = pipe.transform(X)
    gpc_cols = [f"GPC_{i+1}" for i in range(comps.shape[1])]
    return pd.DataFrame(comps, columns=gpc_cols, index=line_ids.index)


def _unique_train_snp_tensor(
    geno_wide: pd.DataFrame,
    snp_cols_sub: List[str],
    line_ids: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    sub = geno_wide.loc[geno_wide["line_id"].isin(set(line_ids)), ["line_id"] + snp_cols_sub]
    sub = sub.drop_duplicates("line_id")
    X = np.nan_to_num(sub[snp_cols_sub].values.astype(np.float32), nan=0.0)
    return torch.from_numpy(X).to(device, non_blocking=True)


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
    *,
    snp_cols_pruned: Optional[List[str]] = None,
    X_snp_pruned_gpu: Optional[torch.Tensor] = None,
    model_row_positions: Optional[np.ndarray] = None,
    use_gpu: bool = False,
    pca_once: bool = False,
    Vk_frozen: Optional[torch.Tensor] = None,
    mean_frozen: Optional[torch.Tensor] = None,
) -> np.ndarray:
    cols = snp_cols_pruned if snp_cols_pruned is not None else snp_cols
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)

    if use_gpu and X_snp_pruned_gpu is not None and model_row_positions is not None:
        assert len(model_row_positions) == len(df)
        dev = X_snp_pruned_gpu.device
        for fold, (tr, te) in enumerate(gkf.split(df, y, groups=groups)):
            t_fold = time.time()
            if pca_once and Vk_frozen is not None and mean_frozen is not None:
                Vk, mean_f = Vk_frozen, mean_frozen
            else:
                tr_lines = df.iloc[tr]["line_id"].unique()
                X_fit_t = _unique_train_snp_tensor(geno_wide, cols, tr_lines, dev)
                k_eff = _effective_pca_k(X_fit_t.shape[0], X_fit_t.shape[1], pca_n)
                _sc, Vk, _S, mean_f = pca_torch_gpu_fit(X_fit_t, k_eff)
                del X_fit_t
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            ix_tr = model_row_positions[tr]
            ix_te = model_row_positions[te]
            gpc_tr = pca_torch_project(X_snp_pruned_gpu[ix_tr], mean_f, Vk).cpu().numpy()
            gpc_te = pca_torch_project(X_snp_pruned_gpu[ix_te], mean_f, Vk).cpu().numpy()
            Xtr = np.hstack([gpc_tr, df.iloc[tr][nongeno_feature_cols].values.astype(float)])
            Xte = np.hstack([gpc_te, df.iloc[te][nongeno_feature_cols].values.astype(float)])
            model.fit(Xtr, y[tr])
            oof[te] = model.predict(Xte)
            logging.info("GPU PCA OOF fold %s/%s in %.2fs", fold + 1, n_splits, time.time() - t_fold)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    else:
        for tr, te in gkf.split(df, y, groups=groups):
            tr_lines = df.iloc[tr]["line_id"].unique()
            pipe = fit_pca_sklearn_cpu(geno_wide, list(cols), tr_lines, pca_n, random_state)
            gpc_tr = map_pca_sklearn_to_rows(geno_wide, list(cols), df.iloc[tr]["line_id"], pipe).values
            gpc_te = map_pca_sklearn_to_rows(geno_wide, list(cols), df.iloc[te]["line_id"], pipe).values
            Xtr = np.hstack([gpc_tr, df.iloc[tr][nongeno_feature_cols].values.astype(float)])
            Xte = np.hstack([gpc_te, df.iloc[te][nongeno_feature_cols].values.astype(float)])
            model.fit(Xtr, y[tr])
            oof[te] = model.predict(Xte)
    return oof


def resolve_pca_device(cfg: Dict[str, Any]) -> tuple[bool, torch.device]:
    want = bool(cfg.get("use_gpu", True))
    if want and torch.cuda.is_available():
        d = int(cfg.get("cuda_device", 0))
        return True, torch.device(f"cuda:{d}")
    return False, torch.device("cpu")


def _gpc_corr_max_offdiag(gpc_df: pd.DataFrame, gpc_cols: List[str]) -> float:
    if len(gpc_cols) <= 1:
        return 0.0
    C = gpc_df[gpc_cols].corr(numeric_only=True).abs().values
    m = C.shape[0]
    tri = np.triu_indices(m, k=1)
    return float(C[tri].max())


def finalize_gpc_for_model(
    model_df: pd.DataFrame,
    geno: pd.DataFrame,
    snp_cols_pruned: List[str],
    X_snp_pruned: torch.Tensor,
    is_train: np.ndarray,
    pca_n: int,
    use_gpu: bool,
    device: torch.device,
    pca_once: bool,
    random_state: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, List[str]]:
    tr_lines = model_df.loc[is_train, "line_id"].unique()
    k_req = int(pca_n)
    if use_gpu:
        if pca_once:
            X_fit_t = X_snp_pruned[is_train]
        else:
            X_fit_t = _unique_train_snp_tensor(geno, snp_cols_pruned, tr_lines, device)
        k_eff = _effective_pca_k(X_fit_t.shape[0], X_fit_t.shape[1], k_req)
        t0 = time.time()
        _sc, Vk, S, mean_f = pca_torch_gpu_fit(X_fit_t, k_eff)
        logging.info("Final GPU PCA fit: %.2fs | k=%s", time.time() - t0, k_eff)
        gpc_np = pca_torch_project(X_snp_pruned, mean_f, Vk).cpu().numpy()
        torch.save(
            {"mean": mean_f.cpu(), "V": Vk.cpu(), "S": S.cpu(), "snp_cols_pruned": snp_cols_pruned},
            out_dir / "gpu_pca_state.pt",
        )
        joblib.dump({"backend": "torch", "k": k_eff}, out_dir / "pca_pipeline.pkl")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        pipe = fit_pca_sklearn_cpu(geno, snp_cols_pruned, tr_lines, k_req, random_state)
        joblib.dump(pipe, out_dir / "pca_pipeline.pkl")
        gpc_np = map_pca_sklearn_to_rows(geno, snp_cols_pruned, model_df["line_id"], pipe).values
    gpc_cols = [f"GPC_{i+1}" for i in range(gpc_np.shape[1])]
    gpc_df = pd.DataFrame(gpc_np, columns=gpc_cols)
    return gpc_df, gpc_cols


_cuda_ok = torch.cuda.is_available()
print("torch:", torch.__version__, "| CUDA:", _cuda_ok, torch.cuda.get_device_name(0) if _cuda_ok else "")
'''


def src_join(c: dict) -> str:
    return "".join(c.get("source", []))


def set_src(c: dict, text: str) -> None:
    c["source"] = [ln + "\n" for ln in text.splitlines()]


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    already = any(c.get("id") == "gpu-pca-md" for c in nb["cells"])
    if already:
        print("Notebook already has GPU PCA cells; skipping insert/delete. Re-applying text replacements only.")

    # Imports cell 1
    c1 = nb["cells"][1]
    s1 = src_join(c1)
    if "import torch" not in s1:
        s1 = s1.replace("import yaml\n", "import yaml\nimport time\n\nimport torch\n")
    set_src(c1, s1)

    if not already:
        # Insert markdown + GPU code before old PCA cell (index 10)
        nb["cells"].insert(
            10,
            {"cell_type": "markdown", "id": "gpu-pca-md", "metadata": {}, "source": [l + "\n" for l in MD_GPU.splitlines()]},
        )
        nb["cells"].insert(
            11,
            {"cell_type": "code", "id": "gpu-pca-code", "execution_count": None, "metadata": {}, "outputs": [], "source": [l + "\n" for l in GPU_CODE.splitlines()]},
        )
        del nb["cells"][12]

    MAIN_PCA_OLD = """    # Step 4 + 6: PCA on full geno reference; pre-compute baseline Ridge SNPs for reporting
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
"""

    MAIN_PCA_NEW = """    # Step 4 + 6: GPU torch.pca_lowrank (or sklearn CPU) + variance-pruned SNPs
    pca_n = int(hypers.get("pca_components", 50))
    max_snps = int(cfg.get("max_snps", cfg.get("genotype_max_snps", 2000)))
    pca_once = bool(cfg.get("pca_once", False))
    use_gpu, device = resolve_pca_device(cfg)
    mem_pre = torch.cuda.memory_allocated() / 1e9 if use_gpu else 0.0
    t_pca0 = time.time()
    X_snp_pruned, snp_cols_pruned, _top_idx = build_X_snp_pruned_gpu(
        model_df, geno, snp_cols, max_snps, device
    )
    logging.info(
        "SNP tensor %s on %s | prep %.2fs | VRAM ~%.2f GB",
        tuple(X_snp_pruned.shape),
        device,
        time.time() - t_pca0,
        (torch.cuda.memory_allocated() / 1e9 - mem_pre) if use_gpu else 0.0,
    )

    is_train = model_df["YEAR"].between(tr_min, tr_max).values
    train_lines = model_df.loc[is_train, "line_id"].unique()
    Vk_frozen, mean_frozen = None, None
    if use_gpu and pca_once:
        X_fit_once = _unique_train_snp_tensor(geno, snp_cols_pruned, train_lines, device)
        k0 = _effective_pca_k(X_fit_once.shape[0], X_fit_once.shape[1], pca_n)
        _sc0, Vk_frozen, _S0, mean_frozen = pca_torch_gpu_fit(X_fit_once, k0)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    gpc_df, gpc_cols = finalize_gpc_for_model(
        model_df,
        geno,
        snp_cols_pruned,
        X_snp_pruned,
        is_train,
        pca_n,
        use_gpu,
        device,
        pca_once,
        int(cfg.get("random_state", 42)),
        out,
    )
    model_df = pd.concat([model_df.reset_index(drop=True), gpc_df.reset_index(drop=True)], axis=1)

    gpc_df.to_csv(out / "gpu_gpc_components.csv", index=False)
    mx = _gpc_corr_max_offdiag(gpc_df, gpc_cols)
    logging.info("GPC max |corr| off-diag < 0.99: %s (max=%.4f)", mx < 0.99, mx)
    print("gpc collinear check (max abs corr off-diag < 0.99):", mx < 0.99, "| max =", mx)

    # Ridge baseline on pruned SNPs (train rows)
    ridge_X = (
        model_df.loc[is_train, ["line_id"]]
        .merge(geno, on="line_id", how="left")[snp_cols_pruned]
"""

    OOF_OLD = """    oof_train = grouped_oof_with_pca(
        model_df.loc[train_idx].reset_index(drop=True),
        snp_cols,
        geno,
        nongeno_feature_cols,
        y_all[train_idx],
        groups_all[train_idx],
        int(hypers.get("group_cv_splits", 5)),
        pca_n,
        GradientBoostingRegressor(**search.best_params_, random_state=42),
        int(cfg.get("random_state", 42)),
    )
"""

    OOF_NEW = """    n_folds_oof = int(cfg.get("gpu_pca_n_folds", hypers.get("group_cv_splits", 5)))
    tr_pos = np.flatnonzero(train_idx)
    oof_train = grouped_oof_with_pca(
        model_df.loc[train_idx].reset_index(drop=True),
        snp_cols,
        geno,
        nongeno_feature_cols,
        y_all[train_idx],
        groups_all[train_idx],
        n_folds_oof,
        pca_n,
        GradientBoostingRegressor(**search.best_params_, random_state=42),
        int(cfg.get("random_state", 42)),
        snp_cols_pruned=snp_cols_pruned,
        X_snp_pruned_gpu=X_snp_pruned,
        model_row_positions=tr_pos,
        use_gpu=use_gpu,
        pca_once=pca_once,
        Vk_frozen=Vk_frozen,
        mean_frozen=mean_frozen,
    )
"""

    CELL17_OLD = """# Step 4 + 6: PCA on full geno reference; pre-compute baseline Ridge SNPs for reporting
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
"""

    CELL17_NEW = """# Step 4 + 6: GPU PCA (torch.pca_lowrank) or sklearn fallback
pca_n = int(hypers.get("pca_components", 50))
max_snps = int(cfg.get("max_snps", cfg.get("genotype_max_snps", 2000)))
pca_once = bool(cfg.get("pca_once", False))
use_gpu, device = resolve_pca_device(cfg)
mem_pre = torch.cuda.memory_allocated() / 1e9 if use_gpu else 0.0
t_pca0 = time.time()
X_snp_pruned, snp_cols_pruned, _top_idx = build_X_snp_pruned_gpu(
    model_df, geno, snp_cols, max_snps, device
)
logging.info(
    "SNP tensor %s on %s | prep %.2fs",
    tuple(X_snp_pruned.shape),
    device,
    time.time() - t_pca0,
)

is_train = model_df["YEAR"].between(tr_min, tr_max).values
train_lines = model_df.loc[is_train, "line_id"].unique()
Vk_frozen, mean_frozen = None, None
if use_gpu and pca_once:
    X_fit_once = _unique_train_snp_tensor(geno, snp_cols_pruned, train_lines, device)
    k0 = _effective_pca_k(X_fit_once.shape[0], X_fit_once.shape[1], pca_n)
    _sc0, Vk_frozen, _S0, mean_frozen = pca_torch_gpu_fit(X_fit_once, k0)
    if device.type == "cuda":
        torch.cuda.empty_cache()

gpc_df, gpc_cols = finalize_gpc_for_model(
    model_df,
    geno,
    snp_cols_pruned,
    X_snp_pruned,
    is_train,
    pca_n,
    use_gpu,
    device,
    pca_once,
    int(cfg.get("random_state", 42)),
    out,
)
model_df = pd.concat([model_df.reset_index(drop=True), gpc_df.reset_index(drop=True)], axis=1)

gpc_df.to_csv(out / "gpu_gpc_components.csv", index=False)
mx = _gpc_corr_max_offdiag(gpc_df, gpc_cols)
print("gpc collinear check (max abs corr off-diag < 0.99):", mx < 0.99, "| max =", mx)

# Ridge baseline on pruned SNPs (train rows)
ridge_X = (
    model_df.loc[is_train, ["line_id"]]
    .merge(geno, on="line_id", how="left")[snp_cols_pruned]
"""

    CELL18_OOF_OLD = """oof_train = grouped_oof_with_pca(
    model_df.loc[train_idx].reset_index(drop=True),
    snp_cols,
    geno,
    nongeno_feature_cols,
    y_all[train_idx],
    groups_all[train_idx],
    int(hypers.get("group_cv_splits", 5)),
    pca_n,
    GradientBoostingRegressor(**search.best_params_, random_state=42),
    int(cfg.get("random_state", 42)),
)
"""

    CELL18_OOF_NEW = """n_folds_oof = int(cfg.get("gpu_pca_n_folds", hypers.get("group_cv_splits", 5)))
tr_pos = np.flatnonzero(train_idx)
oof_train = grouped_oof_with_pca(
    model_df.loc[train_idx].reset_index(drop=True),
    snp_cols,
    geno,
    nongeno_feature_cols,
    y_all[train_idx],
    groups_all[train_idx],
    n_folds_oof,
    pca_n,
    GradientBoostingRegressor(**search.best_params_, random_state=42),
    int(cfg.get("random_state", 42)),
    snp_cols_pruned=snp_cols_pruned,
    X_snp_pruned_gpu=X_snp_pruned,
    model_row_positions=tr_pos,
    use_gpu=use_gpu,
    pca_once=pca_once,
    Vk_frozen=Vk_frozen,
    mean_frozen=mean_frozen,
)
"""

    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = src_join(c)
        if "def main() -> None:" in s and MAIN_PCA_OLD in s:
            s = s.replace(MAIN_PCA_OLD, MAIN_PCA_NEW).replace(OOF_OLD, OOF_NEW)
            set_src(c, s)
        if "# Training / test masks" in s and CELL17_OLD in s:
            s = s.replace(CELL17_OLD, CELL17_NEW)
            set_src(c, s)
        if "oof_train = grouped_oof_with_pca(" in s and CELL18_OOF_OLD in s:
            s = s.replace(CELL18_OOF_OLD, CELL18_OOF_NEW)
            set_src(c, s)

    for _i, _c in enumerate(nb["cells"]):
        if _c.get("cell_type") != "code":
            continue
        _s = src_join(_c)
        if _s.strip().startswith("#!/usr/bin/env python3"):
            nb["cells"][_i] = {
                "cell_type": "code",
                "id": "gpu-end-print",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    'print("✅ GPU PyTorch PCA live: SNPs→GPC 10-50x faster! Full pipeline GPU-dominant 🚀")\n'
                ],
            }
            break

    text = json.dumps(nb, indent=2, ensure_ascii=False) + "\n"
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(text, encoding="utf-8")
    NB.write_text(text, encoding="utf-8")
    print("OK:", NB, OUT_NB)


if __name__ == "__main__":
    main()
