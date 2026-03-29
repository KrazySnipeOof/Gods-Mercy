# GPU Setup for Maize GxE Step 7a

This repo can run Step 7a hyperparameter search on GPU using XGBoost when:

- `use_gpu: true` in `pipeline_config.yaml`
- CUDA is available (`torch.cuda.is_available() == True`)
- `xgboost` imports successfully in the same interpreter running the pipeline

## 1) Verify interpreter used by pipeline

```powershell
python -c "import sys; print(sys.executable); print(sys.version)"
```

## 2) Verify XGBoost import in that interpreter

```powershell
python -c "import xgboost as xgb; print('ok', xgb.__version__)"
```

If this fails, install into that same interpreter:

```powershell
python -m pip install xgboost
```

## 3) Verify CUDA visibility

```powershell
python -c "import torch; print('torch_cuda', torch.cuda.is_available())"
```

Optional smoke test (tiny CUDA fit):

```powershell
python -c "import numpy as np, xgboost as xgb; X=np.random.rand(200,12).astype('float32'); y=np.random.rand(200).astype('float32'); m=xgb.XGBRegressor(tree_method='hist', device='cuda:0', objective='reg:squarederror', n_estimators=20, max_depth=4, learning_rate=0.1); m.fit(X,y, verbose=False); print('gpu_fit_ok')"
```

## 4) Run pipeline and confirm GPU Step 7a

Expected log line:

- `=== Step 7a: XGBoost (GPU) + RandomizedSearchCV (GroupKFold) ===`

During Step 7a, monitor GPU usage:

```powershell
nvidia-smi
```

## 5) Fallback behavior

If `xgboost` import fails or CUDA is unavailable, pipeline falls back to sklearn GBR CPU path.

Explicit CPU run:

- set `use_gpu: false` in `pipeline_config.yaml`

This keeps prior behavior while still allowing GPU acceleration when available.
