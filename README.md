# American Option Pricing with gSS and fTT

Applying the function-approximation methods from
[altnnpub](https://github.com/piterbarg/altnnpub) to American option pricing on
real market data.

**Inputs:** asset price, maturity, rate, dividend, implied volatility, and the
European option price.
**Target:** the American option price.

## Status

The goal is to apply all four methods from the reference repository. One is done.

| method | status |
|---|---|
| generalized Stochastic Sampling (gSS), one-dimensional | ✅ complete |
| generalized Stochastic Sampling (gSS), multi-dimensional | ⬜ not started |
| functional Tensor Train (fTT), via alternating least squares | ⬜ in progress |
| ReLU deep neural network | ⬜ not started |

## Results so far

gSS (one-dimensional), on a 30,137-row held-out test set:

| metric | value |
|---|---|
| relative L2 — `norm(pred - actual) / norm(actual)` | **0.0032** |

Configuration, selected by a `sim_range` × `nnodes` grid search:

| parameter | value |
|---|---|
| `sim_range` | 4 |
| `nnodes` (fit model) | 1000 |
| dense test model nodes | 5000 (`max_nodes_mult=5`) |
| kernel scale | 2.00, re-calibrated on the dense model |
| `l2_regularizer` (Ridge alpha) | 1e-6 |
| dtype | float64 |

### Read this number with care

**No baseline is computed in the notebook yet**, so 0.0032 cannot be interpreted
on its own. The European option price is itself a strong predictor of the
American price — the two are very highly correlated — so a large part of that
accuracy may come from the input rather than from the method.

A baseline comparison is still to be decided and added.

## Running

Open `gss_ftt.ipynb`. It imports the `nnu/` package from this directory and reads
`data_ML.csv` from the repository root.

**float64 is required.** It is set globally in the imports cell with
`keras.config.set_floatx("float64")`. The outer regression solves normal
equations whose Gram matrix is numerically singular — smallest eigenvalue around
`-1.4e-9` — and float32 is not precise enough to make that solve stable.

### Memory

At `nnodes = 1000` the kernel layer materialises `[rows, nnodes, ndim]` float64
grids, so a full-training-set pass allocates about **3.1 GiB per buffer**. Budget
roughly 16 GB of free RAM.

Three places control this. None should be changed casually:

| where | what | why |
|---|---|---|
| cells 36, 38 | `batch_size=nsamples` | The `xpts` layer side-loads the whole training set and ignores its input, so a *smaller* batch recomputes the full output per chunk and concatenates — wrong results, and more memory rather than less. |
| cell 31 | chunked gradient | A `GradientTape` retains every intermediate for the backward pass, so the single-shot version exhausts memory at this node count. Accumulating squared gradients per chunk is exact, since `norm(g, axis=0)` is a per-column L2 over samples. |
| `max_nodes_mult`, `regr_batch_size` | cap and block the dense test model | Both trade time for peak allocation. Neither changes results. |

## Data

`data_ML.csv` is **not** included in this repository. Place a copy in the
repository root, next to the notebook, before running.

Columns: `asset_price`, `maturity`, `rate`, `div`, `ivol`, `european_op`,
`american_op`.

## Method credit

`nnu/` and `tf_lbfgs/` are adapted from
[altnnpub](https://github.com/piterbarg/altnnpub) (MIT License, © 2021 Alexandre
Antonov, Vladimir V. Piterbarg). See `LICENSE-altnnpub.txt`. This repository
applies that reference implementation to a new dataset.

The upstream source is **modified**, not vendored as-is:

| file | change |
|---|---|
| both | migrated from Keras 2 to Keras 3 |
| `gss_model_factory.py` | `tf.linalg.eye` inherits the operand dtype, for the float64 solve |
| `gss_model_factory.py` | `Ridge` honours `l2_regularizer` instead of a hardcoded `1e-8` that ignored the caller |
| `gss_model_factory.py` | added `max_nodes_mult` and `regr_batch_size` to bound the dense test model's memory |
| `gss_model_factory.py` | dropped redundant `input_dim` from layer constructors |
| `gss_layer.py` | removed a dead full-size `tf.tile` in `ProdKernelLayer.call` |
