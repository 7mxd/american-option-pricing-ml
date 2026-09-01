# American Option Pricing with Alternatives to Deep Neural Networks

Applying the function-approximation methods from
[altnnpub](https://github.com/piterbarg/altnnpub) — the reference implementation
for Antonov & Piterbarg's work on alternatives to deep neural networks — to
American option pricing on real market data.

**Inputs:** asset price, maturity, rate, dividend, implied volatility, and the
European option price.
**Target:** the American option price.

> **This repository cannot be run as-is.** `data_ML.csv` is proprietary and is not
> included. See [Data](#data).

## Status

Work in progress. The goal is to apply all four methods from the reference
implementation and compare them on identical data.

| method | status |
|---|---|
| gSS, one-dimensional | ✅ complete — `onedim_gss.ipynb` |
| fTT, via alternating least squares | 🚧 in progress |
| gSS, multi-dimensional | ⬜ not started |
| ReLU deep neural network | ⬜ not started |

Each method is added to the repository once it is finished, so only the
completed one appears here.

A cross-method comparison — including naive and linear baselines — comes once
more than one method is finished.

## Results so far

**No baseline has been computed yet, so the number below cannot be interpreted on
its own.** The European option price is itself a strong predictor of the American
price; the two are very highly correlated. An unknown share of the accuracy comes
from that input rather than from the method. Baselines are deferred to the
cross-method comparison, and until then this is an internal checkpoint, not a
claim about how well gSS works.

With that said — gSS (one-dimensional), on a 30,137-row held-out test set:

| metric | value |
|---|---|
| relative L2 — `norm(pred - actual) / norm(actual)` | 0.0032 |
| relative MAE — `mean(abs(pred - actual)) / rms(actual)` | 0.0017 |

Configuration, selected by a `sim_range` × `nnodes` grid search:

| parameter | value |
|---|---|
| `sim_range` | 4 |
| `nnodes` (fit model) | 1000 |
| dense test model nodes | 5000 (`max_nodes_mult=5`) |
| kernel scale | 2.00, re-calibrated on the dense model |
| `l2_regularizer` (Ridge alpha) | 1e-6 |
| dtype | float64 |

## Layout

| file | purpose |
|---|---|
| `data_setup.py` | loads the data and builds the train/test split and scaler. Every notebook imports it, so all methods are evaluated on identical data. |
| `onedim_gss.ipynb` | gSS, one-dimensional |
| `nnu/`, `tf_lbfgs/` | the altnnpub source, modified — see [Method credit](#method-credit) |

## Running

Open a method notebook and run it. Each imports `data_setup` and is independent
of the others.

**float64 is required.** Each notebook sets it with
`keras.config.set_floatx("float64")` before building any layer. The outer
regression solves normal equations whose Gram matrix is numerically singular —
smallest eigenvalue around `-1.4e-9` — and float32 is not precise enough to make
that solve stable.

### Memory

At `nnodes = 1000` the kernel layer materialises `[rows, nnodes, ndim]` float64
grids, so a full-training-set pass allocates about 3.1 GiB per buffer, and peak
process memory reaches roughly 14 GB. Budget accordingly.

Three places in `onedim_gss.ipynb` control this. None should be changed casually:

| where | what | why |
|---|---|---|
| cells 16, 18 | `batch_size=nsamples` | The `xpts` layer side-loads the whole training set and ignores its input, so a *smaller* batch recomputes the full output per chunk and concatenates — wrong results, and more memory rather than less. |
| cell 11 | chunked gradient | A `GradientTape` retains every intermediate for the backward pass, so the single-shot version exhausts memory at this node count. Accumulating squared gradients per chunk is exact, since `norm(g, axis=0)` is a per-column L2 over samples. |
| cells 23, 26 | `max_nodes_mult`, `regr_batch_size` | Cap and block the dense test model. Both trade time for peak allocation; neither changes results. |

Memory is not returned to the operating system when a cell finishes — the
allocator pools it — so restarting the kernel is the only way to reclaim it. This
is expected, not a leak: the footprint reaches a high-water mark and stays flat.

## Data

`data_ML.csv` is **not** included in this repository and is not publicly
available. Without it, nothing here can be reproduced.

Columns: `asset_price`, `maturity`, `rate`, `div`, `ivol`, `european_op`,
`american_op`. Place a copy in the repository root, next to the notebooks.

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
