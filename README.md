# American Option Pricing with gSS and fTT

Applying two function-approximation methods to American option pricing:

- generalized Stochastic Sampling (gSS)
- functional Tensor Train (fTT), fitted by alternating least squares

Inputs are asset price, maturity, rate, dividend and implied volatility, plus
the European option price; the target is the American option price.

## Results

gSS on a 30,137-row held-out test set, measured as relative L2 error
(`norm(pred - actual) / norm(actual)`):

| approach | relative L2 |
|---|---|
| predict `european_op` directly | 0.0656 |
| linear regression, 6 features | 0.0447 |
| gSS, original configuration | 0.0112 |
| **gSS, tuned** | **0.0032** |

Read the first row before the last one. `corr(european_op, american_op) = 0.994`,
so the European price alone is already a strong predictor and the model is
largely learning the early-exercise premium on top of it. The headline number is
real but should always be quoted against that baseline.

Tuned configuration, selected by a `sim_range` × `nnodes` grid search:

| parameter | value |
|---|---|
| `sim_range` | 4 |
| `nnodes` (fit model) | 1000 |
| dense test model nodes | 5000 (`max_nodes_mult=5`) |
| kernel scale | 2.00, re-calibrated on the dense model |
| `l2_regularizer` (Ridge alpha) | 1e-6 |
| dtype | float64 |

## Running

Open `gss_ftt.ipynb`. It imports the `nnu/` package from this directory and
reads `data_ML.csv` from the repository root.

**float64 is required**, set globally in the imports cell via
`keras.config.set_floatx("float64")`. The outer regression solves normal
equations whose Gram matrix is numerically singular — smallest eigenvalue around
-1.4e-9 — and float32 is not sufficient to make that solve stable.

### Memory

At `nnodes=1000` the kernel layer materialises `[rows, nnodes, ndim]` float64
grids, so a full-training-set pass allocates **~3.1 GiB per buffer**. Budget
around 16 GB of free RAM. Three places control this and should not be changed
casually:

- **Cells 36 and 38** pass `batch_size=nsamples` deliberately. The `xpts` layer
  side-loads the whole training set and ignores its input, so a smaller batch
  recomputes the full output per chunk and concatenates — wrong results, and
  more memory rather than less.
- **Cell 31** computes gradients in chunks. A `GradientTape` retains every
  intermediate for the backward pass, so the single-shot version exhausts memory
  at this node count. Accumulating squared gradients per chunk is exact, since
  `norm(g, axis=0)` is a per-column L2 over samples.
- **`max_nodes_mult`** caps the dense test model's node count, and
  `regr_batch_size` blocks its kernel evaluation. Both trade time for peak
  allocation and neither changes results.

## Data

`data_ML.csv` is NOT included in this repository. Place a copy in the repository
root, next to the notebook, before running. Columns: asset_price, maturity,
rate, div, ivol, european_op, american_op.

## Method credit

`nnu/` and `tf_lbfgs/` are adapted from
https://github.com/piterbarg/altnnpub (MIT License,
(c) 2021 Alexandre Antonov, Vladimir V. Piterbarg). See LICENSE-altnnpub.txt.
The notebook applies that reference implementation to a new dataset.

The upstream source is **modified**, not vendored as-is:

- migrated from Keras 2 to Keras 3
- `gss_model_factory.py` — `tf.linalg.eye` inherits the operand dtype, for the
  float64 solve
- `gss_model_factory.py` — `Ridge` honours `l2_regularizer` instead of a
  hardcoded `1e-8` that ignored the caller
- `gss_model_factory.py` — added `max_nodes_mult` and `regr_batch_size` to bound
  the dense test model's memory
- `gss_model_factory.py` — dropped redundant `input_dim` from layer constructors
- `gss_layer.py` — removed a dead full-size `tf.tile` in `ProdKernelLayer.call`
