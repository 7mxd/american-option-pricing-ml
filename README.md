# American Option Pricing with gSS and fTT

Applying two function-approximation methods to American option pricing:

- generalized Stochastic Sampling (gSS)
- functional Tensor Train (fTT), fitted by alternating least squares

Inputs are asset price, maturity, rate, dividend and implied volatility, plus
the European option price; the target is the American option price.

## Running

Open `gss_ftt.ipynb`. It imports the `nnu/` package from this directory and
reads `data_ML.csv` from the repository root.

## Data

`data_ML.csv` is NOT included in this repository. Place a copy in the repository
root, next to the notebook, before running. Columns: asset_price, maturity,
rate, div, ivol, european_op, american_op.

## Method credit

`nnu/` and `tf_lbfgs/` are unmodified copies from
https://github.com/piterbarg/altnnpub (MIT License,
(c) 2021 Alexandre Antonov, Vladimir V. Piterbarg). See LICENSE-altnnpub.txt.
The notebook applies that reference implementation to a new dataset.
