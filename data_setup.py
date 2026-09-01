# --- Libraries ---
import pandas as pd
from sklearn.preprocessing import StandardScaler

__all__ = ["data", "train", "test", "scaler", "ndim", "trainX", "trainY", 
           "testX", "testY", "nsamples", "describe"]

# --- Reading and viewing the data ---
data = pd.read_csv("data_ML.csv")  # Read the data from the CSV file

# "We can observe that the dataset includes `100455 observations` and `7 parameters`. " \
# "In our case, we are going to use the first 6 parameters to predict the value of the " \
# "7th parameter (`american_op`).")

# Splitting the dataset into training and testing sets
train = data.sample(frac = 0.70, random_state = 1)
test = data.drop(train.index)

# Learn the scaling from the TRAINING data only
scaler = StandardScaler()
scaler.fit(train.to_numpy()[:, :-1])

# Number of dimensions / features 
# (asset_price, maturity, rate, div, ivol, european_op)
ndim = 6 

# Scaled inputs and outputs/targets for training and testing sets
trainX = scaler.transform(train.to_numpy()[:, :-1])
trainY = train.to_numpy()[:,-1]
testX = scaler.transform(test.to_numpy()[:, :-1])
testY = test.to_numpy()[:,-1]

# Number of samples / number of rows (of the training set)
nsamples = train.shape[0] 

def describe():
    """Dataset overview."""
    data.info()
    print(f"\ntrain: {train.shape[0]} rows, test: {test.shape[0]} rows")
    return data.head(5)
