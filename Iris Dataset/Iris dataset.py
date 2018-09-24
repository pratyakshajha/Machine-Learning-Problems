from sklearn import datasets
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from sklearn.metrics import precision_score
from sklearn.externals import joblib

# Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split into train - test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

# Set parameters
param = {
    'max_depth': 3,  # max depth of a tree
    'eta': 0.3,      # training step for an iteration
    'silent': 1,     # logging mode
    'objective': 'multi:softprob',  # error evaluation for multiclass
    'num_class':3    # number of classes
}
num_round = 30       # number of iterations

# Train
mdl = xgb.train(param, dtrain, num_round)

# Evaluate
preds = mdl.predict(dtest)
print(preds)

best_preds = np.asarray([np.argmax(val) for val in preds])

print(precision_score(y_test, best_preds, average = 'macro'))

# Save model
joblib.dump(mdl, 'iris_model.pkl', compress = True)

# Load saved model
mdl_loaded = joblib.load('iris_model.pkl')

# Check whether loaded model gives same result as actual one
print((mdl_loaded.predict(dtest) == preds).all())

