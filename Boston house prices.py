import time
from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Load data
boston = datasets.load_boston()
dat = pd.DataFrame(boston.data, columns=boston.feature_names)
dat.head()

target = pd.DataFrame(boston.target, columns=["MEDV"])
target.head()

df = dat.copy()
df = pd.concat([df, target], axis=1)
df.head()

# Analyse data
print(df.info())
print(df.describe())

plt.style.use('seaborn') # switch to seaborn style
df.hist(bins = 10, figsize = (10,8));
plt.show();

corr_matrix = df.corr()

sns.heatmap(corr_matrix);
plt.show()
print(corr_matrix['MEDV'])

print(boston['DESCR'])

dat1 = df.drop(['CHAS','DIS'], axis=1)

# Split and transform data
X_train, X_test, y_train, y_test = train_test_split(dat1, target, test_size = 0.2, random_state=42)
y_train = y_train.values.ravel()

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

# Train data - Cross validate
models = []
models.append(('SVR', SVR()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('RF', RandomForestRegressor()))
models.append(('ET', ExtraTreesRegressor()))

scoring = 'neg_mean_squared_error'

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# Tune algorithm
pipeline = make_pipeline(ExtraTreesRegressor(random_state=42))

hyperparameters = { 'extratreesregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'extratreesregressor__max_depth': [5, 3],
                  'extratreesregressor__n_estimators': [100, 150, 200, 250]}

start = time.time()
print('\nRunning GridSearch')
clf = GridSearchCV(pipeline, hyperparameters, cv=10, scoring = scoring)
clf.fit(X_train, y_train)
end = time.time()
print('Time taken by GridSearch {}'.format(end - start))

start = time.time()
print('\nRunning RandomizedSearch')
clf1 = RandomizedSearchCV(pipeline, hyperparameters, cv=10, random_state=42)
clf1.fit(X_train, y_train)
end = time.time()
print('Time taken by RandomizedSearch {}'.format(end - start))

# Evaluate
pred = clf.predict(X_test)
print("MSE for GridSearchCV: {}". format(mean_squared_error(y_test, pred)))

pred1 = clf1.predict(X_test)
print("MSE for RandomizedSearchCV: {}". format(mean_squared_error(y_test, pred1)))

# Save model 
joblib.dump(clf1, 'boston_regressor.pkl')

# Load saved model
clf2 = joblib.load('boston_regressor.pkl')

