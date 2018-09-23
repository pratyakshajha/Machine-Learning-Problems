from sklearn import datasets
import pandas as pd
from matplotlib import pyplot as plt
import seaborn.apionly as snsapi
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import mean_squared_error

from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

boston = datasets.load_boston()
dat = pd.DataFrame(boston.data, columns=boston.feature_names)
dat.head()

target = pd.DataFrame(boston.target, columns=["MEDV"])
target.head()

df = dat.copy()
df = pd.concat([df, target], axis=1)
df.head()

df.info()
df.describe()

snsapi.set()
df.hist(bins = 10, figsize = (15,10));
plt.show();

corr_matrix = df.corr()
corr_matrix['MEDV']

sns.heatmap(corr_matrix);
plt.show()

print(boston['DESCR'])

dat1 = df.loc[:, ['CRIM', 'ZN', 'INDUS', 'NOX', 'RM', 'AGE', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]

X_train, X_test, y_train, y_test = train_test_split(dat1, target, test_size = 0.2, random_state=42)
y_train = y_train.values.ravel()

models = []
models.append(('SVR', SVR()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('DT', DecisionTreeRegressor()))
models.append(('RF', RandomForestRegressor()))
models.append(('l', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('R', Ridge()))
models.append(('BR', BayesianRidge()))
models.append(('GBR', GradientBoostingRegressor()))
models.append(('RF', AdaBoostRegressor()))
models.append(('ET', ExtraTreesRegressor()))
models.append(('BgR', BaggingRegressor()))

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

pipeline = make_pipeline(preprocessing.StandardScaler(), GradientBoostingRegressor(random_state=42))

hyperparameters = { 'gradientboostingregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'gradientboostingregressor__max_depth': [None, 5, 3, 1],
                  'gradientboostingregressor__n_estimators': [100, 150, 200, 250]}

print('\nRunning GridSearch')
clf = GridSearchCV(pipeline, hyperparameters, cv=10, scoring = scoring)
clf.fit(X_train, y_train)

print('\nRunning RandomizedSearch')
clf1 = RandomizedSearchCV(pipeline, hyperparameters, cv=10, random_state=42)
clf1.fit(X_train, y_train)

pred = clf.predict(X_test)
print("MSE for GridSearchCV: {}". format(mean_squared_error(y_test, pred)))

pred1 = clf1.predict(X_test)
print("MSE for RandomizedSearchCV: {}". format(mean_squared_error(y_test, pred1)))

from sklearn.externals import joblib 
joblib.dump(clf1, 'boston_regressor.pkl')

clf2 = joblib.load('boston_regressor.pkl')

