from sklearn import datasets
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib 

from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

cancer = datasets.load_breast_cancer()
dat = pd.DataFrame(cancer.data, columns=cancer.feature_names)
dat.head()

target = pd.DataFrame(cancer.target, columns = ['diagnosis'])
target.head()

df = dat.copy()
df = pd.concat([df, target], axis=1)
df.head()

df.columns
df.info()
df.describe()

corr_matrix = df.corr()
corr_matrix['diagnosis']

f, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corr_matrix, linewidth = 0.2, ax = ax);
plt.show()

newCol = []
for i, col in enumerate(df.columns):
    if corr_matrix['diagnosis'][i] > 0.3 or corr_matrix['diagnosis'][i] < -0.3:
        newCol.append(col)

dat1 = df.loc[:, newCol]

X_train, X_test, y_train, y_test = train_test_split(dat1, target, test_size = 0.2, random_state=42)
y_train = y_train.values.ravel()

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

models = []
models.append(('LR', LogisticRegression()))
models.append(('SVC', SVC()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('GBC', GradientBoostingClassifier()))
models.append(('ET', ExtraTreesClassifier()))
models.append(('RF', RandomForestClassifier()))

scoring = 'accuracy'

results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train);

pred = clf.predict(X_test)
cm = confusion_matrix(y_test, pred)  
print('Confusion matrix: {}'.format(cm))
print("Accuracy for GridSearchCV: {}". format(accuracy_score(y_test, pred)))

joblib.dump(clf, 'cancer_classifier.pkl')

clf2 = joblib.load('cancer_classifier.pkl')
