# Classification problem: Computer says "NO"
```py
import pandas as pd
data = pd.read_csv('/content/train.csv')

import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_auc_score, roc_curve

print(data.info())
data.head()



"""# Preprocessing

## Rid of missing values
"""

data.isnull().any()

data['Adjusted_Educ'] = np.where(data['education'].isnull(), 1, 0) # Note the replacement with average value

data.education = data.education.fillna('SCH')

"""## Binary columns

"""

bin_cols = ['sex','car','car_type','foreign_passport']
data.sex = data.sex.map(dict(M=1, F=0)) # Males are 1
data.car = data.car.map(dict(Y=1, N=0))
data.car_type = data.car_type.map(dict(Y=1, N=0))
data.foreign_passport = data.foreign_passport.map(dict(Y=1, N=0))

"""## Separate columns"""

non_obj_cols = data.select_dtypes(include=np.number).columns.tolist()
non_obj_cols

obj_cols = data.select_dtypes(exclude=["number","bool_"])
obj_cols

bool_cols = [col for col in data if np.isin(data[col].unique(), [0, 1]).all()]
bool_cols

num_cols = [col for col in non_obj_cols if col not in bool_cols]
num_cols

"""# EDA

## Distributions
"""

for i in num_cols:
    plt.figure()
    sns.distplot(data[i][data[i] > 0], kde = False, rug=False)
    plt.title(i)
    plt.show()

defaults = data.groupby('default').client_id.count()
plt.figure(figsize=(12,6))
plt.xticks(rotation=75)
plt.title('Defaults')
plott = sns.barplot(x = ['no','yes'], y = defaults)
plott.set(xlabel = 'defaults', ylabel = 'No. of clients');

sns.catplot(x="default", y="income", kind="box", data=data);

sns.catplot(x="default", y="age", kind="box", data=data);

"""## Correlation matrix

"""

sns.heatmap(data[non_obj_cols].corr().abs(), vmin=0, vmax=1)

data

from statsmodels.stats.outliers_influence import variance_inflation_factor
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
X= data.iloc[:,2:-2]
calc_vif(X)

"""# Pre-model

## Feature SElection

### Importance
"""

imp_num = pd.Series(f_classif(data[num_cols + ['sex']], data['default'])[0], index = num_cols+ ['sex'])
imp_num.sort_values(inplace = True)
imp_num.plot(kind = 'barh')

"""## Encoding"""

bin_cols = ['sex',
 'car',
 'car_type',
 'good_work',
 'foreign_passport',
 'Adjusted_Educ']

num_cols = [
 'age',
'decline_app_cnt','bki_request_cnt',
 'income']

cat_cols = ['education'	,'home_address','work_address']

data

X_cat = OneHotEncoder(sparse = False).fit_transform(data[cat_cols].values)



"""## Scaling"""

X_num = StandardScaler().fit_transform(data[num_cols].values)

"""## Finishing touches / unification

"""

# Объединяем

X = np.hstack([X_num, data[bin_cols].values, X_cat])
Y = data['default'].values

"""# Modeling

## Test split
"""

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

"""## Model"""

reg = LogisticRegression()
reg.fit(X_train, y_train)

"""## Model Quality

"""

probs = reg.predict_proba(X_test)
probs = probs[:,1]


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()

"""## Tuning

"""

from sklearn.model_selection import GridSearchCV

# Добавим типы регуляризации
penalty = ['11', 'l2']

# Зададим ограничения для параметра регуляризации
C = np.logspace(0, 4, 10)

# Создадим гиперпараметры
hyperparameters = dict(C=C, penalty=penalty)

model = LogisticRegression()
model.fit(X_train, y_train)

# Создаем сетку поиска с использованием 5-кратной перекрестной проверки
clf = GridSearchCV(model, hyperparameters, cv=5, verbose=0)

best_model = clf.fit(X_train, y_train)

# View best hyperparameters
print('Лучшее Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Лучшее C:', best_model.best_estimator_.get_params()['C'])



from sklearn.linear_model import LogisticRegressionCV
clf = LogisticRegressionCV(cv=5, random_state=0).fit(X_test, y_test)
probs = clf.predict_proba(X_test)
probs = probs[:,1]


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()

from sklearn.model_selection import cross_val_score
ridge = linear_model.Ridge()
lasso = linear_model.Lasso()
elastic = linear_model.ElasticNet()
lasso_lars = linear_model.LassoLars()
bayesian_ridge = linear_model.BayesianRidge()
logistic = linear_model.LogisticRegression(solver='liblinear')
sgd = linear_model.SGDClassifier()
models = [ridge, lasso, elastic, lasso_lars, bayesian_ridge, logistic, sgd]
# function to get cross validation scores
def get_cv_scores(model):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')
# loop through list of models
for model in models:
    print(model)
    get_cv_scores(model)

penalty = ['l1', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
class_weight = [{1:0.5, 0:0.5}, {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}]
solver = ['liblinear', 'saga']

param_grid = dict(penalty=penalty,
                  C=C,
                  class_weight=class_weight,
                  solver=solver)

grid = GridSearchCV(estimator=logistic, param_grid=param_grid, scoring='roc_auc', verbose=1, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print('Best Score: ', grid_result.best_score_)
print('Best Params: ', grid_result.best_params_)

logistic = linear_model.LogisticRegression(C=1, class_weight={1:0.6, 0:0.4}, penalty='l1', solver='liblinear')
get_cv_scores(logistic)



probs = logistic.fit(X_train, y_train).predict_proba(X_test)
probs = probs[:,1]


fpr, tpr, threshold = roc_curve(y_test, probs)
roc_auc = roc_auc_score(y_test, probs)

plt.figure()
plt.plot([0, 1], label='Baseline', linestyle='--')
plt.plot(fpr, tpr, label = 'Regression')
plt.title('Logistic Regression ROC AUC = %0.3f' % roc_auc)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()
