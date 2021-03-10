# Classification Problem
```py
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/DAAG/spam7.csv')
RANDOM_SEED = 42

# Cleaning
df['spam'] = df.yesno.map({"y" : 1, "n": 0})
df.drop(columns=['Unnamed: 0','yesno'], inplace=True)
X = df.drop(columns='spam')
y = df.spam


# Feature Engineering
from itertools import combinations_with_replacement 
op = combinations_with_replacement(df.columns,2)
df = pd.DataFrame()

for p in combinations_with_replacement(X.columns,2):
        title = p
        df[title] = X[p[0]]*X[p[1]]  



new_feat = pd.concat([X, df], axis=1) 

# Split
from sklearn.model_selection import train_test_split
X = new_feat
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = RANDOM_SEED)
# Train baseline
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.1, random_state = RANDOM_SEED,
                                 n_estimators=100,max_depth=3, min_samples_split=2,
                                 min_samples_leaf=1, subsample=1,max_features=None)
# Compute Accuracy
from sklearn.metrics import accuracy_score
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
print("GBC accuracy is %2.2f" % accuracy_score(
    y_test, y_pred))
# Feature Importances
print("Feature Importances: ",pd.DataFrame([gbc.feature_importances_],  
                   columns=new_feat.columns.tolist()).sort_values(by=0,axis=1))
# parameter grid for testing
param_grid = {'learning_rate':[0.00001, 0.0001, 0.001, 0.01, 0.1, 1], 
              'n_estimators':[100, 250, 500, 750, 1000, 1250, 1500, 1750]}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(GradientBoostingClassifier(random_state=RANDOM_SEED),param_grid=param_grid, scoring='accuracy',n_jobs=-1,cv=5)
grid.fit(X_test,y_test)

# NEw Model
best_model = GradientBoostingClassifier(**grid.best_params_)
best_model.fit(X_train,y_train)


from sklearn.model_selection import cross_val_score
estimate_accuracy(best_model, X, y, cv=10)


params = {'max_depth':np.linspace(5,15, 11).tolist()}
new_grid = GridSearchCV(GradientBoostingClassifier(random_state=RANDOM_SEED,**grid.best_params_), param_grid=params)
new_grid.fit(X_train, y_train)

new_grid.best_params_
