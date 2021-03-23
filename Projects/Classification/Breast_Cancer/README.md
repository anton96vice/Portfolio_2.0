# Breast Cancer Classification with SVC
```py
# BREAST CANCER CLASSIFICATION

# 1. The Problem

- Cancer can be either benign or malignant. We are set to classify the cancer based on the set of 30 features:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry 
        - fractal dimension ("coastline approximation" - 1)

- Number of Samples: 569; 212 of which are Malignant and 357 Benign


https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

![image.png](https://www.gvec.org/wp-content/uploads/2019/10/Breast-Cancer-Awareness-Month-2019.jpg)

# 2. Imports
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
# %matplotlib inline

# Import the data from Sklearn 
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

print(cancer['DESCR'])

print(cancer['feature_names'])
print(cancer['data'].shape)

# create a DataFrame from Cancer data
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']],
                          columns = np.append(cancer['feature_names'], ['target']))

df_cancer.sample(5)

"""# 3. EDA"""

sns.set(style="darkgrid")
plt.style.use("dark_background")
sns.set_palette('RdPu')
sns.pairplot(df_cancer, 
             hue = 'target', 
             vars = ['mean radius', 
                     'mean texture',
                     'mean area',
                     'mean perimeter', 
                     'mean smoothness'],
             palette="RdPu",
              diag_kind="hist")

plt.rcParams.update({"grid.linewidth":0.5, "grid.alpha":0.5})
sns.countplot(df_cancer['target'],palette='RdPu')

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer, palette='rocket')

sns.jointplot(x="mean area", y="mean smoothness", hue ='target', data = df_cancer, palette = 'RdPu')

# Let's check the correlation between the variables 
# Strong correlation between the mean radius and mean perimeter, mean area and mean primeter
plt.figure(figsize=(20,10)) 
sns.heatmap(df_cancer.corr(), annot=True, cmap='RdPu')

"""# 4. Training"""

X = df_cancer.drop(['target'],axis=1)
y = df_cancer['target']
display(X,y)

# split the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, shuffle = True, random_state=69)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train, y_train)

"""# 5. Validation"""

y_predict = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True, cmap='RdPu')
print(classification_report(y_test, y_predict))

"""# 6. Scaling"""

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_train_scaled

sns.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train, palette='rocket')

sns.scatterplot(x = X_train_scaled['mean area'], y = X_train_scaled['mean smoothness'], hue = y_train, palette='cool')

X_test_scaled = sc.fit_transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

from sklearn.svm import SVC 
from sklearn.metrics import classification_report, confusion_matrix

svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)

y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)
print(classification_report(y_test,y_predict))
sns.heatmap(cm,annot=True,fmt="d", cmap='RdPu')

"""# Tuning"""

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)

grid.best_params_

grid.best_estimator_

grid_predictions = grid.predict(X_test_scaled)

cm = confusion_matrix(y_test, grid_predictions)
sns.heatmap(cm, annot=True, cmap='RdPu')
print(classification_report(y_test,grid_predictions))
