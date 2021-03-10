# Report on Happiness levels in Belarus from 2015 to 2019 and visualization of data
## The Code
```py
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

"""# Imports"""

data2015 = pd.read_csv("2015.csv")
data2016 = pd.read_csv("2016.csv")
data2017 = pd.read_csv("2017.csv")
data2018 = pd.read_csv("2018.csv")
data2019 = pd.read_csv("2019.csv")
data2015.head()

"""# Prepping data files and Concatenating dataframes"""

Belarus2015 = data2015.loc[data2015["Country"]=="Belarus"]
Belarus2015 = Belarus2015.drop(["Region", "Standard Error",'Dystopia Residual'], axis=1)
Belarus2015 = Belarus2015.rename(columns={"Country":"Country",
                                          "Happiness Score": "Happiness", 
                                          "Happiness Rank":"Rank", 
                                          "Economy (GDP per Capita)":"Economy",
                                          "Health (Life Expectancy)" : "Health",
                                          "Trust (Government Corruption)": "Trust",
                                          "Family":"Family",
                                          "Freedom": "Freedom"})
Belarus2015

Belarus2016 = data2016.loc[data2016["Country"]=="Belarus"]
Belarus2016 = Belarus2016.drop(["Region", 'Lower Confidence Interval', 'Dystopia Residual','Upper Confidence Interval'], axis=1)
Belarus2016 = Belarus2016.rename(columns={"Country":"Country",
                                          "Happiness Score": "Happiness", 
                                          "Happiness Rank":"Rank", 
                                          "Economy (GDP per Capita)":"Economy",
                                          "Health (Life Expectancy)" : "Health",
                                          "Trust (Government Corruption)": "Trust",
                                          "Family":"Family",
                                          "Freedom": "Freedom"})
Belarus2016

Belarus2017 = data2017.loc[data2017["Country"]=="Belarus"]
Belarus2017 = Belarus2017.drop(['Whisker.high','Dystopia.Residual','Whisker.low'], axis=1)
Belarus2017 = Belarus2017.rename(columns={"Country":"Country",
                                          "Happiness.Score": "Happiness", 
                                          "Happiness.Rank":"Rank", 
                                          "Economy..GDP.per.Capita.":"Economy",
                                          "Health..Life.Expectancy." : "Health",
                                          "Trust..Government.Corruption.": "Trust",
                                          "Family":"Family",
                                          "Freedom": "Freedom", "Dystopia.Residual" : "Dystopia Residual"})
Belarus2017

Belarus2019 = data2019.loc[data2019["Country or region"]=="Belarus"]

Belarus2019 = Belarus2019.rename(columns={"Country or region":"Country",
                                          "Score": "Happiness", 
                                          "Overall rank":"Rank", 
                                          "GDP per capita":"Economy",
                                          "Healthy life expectancy" : "Health",
                                          "Perceptions of corruption": "Trust",
                                          "Social support":"Family",
                                          "Freedom to make life choices": "Freedom"})

Belarus2019

Belarus2018 = data2018.loc[data2018["Country or region"]=="Belarus"]
Belarus2018 = Belarus2018.rename(columns={"Country or region":"Country",
                                          "Score": "Happiness", 
                                          "Overall rank":"Rank", 
                                          "GDP per capita":"Economy",
                                          "Healthy life expectancy" : "Health",
                                          "Perceptions of corruption": "Trust",
                                          "Social support":"Family",
                                          "Freedom to make life choices": "Freedom"})
Belarus2018

belarus = pd.concat([Belarus2015, Belarus2016, Belarus2017, Belarus2018, Belarus2019])
belarus = belarus.rename(index={58:"2015", 60:"2016", 66:"2017", 72:"2018", 80:"2019"})
belarus = belarus.drop('Country', axis=1)
belarus

"""# Plotting and EDA"""

import seaborn as sns
from numpy import median

ax = sns.barplot(x=belarus.index, y=belarus.Happiness,data=belarus,estimator=median,  ci=68, capsize=.2, palette="Blues_d")

"""Happiness is going down year by year steadily"""

axs = sns.lineplot(x=belarus.index, y=belarus["Freedom"])

"""It is a sharp slope, but something happened between 2016 and 2019 that involved freedom movement.

Now, let's look at corruption
"""

sns.lineplot(x=belarus.index, y=belarus["Trust"], palette="mako_r")

"""Yikes!"""

import matplotlib.pyplot as plt

f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(belarus.corr(),annot=True,linewidth=.5,fmt='.1f',ax=ax)
plt.show()

"""Interesting correlation between Family (or social support) with Happiness!

# Predicting the 2020
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

belarus



X = belarus.drop(columns='Happiness')
y = belarus.Happiness



from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     random_state=1)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
forest = RandomForestRegressor()
params = [
    {'n_estimators' : [3,10,5], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators': [3,5,10], 'max_features':[2,3,4,5]}
]

clf = GridSearchCV(forest, params, cv=3, scoring='neg_mean_squared_error', return_train_score=True)
clf.fit(X_train, y_train)
clf.best_params_

model = clf.best_estimator_

model.fit(X,y)
