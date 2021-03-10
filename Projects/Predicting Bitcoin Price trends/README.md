### General Information
The project presented here is for demonstration purpose only.

## Purpose
To demonstrate knowledge of following skills:
* python programming skills
* Building regression models
* Cross Validation of models

## Data
Data used is the representation of Bitcoin price for the period from 2019-09-19 to 2020-09-19

## Thought Process
The idea is to split the dataset into periods of 3 weeks, take two first weeks and make a prediction of the following week. 
The test set is then compared with the prediction to demonstrate the Mean Absolute Error of the models' predictions

## The Code
 ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    data = pd.read_csv('BTC-USD.csv')

    data.head()

    curs = data.Close.round(3).dropna()

    print(curs.describe())
    curs.plot()

    # X - prediction intro data (14 days prior)
    # y - intro data the predictor (7 days future)
    future_days = 7
    past_days = 14

    start = past_days
    end = len(curs) - future_days
    total = end - start

    print(f"start = {start}, end = {end}, total = {total}")

    past_X = []
    future_y = []

    for i in range(start,end):
        X = curs[i-past_days:i]
        past_X.append(list(X))
        y = curs[i:i+future_days]
        future_y.append(list(y))

    past_columns = []

    for i in range(past_days):
        past_columns.append(f"past_{i}")
    print(past_columns)

    future_columns = []

    for i in range(future_days):
        future_columns.append(f"future_{i}")
    print(future_columns)

    df_X = pd.DataFrame(data = past_X, columns = past_columns)

    df_y = pd.DataFrame(data = future_y, columns = future_columns)

    # 1. Train model:
        #Train set

    X_train = df_X[0:-10]    
    y_train = df_y[0:-10]    
    # 2. Test model:
        #Test set
    X_test = df_X[-10:]
    y_test = df_y[-10:]

    X_train

    # Regression - predicting the single digit outcome (non-descreet)
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    #list models
    models_list = ("KNeighborsRegressor", "RandomForestRegressor", "NN")

    ##Knn
    Knn = KNeighborsRegressor()
    Knn.fit(X_train, y_train)


    #forest
    forest = RandomForestRegressor(n_estimators=330)
    forest.fit(X_train, y_train)


    #NN
    nn = MLPRegressor(random_state=1, max_iter=1000).fit(X_train, y_train)

    knn_prediction = Knn.predict(X_test)
    forest_prediction = forest.predict(X_test)
    nn_prediction = nn.predict(X_test)
    models = (knn_prediction, forest_prediction, nn_prediction)
    test_models = (knn_prediction[0], forest_prediction[0], nn_prediction[0])

    #plot
    c = 0
    for test in test_models:
        plt.plot(test, label=models_list[c])
        c += 1

    plt.plot(y_test.iloc[0], label = "Real Data")
    plt.legend()

    ###Measuring Error
    ##Metrics

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_squared_log_error

    metrics_list = ("mae", "mse", "msle")
    c = 0
    for test in test_models:

        mae = mean_absolute_error(test, y_test.iloc[0])
        mse = mean_squared_error(test, y_test.iloc[0])
        msle = mean_squared_log_error(test, y_test.iloc[0])
        print(f"{models_list[c]} errors:  MAE-{mae}, MSE-{mse}, msle-{msle}")
        c += 1

    """##COnclusion
    Neural Network performed best at predicting the pattern of behavior of the trend; However, 
    Knn did best at predicting the value of the bitcoin
    """

    from sklearn.model_selection import GridSearchCV
    from numpy import linspace

    ##CROSS VALIDATION
    Gmodel = MLPRegressor(random_state = 42)
    param_grid = {
        "max_iter" : [500,1000,2000],
        "activation" : ["logistic", "relu"]
    }

    GS = GridSearchCV(Gmodel, param_grid, scoring = "neg_mean_absolute_error", cv=3)

    GS.fit(X_train, y_train)

    best_model = GS.best_estimator_

    best_model

