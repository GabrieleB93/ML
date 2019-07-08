from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from utils import *
from sklearn.preprocessing import StandardScaler


def rfr_model(X, y):  # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(criterion='mse'),
        param_grid={
            'max_depth': range(3, 7),
            'n_estimators': (10, 50, 100, 1000),
        },
        cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_

    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],
                                random_state=False, verbose=False)

    scores = cross_validate(rfr, X, y, cv=3, scoring=scoring)
    print(scores)


X, Y = getTrainData(CUP, '1:11', '11:13', ',')

rfr_model(X, Y)