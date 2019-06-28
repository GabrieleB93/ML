from os.path import exists
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from utils import scoring, getTrainData, print_and_saveGrid
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utils import *
import numpy as np

data_path_POLY = "../DATA/grid_search_result_SVR_POLY"
data_path_RBF = "../DATA/grid_search_result_SVR_RBF"


def main():
    if exists(data_path_POLY and data_path_RBF):
        data_RBF = pd.read_csv(data_path_RBF, sep=',', index_col=False)
        data_POLY = pd.read_csv(data_path_POLY, sep=',', index_col=False)

        best_row_POLY = data_POLY[data_POLY.mee == data_POLY.mee.min()]
        best_row_RBF = data_RBF[data_RBF.mee == data_RBF.mee.min()]

        epsilon_RBF = data_RBF.sort_values('mee').iloc[0]['epsilon']
        epsilon_POLY = data_POLY.sort_values('mee').iloc[0]['epsilon']
        degree = int(data_POLY.sort_values('mee').iloc[0]['degree'])

        C_range_RBF = getIntervalHyperP(data_RBF, 'C')
        C_range_POLY = getIntervalHyperP(data_POLY, 'C')
        gamma_range_RBF = getIntervalHyperP(data_RBF, 'gamma')

        X, Y = getTrainData()

        print(epsilon_POLY)
        print(epsilon_RBF)
        print(degree)

        print(C_range_POLY)
        print(C_range_RBF)
        print(gamma_range_RBF)

        # Pipeline per SVR multiOutput
        SVR_RBF = Pipeline([('reg', MultiOutputRegressor(SVR(verbose=True, kernel='rbf')))])
        SVR_POLY = Pipeline(
            [('reg', MultiOutputRegressor(SVR(verbose=True, kernel='poly')))])

        # Parameters per gridSearch
        grid_param_svr_rbf = {
            'reg__estimator__C': C_range_RBF, 'reg__estimator__gamma': gamma_range_RBF,
            'reg__estimator__epsilon': [epsilon_RBF]}
        grid_param_svr_poly = {
            'reg__estimator__C': C_range_POLY, 'reg__estimator__degree': [degree], 'reg__estimator__epsilon': [epsilon_POLY]}

        # GridSearch and CrossValidation
        mlt1 = GridSearchCV(estimator=SVR_RBF, param_grid=grid_param_svr_rbf, refit=False, return_train_score=True,
                            cv=3,
                            scoring=scoring
                            )
        mlt2 = GridSearchCV(estimator=SVR_POLY, param_grid=grid_param_svr_poly, refit=False, return_train_score=True,
                            cv=3,
                            scoring=scoring)

        # Start training and  eventually plot
        print("Start SVR grid with RBF")
        print_and_saveGrid(mlt1.fit(X, Y), save=True, plot=True, nameResult='grid_search_result_SVR_RBF_2',
                           Type='SVR_RBF')
        print("Start SVR grid with POLY")
        print_and_saveGrid(mlt2.fit(X, Y), save=True, plot=False, nameResult='grid_search_result_SVR_POLY_2',
                           Type='SVR_POLY')


if __name__ == '__main__':
    main()
