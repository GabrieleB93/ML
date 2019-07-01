from os.path import exists
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utils import *
from config import *


def main():
    if exists(FIRST_GRID_POLY and FIRST_GRID_RBF):
        data_RBF = pd.read_csv(FIRST_GRID_RBF, sep=',', index_col=False)
        data_POLY = pd.read_csv(FIRST_GRID_POLY, sep=',', index_col=False)

        epsilon_RBF = data_RBF.sort_values('mee').iloc[0]['epsilon']
        epsilon_POLY = data_POLY.sort_values('mee').iloc[0]['epsilon']
        degree = int(data_POLY.sort_values('mee').iloc[0]['degree'])

        C_range_RBF = getIntervalHyperP(data_RBF, 'C')
        C_range_POLY = getIntervalHyperP(data_POLY, 'C')
        gamma_range_RBF = getIntervalHyperP(data_RBF, 'gamma')

        X, Y = getTrainData(CUP)

        print(epsilon_POLY)
        print(epsilon_RBF)
        print(degree)

        print(C_range_POLY)
        print(C_range_RBF)
        print(gamma_range_RBF)

        # Pipeline per SVR multiOutput
        SVR_RBF = Pipeline([('reg', MultiOutputRegressor(SVR(verbose=True, kernel='rbf')))])
        SVR_POLY = Pipeline(
            [('reg', MultiOutputRegressor(SVR(verbose=True, kernel='poly', gamma=0.1)))])

        # Parameters per gridSearch
        grid_param_svr_rbf = {
            'reg__estimator__C': C_range_RBF, 'reg__estimator__gamma': gamma_range_RBF,
            'reg__estimator__epsilon': [epsilon_RBF]}
        grid_param_svr_poly = {
            'reg__estimator__C': C_range_POLY, 'reg__estimator__degree': [degree],
            'reg__estimator__epsilon': [epsilon_POLY]}

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

    else:
        print("First make SVM")


if __name__ == '__main__':
    main()
