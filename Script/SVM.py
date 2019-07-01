from sklearn.multioutput import MultiOutputRegressor
from utils import getTrainData, print_and_saveGrid
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from config import *

C = [0.1, 1, 10, 100]
gamma = [0.001, 0.01, 0.1, 1, 10, 100]
epsilon = [0.001, 0.01]
degree = [2, 3, 4]
# gamma_poly = [0.01, 0.1]


def main():
    X, Y = getTrainData(CUP, '1:11', '11:14', ',')


    # Pipeline per SVR multiOutput
    SVR_RBF = Pipeline([('reg', MultiOutputRegressor(SVR(verbose=True, kernel='rbf')))])
    SVR_POLY = Pipeline([('reg', MultiOutputRegressor(SVR(verbose=True, kernel='poly')))])

    # Parameters per gridSearch
    grid_param_svr_rbf = {
        'reg__estimator__C': C, 'reg__estimator__gamma': gamma, 'reg__estimator__epsilon': epsilon}
    grid_param_svr_poly = {
        'reg__estimator__C': C, 'reg__estimator__degree': degree, 'reg__estimator__epsilon': epsilon,
        # 'reg__estimator__gamma': gamma_poly
    }

    # GridSearch and CrossValidation
    mlt1 = GridSearchCV(estimator=SVR_RBF, param_grid=grid_param_svr_rbf, refit=False, return_train_score=True, cv=3,
                        scoring=scoring)
    mlt2 = GridSearchCV(estimator=SVR_POLY, param_grid=grid_param_svr_poly, refit=False, return_train_score=True, cv=3,
                        scoring=scoring)

    # Start training and  eventually plot
    print("Start SVR grid with RBF")
    print_and_saveGrid(mlt1.fit(X, Y), save=True, plot=True, nameResult='grid_search_result_SVR_RBF:TODELTE', Type='SVR_RBF')
    print("Start SVR grid with POLY")
    # print_and_saveGrid(mlt2.fit(X, Y), save=True, plot=True, nameResult='grid_search_result_SVR_POLY', Type='SVR_POLY')


if __name__ == '__main__':
    main()
