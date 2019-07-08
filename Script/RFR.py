from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import MinMaxScaler
from utils import *
from sklearn.preprocessing import StandardScaler

Predict = False
CV = True
GRID = False
# Plot = 'OLD'
Plot = 'NEW'


def rfr_model(x, y):  # Perform Grid-Search

    if GRID:
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(criterion='mse'),
            param_grid={
                'max_depth': range(2, 10),
                'n_estimators': (10, 50, 100, 1000),
                'min_samples_split': [2]
            },
            cv=3, scoring=scoring, verbose=0, n_jobs=-1, refit=False)

        grid_result = gsc.fit(x, y)
        best_params = grid_result
        print_and_saveGrid(grid_result, True, True, 'grid_search_result_RFR', 'RFR')
        # print(best_params)
        # print(grid_result.cv_results_['mean_test_score'])
        print(grid_result.cv_results_['mean_test_mee'])
        print(grid_result.cv_results_['mean_test_loss'])
        print(grid_result.cv_results_['params'])

    if FIRST_GRID_RFR:

        print("IN BEOFRE CROSS")
        data = pd.read_csv(FIRST_GRID_RFR, sep=',', index_col=False)

        best_row = data[data.mee == data.mee.min()]
        max_depth = int(best_row.iloc[0]['max_depth'])
        n_estimators = int(best_row.iloc[0]['n_estimators'])
        min_samples_split = int(best_row.iloc[0]['min_samples_split'])

        rfr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, min_samples_split=min_samples_split,
                                    random_state=False, verbose=False)

        print(best_row)
        if CV:
            print('IN CV')
            plt.figure()
            plt.title('LEARNING CURVE')
            plt.xlabel("Training examples")
            plt.ylabel("Score")
            train_sizes, train_scores, test_scores = learning_curve(
                rfr, x, y, cv=3, n_jobs=-1, scoring=make_scorer(mean_euclidean_error)   )

            print("TRAIN")
            print(train_scores)
            print("TEST")
            print(test_scores)
            print("SIZE")
            print(train_sizes)

            if Plot == 'OLD':
                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                test_scores_mean = np.mean(test_scores, axis=1)
                test_scores_std = np.std(test_scores, axis=1)
                plt.grid()

                plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
                plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
                plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                         label="Training score")
                plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                         label="Cross-validation score")

                plt.legend(loc="best")
                plt.show()
            elif Plot == 'NEW':
                plt.plot(train_scores, label="TR Loss")
                plt.plot(test_scores, label="VL Loss")
                plt.show()
            # scores = cross_validate(rfr, x, y, cv=3, scoring=scoring)
            # print(scores)

        if Predict:
            X = pd.read_csv("../DATA/ML-CUP18-TS.csv", comment='#', header=None)

            y = rfr.fit(x, y).predict(X.iloc[:, 1:11])

            fig = plt.figure()
            plt.plot(y[:, 0], y[:, 1], 'ro')
            plt.show()

    # print(scores)


X, Y = getTrainData(CUP, '1:11', '11:13', ',')

rfr_model(X, Y)
