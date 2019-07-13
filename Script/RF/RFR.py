from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from utils import *
import sys

Predict = False
CV = False
GRID = False
Plot = 'OLD'


# Plot = 'NEW'


def rfr_model(x, y):  # Perform Grid-Search

    if GRID:
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(criterion='mse'),
            param_grid={
                'max_depth': range(1, 11),
                'n_estimators': (10, 50, 100, 500, 1000),
                # 'n_estimators': (10,50),
                'min_samples_split': [2, 3, 4, 5],
                # 'min_samples_leaf': [1,2,3],
                'bootstrap': [True],
                # 'random_state': [1],
                'max_features': [10, 3],
                # 'min_impurity_decrease': [0., 1.],

            },
            cv=3, scoring=scoring, verbose=0, n_jobs=-1, refit=False)

        grid_result = gsc.fit(x, y)
        best_params = grid_result
        print_and_saveGrid(grid_result, True, False, 'grid_search_result_RFR', 'RFR')
        # print(best_params)
        # print(grid_result.cv_results_['mean_test_score'])
        print(grid_result.cv_results_['mean_test_mee'])
        print(grid_result.cv_results_['mean_test_loss'])
        print(grid_result.cv_results_['params'])

    if exists(FIRST_GRID_RFR):

        print("IN BEOFRE CROSS")
        data = pd.read_csv(FIRST_GRID_RFR, sep=',', index_col=False)

        best_row = data[data.mee == data.mee.min()]
        max_depth = int(best_row.iloc[0]['max_depth'])
        n_estimators = int(best_row.iloc[0]['n_estimators'])
        bootstrap = best_row.iloc[0]['bootstrap']
        max_features = int(best_row.iloc[0]['max_features'])
        min_samples_split = int(best_row.iloc[0]['min_samples_split'])

        if bootstrap:
            result = True
        else:
            result = False

        rfr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                    min_samples_split=min_samples_split,
                                    max_features=max_features,
                                    bootstrap=bootstrap,
                                    max_leaf_nodes=None,
                                    random_state=False, verbose=False, oob_score=result)

        print(best_row)
        if CV:
            print('IN CV')
            fig = plt.figure()
            plt.title('LEARNING CURVE RFR')
            plt.xlabel("Training examples")
            plt.ylabel("MSE")
            train_sizes, train_scores, test_scores = learning_curve(
                rfr, x, y, cv=3, n_jobs=-1
                # , scoring=make_scorer(mean_euclidean_error)
                , scoring='neg_mean_squared_error'
            )

            print("TRAIN")
            print(train_scores)
            print("TEST")
            print(test_scores)
            print("SIZE")
            print(train_sizes)

            if Plot == 'OLD':
                train_scores_mean = abs(np.mean(train_scores, axis=1))
                train_scores_std = abs(np.std(train_scores, axis=1))
                test_scores_mean = abs(np.mean(test_scores, axis=1))
                test_scores_std = abs(np.std(test_scores, axis=1))
                plt.grid()

                plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                 train_scores_mean + train_scores_std, alpha=0.1,
                                 color="r")
                plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
                plt.plot(train_sizes, train_scores_mean, '-', color="blue",
                         label="TR LOSS")
                plt.plot(train_sizes, test_scores_mean, '--', color="orange",
                         label="VL LOSS")

                plt.legend(loc="best")
                plt.show()

                directory = "../../Image/RFR/"
                t = strftime("%H_%M")
                file = "RFR_LC"+t+".png"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                fig.savefig(directory + file)

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


def parseArg(arg):
    if arg.lower() == 'grid':
        global GRID
        GRID = True
    elif arg.lower() == 'cv':
        global CV
        CV = True
    elif arg.lower() == 'predict':
        global Predict
        Predict = True
    else:
        sys.exit("Argument not recognized")


if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit("Need 1 argument")
    else:
        arg = sys.argv[1]
        parseArg(arg)
        print(arg)
        print(CV)

        X, Y = getTrainData(CUP_NEW, '1:11', '11:13', ',')
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        rfr_model(X, Y)
