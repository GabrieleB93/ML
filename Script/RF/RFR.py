from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from utils import *
from sklearn.preprocessing import StandardScaler

Predict = False
CV = False
GRID = False
Plot = 'OLD'

# According to the arguments, it will have different effects. See more on REadMe.dm
def rfr_model(x, y):

    if GRID:
        # Define grid and the list of the values for each hyperparameters
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(criterion='mse'),
            param_grid={
                'max_depth': range(1, 11),
                'n_estimators': (10, 50, 100, 500, 1000),
                'min_samples_split': [2, 3, 4, 5],
                'bootstrap': [True],
                'max_features': [10, 3],
            },
            cv=3, scoring=scoring, verbose=0, n_jobs=-1, refit=False)

        grid_result = gsc.fit(x, y)

        # Print on CSV file the results of the grid search
        print_and_saveGrid(grid_result, True, False, 'grid_search_result_RFR', 'RFR')

        print(grid_result.cv_results_['mean_test_mee'])
        print(grid_result.cv_results_['mean_test_loss'])
        print(grid_result.cv_results_['params'])

    if CV:

        print("CV")
        data = pd.read_csv(FIRST_GRID_RFR, sep=',', index_col=False)

        # Select the best model, chosen in the previous grid search
        best_row = data[data.mee == data.mee.min()]
        max_depth = int(best_row.iloc[0]['max_depth'])
        n_estimators = int(best_row.iloc[0]['n_estimators'])
        bootstrap = best_row.iloc[0]['bootstrap']
        max_features = int(best_row.iloc[0]['max_features'])
        min_samples_split = int(best_row.iloc[0]['min_samples_split'])
        max_leaf = 30

        if bootstrap:
            result = True
        else:
            result = False

        # Define the model
        rfr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators,
                                    min_samples_split=min_samples_split,
                                    max_features=max_features,
                                    bootstrap=bootstrap,
                                    max_leaf_nodes=max_leaf,
                                    random_state=False, verbose=False, oob_score=result)

        print(best_row)

        print('IN CV')
        fig = plt.figure()
        plt.title('LEARNING CURVE RFR')
        plt.xlabel("Training examples")
        plt.ylabel("MSE")

        # Make a k-fold cross validation to plot a default learning curve
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
            file = "RFR_LC_" + str(max_leaf) + "_" + t + ".png"
            if not os.path.exists(directory):
                os.makedirs(directory)
            fig.savefig(directory + file)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit("Need 1 argument")
    else:
        arg = sys.argv[1]
        parseArg(arg)
        print(arg)
        print(CV)

        X, Y = getTrainData(CUP, '1:11', '11:13', ',')
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        rfr_model(X, Y)
