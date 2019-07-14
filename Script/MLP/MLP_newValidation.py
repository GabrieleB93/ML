from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from utils import *
from config import *

# According to the arguments, it will have different effects. See more on REadMe.dm
def main():
    if exists(FIRST_GRID_NN):

        grid_result = pd.read_csv(FIRST_GRID_NN, sep=',', index_col=False)

        best_row = grid_result[grid_result.mee == grid_result.mee.min()]
        n_layers = int(best_row.iloc[0]['n_layers'])
        hidden_layers_units = int(best_row.iloc[0]['hidden_layers_size'])
        batch = int(best_row.iloc[0]['batch_size'])
        # learning_rate = best_row.iloc[0]['learning_rate']
        learning_rate = 0.0001
        epochs = 3000
        momentum = 0.9
        lamda = 0

        # If there is no argument, use the tr_set and val_set to plot the curves
        if len(sys.argv) == 1:
            X_train, Y_train = getTrainData(TRAIN_SET, '1:11', '11:13', ',')
            X_val, Y_val = getTrainData(VAL_SET, '1:11', '11:13', ',')

            # train the model and plot the learning curves
            train_and_plot_MLP(X_train, Y_train, X_val, Y_val, n_layers, hidden_layers_units, learning_rate, momentum,
                               batch, epochs, lamda)

        # If the argument is 'CV', the model is Cross-validated with 3-Fold, through the GridSearchCV. Indeed, without
        #  a list of parameters the function executes a standard k-fold cross validation
        if len(sys.argv) > 1 and sys.argv[1].lower() == 'cv':
            X, Y = getTrainData(CUP, '1:11', '11:13', ',')
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)

            model = KerasRegressor(build_fn=create_model, epochs=epochs, verbose=2)
            param_grid = dict(learn_rate=learning_rate, units=hidden_layers_units, level=n_layers, batch_size=batch)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, refit=False, return_train_score=True,
                                cv=3, scoring=make_scorer(mean_euclidean_error))
            result = grid.fit(X, Y)
            print(result.cv_results_['mean_test_mee'])
    else:
        print("Make grid search first")


if __name__ == '__main__':
    main()
