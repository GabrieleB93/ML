from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from utils import *
from sklearn.preprocessing import StandardScaler
from config import *

# Variabili
learn_rate = [0.0001, 0.001]
units = [10, 50, 100, 500]
level = [1, 2, 3, 5]
epochs = 5000


def main():
    X, Y = getTrainData(CUP, '1:11', '11:13', ',')
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    batch_size = [X.shape[0], 64, 128]
    print(X.shape[0])

    # create model
    model = KerasRegressor(build_fn=create_model, epochs=epochs, verbose=2)

    # define the grid search parameters
    param_grid = dict(learn_rate=learn_rate, units=units, level=level, batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, refit=False, return_train_score=True, cv=3,
                        scoring=scoring)

    # Print on CSV file the results of the grid search
    print_and_saveGrid(grid.fit(X, Y), save=True, plot=False, nameResult='grid_search_result_MLP', Type='NN')


if __name__ == "__main__":
    main()
