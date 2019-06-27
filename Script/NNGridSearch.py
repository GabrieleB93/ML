from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from keras.wrappers.scikit_learn import KerasRegressor
from utils import *

# Variabili
scoring = {'loss': 'neg_mean_squared_error', 'mee': make_scorer(mean_euclidean_error)}
learn_rate = [0.0001, 0.001, 0.0002]
units = [100, 500]
level = [1, 3, 5]
epochs = 5000


def main():
    X, Y = getTrainData()
    batch_size = [X.shape[0], 64]
    print(X.shape[0])

    # create model
    model = KerasRegressor(build_fn=create_model, epochs=epochs, verbose=2)

    # define the grid search parameters
    param_grid = dict(learn_rate=learn_rate, units=units, level=level, batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, refit=False, return_train_score=True, cv=3,
                        scoring=scoring)

    print_and_saveGrid(grid.fit(X, Y), 'loss', 'mee', True, 'grid_search_result_MLP', 'NN')


if __name__ == "__main__":
    main()
