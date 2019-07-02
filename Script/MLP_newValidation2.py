from utils import *
from config import *

grid_result_path = "../DATA/grid_search_result_MLP_OLD"


def main():
    # X_train, Y_train, X_val, Y_val = split_development_set(20)

    X_train, Y_train = getTrainData(TRAIN_SET, '1:11', '11:14', ',')
    X_val, Y_val = getTrainData(VAL_SET, '1:11', '11:14', ',')

    if exists(grid_result_path):
        grid_result = pd.read_csv(grid_result_path, sep=',', index_col=False)

        best_row = grid_result[grid_result.mee == grid_result.mee.min()]
        n_layers = int(best_row.iloc[0]['n_layers'])
        hidden_layers_units = int(best_row.iloc[0]['hidden_layers_size'])
        batch = int(best_row.iloc[0]['batch_size'])
        learning_rate = best_row.iloc[0]['learning_rate']
        epochs = 5000
        momentum = 0.9

        train_and_plot_MLP(X_train, Y_train, X_val, Y_val, n_layers, hidden_layers_units, learning_rate, momentum,
                           batch, epochs, 0)

        train_and_plot_MLP(X_train, Y_train, X_val, Y_val, n_layers, hidden_layers_units, learning_rate, momentum,
                           batch, epochs, 0.001)

        train_and_plot_MLP(X_train, Y_train, X_val, Y_val, n_layers, hidden_layers_units, learning_rate, momentum,
                           128, 0.0001)

        train_and_plot_MLP(X_train, Y_train, X_val, Y_val, n_layers, hidden_layers_units, learning_rate, 0.6,
                           915, 0.0002)
        #
        # train_and_plot_MLP(X_train, Y_train, X_val, Y_val, n_layers, hidden_layers_units, 0.0004, 0.6,
        #                    915, epochs)
        #
        # train_and_plot_MLP(X_train, Y_train, X_val, Y_val, n_layers, hidden_layers_units, 0.0004, 0.6,
        #                    915, 10000)


if __name__ == '__main__':
    main()
