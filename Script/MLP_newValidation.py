import pandas as pd
from os.path import exists
from utils import *
import matplotlib.pyplot as plt

data_path = "../DATA/grid_search_result_MPL"
epochs = 5000


def main():
    if exists(data_path):
        data = pd.read_csv(data_path, sep=',', index_col=False)

        best_row = data[data.mee == data.mee.min()]
        hidden_layers_units = int(best_row.iloc[0]['hidden_layers_size'])
        batch = int(best_row.iloc[0]['batch_size'])
        n_layers = int(best_row.iloc[0]['n_layers'])
        learning_rate = best_row.iloc[0]['learning_rate']

        X, Y = getTrainData()
        model = create_model(learning_rate, hidden_layers_units, n_layers)
        history = model.fit(X, Y, shuffle=True, epochs=epochs, verbose=2, batch_size=batch, validation_split=0.2)

        print("Min loss:", min(history.history['val_loss']))

        plt.plot(history.history['loss'], label="TR Loss")
        plt.plot(history.history['val_loss'], label="VL Loss")
        plt.ylim((0, 2))
        plt.legend(loc='upper right')
        plt.show()





    else:
        print("First gridSearch on NN")


if __name__ == '__main__':
    main()
