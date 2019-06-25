import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import NN2

# read file
data_path = "../DATA/training_set.csv"
data = pd.read_csv(data_path, header=None)
print(data.shape)
x = np.array(data.iloc[:, 1:11])
y = np.array(data.iloc[:, 11:])
print(x.shape)
print(y.shape)

# batches = x.shape[0]

scaler = StandardScaler()

scaler.fit_transform(x)

batches = [64, 914]
learning_rates = [0.001, 0.0002, 0.0001]
layers = [1, 3, 5]
layer_size = [100, 500]
n_epochs = 5000


results_records = {'n_layers': [], 'hidden_layers_size': [], 'batch_size': [], 'learning_rate': [], 'validation_loss': [], 'mee': []}
for batch_size in batches:
    for n_layers in layers:
        for hidden_size in layer_size:
            for lr in learning_rates:
                loss, mee = NN2.train_and_validation(x, y, batch_size=batch_size, lr=lr, n_layers=n_layers, hidden_size=hidden_size,
                                                     epochs=n_epochs, activation='relu')
                results_records['n_layers'].append(n_layers)
                results_records['hidden_layers_size'].append(hidden_size)
                results_records['batch_size'].append(batch_size)
                results_records['learning_rate'].append(lr)
                results_records['validation_loss'].append(loss)
                results_records['mee'].append(mee)

                print("Batch size: ", batch_size, " N. layers: ", n_layers, " Hidden layer size: ", hidden_size, " Learning rate: ", lr, " DONE.")
                print("RESULTS - MEE: ", mee, " Validation loss: ", loss)

print(results_records)
results = pd.DataFrame(data=results_records)

filepath = "../DATA/grid_search_result_MPL"
file = open(filepath, mode='w')
results.to_csv(file, sep=',', header=True, index=False)

print(results)