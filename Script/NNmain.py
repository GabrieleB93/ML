import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# read file
data_path = "training_set.csv"
data = pd.read_csv(data_path, header=None)
print(data.shape)
x = np.array(data.iloc[:, 1:11])
y = np.array(data.iloc[:, 11:])
print(x.shape)
print(y.shape)


# batches = x.shape[0]

scaler = StandardScaler()

scaler.fit_transform(x)

loss, mee = NN2.train(x, y, batch_size=914, lr=0.0001, n_layers=1, hidden_size=50, epochs=500, activation='relu')

print("MEE: ", mee)
print("LOSS: ", loss)

