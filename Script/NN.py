from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


#read file
data_path = "../DATA/training_set.csv"
data = pd.read_csv(data_path, header=None)
print(data.shape)
x = np.array(data.iloc[:, 1:11])
y = np.array(data.iloc[:, 11:])
print(x.shape)
print(y.shape)

# define and fit the final model
model = Sequential()
model.add(Dense(50, input_dim=10, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2,  activation='relu'))
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(x,y,epochs=100, verbose=1, batch_size=None)