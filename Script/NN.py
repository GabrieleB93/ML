from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

batches = 914

# read file
data_path = "../DATA/dev_set.csv"
data = pd.read_csv(data_path, header=None)
print(data.shape)
x = np.array(data.iloc[:, 1:11])
y = np.array(data.iloc[:, 11:])
print(x.shape)
print(y.shape)

scaler = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, shuffle=True)

scaler.fit(x_train)
scaler.transform(x_train)
scaler.transform(x_test)

# define and fit the final model
model = Sequential()
model.add(Dense(50, input_dim=10, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='linear'))
mySGD = SGD(lr=0.0002, momentum=0.9, nesterov=False, decay=0)
model.compile(loss='mean_squared_error', optimizer=mySGD, metrics=['accuracy'])
#model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
hist = model.fit(x_train, y_train, shuffle=True, epochs=5000, verbose=2, batch_size=batches, validation_data=(x_test, y_test))

print("Min loss:", min(hist.history['val_loss']))

pyplot.plot(hist.history['loss'], label="TR Loss")
pyplot.plot(hist.history['val_loss'], label="VL Loss")
pyplot.ylim((0, 2))
pyplot.legend(loc='upper right')
pyplot.show()


