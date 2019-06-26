import numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from keras.models import Sequential
from sklearn.metrics import make_scorer
from keras.layers import Dense
from sklearn.metrics import log_loss, SCORERS
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, Adam


def score_f(y_true, y_pred, sample_weight):
    return log_loss(y_true.values, y_pred,
                    sample_weight=sample_weight.loc[y_true.index.values].values.reshape(-1),
                    normalize=True)


# Function to create model, required for KerasClassifier
def create_model(learn_rate=0.01, momentum=0, optimizer='SGD', units=100):
    # create model
    model = Sequential()
    model.add(Dense(units=units, input_dim=10, activation='relu'))
    model.add(Dense(2, activation='linear'))
    # Compile model
    opt = optimizer
    if optimizer == "SGD":
        opt = SGD(lr=learn_rate, momentum=momentum)
    else:
        opt = Adam(learn_rate)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    return model


data_path = "../DATA/training_set.csv"
data = pd.read_csv(data_path, header=None, sep=',')
print(data.shape)
X = np.array(data.iloc[:, 1:11])
Y = np.array(data.iloc[:, 11:])

index = ['r%d' % x for x in range(len(Y))]
y_frame = pd.DataFrame(Y, index=index)
sample_weight = np.array([1 + 100 * (i % 25) for i in range(len(X))])
sample_weight_frame = pd.DataFrame(sample_weight, index=index)

# create model
model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=914, verbose=2)
# define the grid search parameters
learn_rate = [0.001, 0.0001]
momentum = [0.8, 0.9]
units = [50, 100]
optimizer = ['SGD']
param_grid = dict(learn_rate=learn_rate, momentum=momentum, units=units, optimizer=optimizer)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)


inner_cv = RepeatedKFold(n_splits=3, n_repeats=1, random_state=1337)



grid = GridSearchCV(estimator=model,
                    scoring='brier_score_loss',
                    cv=inner_cv,
                    param_grid=param_grid,
                    refit=False,
                    return_train_score=True,
                    iid=False)

grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
