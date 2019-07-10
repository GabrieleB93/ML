from sklearn.metrics import make_scorer
import numpy as np

Monk1TS = '../../DATA/MONK/monks-1.test'
Monk1TR = '../../DATA/MONK/monks-1.train'
Monk2TS = '../../DATA/MONK/monks-2.test'
Monk2TR = '../../DATA/MONK/monks-2.train'
Monk3TS = '../../DATA/MONK/monks-3.test'
Monk3TR = '../../DATA/MONK/monks-3.train'

ALL_DATA = '../../DATA/ML-CUP18-TR.csv'

BLIND_DATA = '../../DATA/ML-CUP18-TS.csv'

CUP = "../../DATA/training_set.csv"
TRAIN_SET = "../../DATA/tr_set.csv"
VAL_SET = "../../DATA/val_set.csv"

FIRST_GRID_POLY = "../../DATA/SVR_POLY/grid_search_result_SVR_POLY.csv"

FIRST_GRID_RBF = "../../DATA/SVR_RBF/grid_search_result_SVR_RBF.csv"

FIRST_GRID_NN = "../../DATA/NN/grid_search_result_MPL.csv"
SECOND_GRID_NN = "../../DATA/NN/grid_search_result_MPL2.csv"

FIRST_GRID_RFR = "../../DATA/RFR/grid_search_result_RFR.csv"

FIRST_GRID_ETR = "../../DATA/ETR/grid_search_result_ETR.csv"


def mean_euclidean_error(X, Y):
    sum = 0
    for x, y in zip(X, Y):
        sum += np.linalg.norm(x - y)
    return sum / X.shape[0]


scoring = {'loss': 'neg_mean_squared_error', 'mee': make_scorer(mean_euclidean_error)}
