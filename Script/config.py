from sklearn.metrics import make_scorer
import numpy as np

Monk1TS = '../DATA/Monk/monks-1.test'
Monk1TR = '../DATA/Monk/monks-1.train'
Monk2TS = '../DATA/Monk/monks-2.test'
Monk2TR = '../DATA/Monk/monks-2.train'
Monk3TS = '../DATA/Monk/monks-3.test'
Monk3TR = '../DATA/Monk/monks-3.train'
ALL_DATA = '../DATA/ML-CUP18-TR.csv'

BLIND_DATA = '../DATA/ML-CUP18-TS.csv'

CUP = "../DATA/training_set.csv"
TRAIN_SET = "../DATA/tr_set.csv"
VAL_SET = "../DATA/val_set.csv"

FIRST_GRID_POLY = "../DATA/grid_search_result_SVR_POLY.csv"

FIRST_GRID_RBF = "../DATA/grid_search_result_SVR_RBF.csv"

FIRST_GRID_NN = "../DATA/grid_search_result_MPL.csv"

FIRST_GRID_RFR = "../DATA/grid_search_result_RFR.csv"
SECOND_GRID_RFR = "../DATA/grid_search_result_RFR1.csv"
THIRD_GRID_RFR = "../DATA/grid_search_result_RFR2.csv"

FIRST_GRID_ETR = "../DATA/grid_search_result_ETR.csv"
SECOND_GRID_ETR = "../DATA/grid_search_result_ETR1.csv"
THIRD_GRID_ETR = "../DATA/grid_search_result_ETR2.csv"

def mean_euclidean_error(X, Y):
    sum = 0
    for x, y in zip(X, Y):
        sum += np.linalg.norm(x - y)
    return sum / X.shape[0]


scoring = {'loss': 'neg_mean_squared_error', 'mee': make_scorer(mean_euclidean_error)}



