from sklearn.metrics import make_scorer
from utils import mean_euclidean_error

Monk1TS = '../DATA/Monk/monks-1.test'
Monk1TR = '../DATA/Monk/monks-1.train'
Monk2TS = '../DATA/Monk/monks-2.test'
Monk2TR = '../DATA/Monk/monks-2.train'
Monk3TS = '../DATA/Monk/monks-3.test'
Monk3TR = '../DATA/Monk/monks-3.train'
CUP = "../DATA/training_set.csv"
TRAIN_SET = "../DATA/tr_set.csv"
VAL_SET = "../DATA/val_set.csv"
FIRST_GRID_POLY = "../DATA/grid_search_result_SVR_POLY.csv"
FIRST_GRID_RBF = "../DATA/grid_search_result_SVR_RBF.csv"
FIRST_GRID_NN = "../DATA/grid_search_result_MPL.csv"
scoring = {'loss': 'neg_mean_squared_error', 'mee': make_scorer(mean_euclidean_error)}
