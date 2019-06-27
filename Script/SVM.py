# # Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
# # License: BSD 3 clause
#
#
import time
#
import numpy as np
from utils import *
from sklearn.metrics import make_scorer
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge

import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor

rng = np.random.RandomState(0)
# C = [1.0,10.0,100.0,1000.0]
# gamma = [0.1, 0.2]

# #############################################################################
# Generate sample data
# X = 5 * rng.rand(10000, 1)
# y = np.sin(X).ravel()
#
# X, y = getTrainData()

# Add noise to targets
# y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

# X_plot = np.linspace(0, 5, 100000)[:, None]

# #############################################################################
# Fit regression model
# train_size = 100
# param_grid = dict(C = C, gamma = gamma)
# model = SVR(kernel='rbf', gamma=0.1, C=1.0)
# mlt = MultiOutputRegressor(model)
# mlt = model
# svr = GridSearchCV(estimator=mlt, cv=3, param_grid=param_grid)
# svr = mlt

# kr = GridSearchCV(MultiOutputRegressor(KernelRidge(kernel='rbf', gamma=0.1)), cv=5,
#                   param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
#                               "gamma": np.logspace(-2, 2, 5)})

# t0 = time.time()
#
# print(X[:train_size])
#
# print(y[:train_size])

# svr.fit(X[:train_size], y[:train_size])
# svr_fit = time.time() - t0
# print("SVR complexity and bandwidth selected and model fitted in %.3f s"
#       % svr_fit)
#
# t0 = time.time()
# kr.fit(X[:train_size], y[:train_size])
# kr_fit = time.time() - t0
# print("KRR complexity and bandwidth selected and model fitted in %.3f s"
#       % kr_fit)
# sv_ratio = svr.best_estimator_
# print(sv_ratio)
# sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
# print("Support vector ratio: %.3f" % sv_ratio)
#
# t0 = time.time()
# y_svr = svr.predict(X_plot)
# svr_predict = time.time() - t0
# print("SVR prediction for %d inputs in %.3f s"
#       % (X_plot.shape[0], svr_predict))
#
# t0 = time.time()
# # y_kr = kr.predict(X_plot)
# kr_predict = time.time() - t0
# print("KRR prediction for %d inputs in %.3f s"
#       % (X_plot.shape[0], kr_predict))
#
# # #############################################################################
# # Look at the results
# sv_ind = svr.best_estimator_.support_
# plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
#             zorder=2, edgecolors=(0, 0, 0))
# plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1,
#             edgecolors=(0, 0, 0))
# plt.plot(X_plot, y_svr, c='r',
#          label='SVR (fit: %.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
# # plt.plot(X_plot, y_kr, c='g',
# #          label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('SVR versus Kernel Ridge')
# plt.legend()
#
# # Visualize training and prediction time
# plt.figure()
#
# # Generate sample data
# X = 5 * rng.rand(10000, 1)
# y = np.sin(X).ravel()
# # y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
# sizes = np.logspace(1, 4, 7).astype(np.int)
# for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,
#                                            gamma=10),
#                         "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
#     train_time = []
#     test_time = []
#     for train_test_size in sizes:
#         t0 = time.time()
#         estimator.fit(X[:train_test_size], y[:train_test_size])
#         train_time.append(time.time() - t0)
#
#         t0 = time.time()
#         estimator.predict(X_plot[:1000])
#         test_time.append(time.time() - t0)
#
#     plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g",
#              label="%s (train)" % name)
#     plt.plot(sizes, test_time, 'o--', color="r" if name == "SVR" else "g",
#              label="%s (test)" % name)
#
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Train size")
# plt.ylabel("Time (seconds)")
# plt.title('Execution Time')
# plt.legend(loc="best")
#
# # Visualize learning curves
# plt.figure()
#
# svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
# kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
# train_sizes, train_scores_svr, test_scores_svr = \
#     learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
#                    scoring="neg_mean_squared_error", cv=10)
# train_sizes_abs, train_scores_kr, test_scores_kr = \
#     learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
#                    scoring="neg_mean_squared_error", cv=10)
#
# plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
#          label="SVR")
# plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g",
#          label="KRR")
# plt.xlabel("Train size")
# plt.ylabel("Mean Squared Error")
# plt.title('Learning curves')
# plt.legend(loc="best")
#
# plt.show()

from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

X, y = getTrainData()
# C = [15.0]
gamma = [0.1, 0.2]
#
# pipe_svr = Pipeline([('reg', MultiOutputRegressor(SVR()))])
#
#
# param_grid = dict(kernel=['rbf'], epsilon=[0.1], verbose=[True])
# grid_param_svr = {
#     'reg__estimator__C': [0.1,1,10], 'reg_estimator__gamma': gamma
# }
#
# mlt = MultiOutputRegressor(SVR())
# mlt = GridSearchCV(estimator=pipe_svr, param_grid=grid_param_svr, refit=False, return_train_score=True, cv=3)
# print(mlt.fit(X, y).param_grid)

pipe_svr = Pipeline([('reg', MultiOutputRegressor(SVR()))])

# grid_param_svr = {
#     'reg__estimator__gamma': [0.1,1,10], 'reg__estimator__C':[1.0]}
#
# gs_svr = (GridSearchCV(estimator=pipe_svr,
#                       param_grid=grid_param_svr,
#                       cv=2,
#                       scoring = 'neg_mean_squared_error',
#                       n_jobs = -1))
#
# gs_svr = gs_svr.fit(X,y)
# print(gs_svr.best_score_)


#
SVR_RBF = Pipeline([('reg', MultiOutputRegressor(SVR(verbose=True, kernel='rbf')))])
SVR_POLY = Pipeline([('reg', MultiOutputRegressor(SVR(verbose=True, kernel='poly')))])

grid_param_svr_rbf = {
    'reg__estimator__C': [0.1, 1, 10, 100], 'reg__estimator__gamma': [0.1, 0.2], 'reg__estimator__epsilon': [0.001, 0.01]}
grid_param_svr_poly = {
    'reg__estimator__C': [0.1, 1, 10, 100], 'reg__estimator__degree': [2, 3], 'reg__estimator__gamma': [0.1, 0.2]}

# mlt = MultiOutputRegressor(SVR())
scoring = {'loss': 'neg_mean_squared_error', 'mee': make_scorer(mean_euclidean_error)}
mlt1 = GridSearchCV(estimator=SVR_RBF, param_grid=grid_param_svr_rbf, refit=False, return_train_score=True, cv=3,
                    scoring=scoring
                    )
mlt2 = GridSearchCV(estimator=SVR_POLY, param_grid=grid_param_svr_poly, refit=False, return_train_score=True, cv=3,
                    scoring=scoring)
print_and_saveGrid(mlt1.fit(X, y), 'loss', 'mee', True, 'grid_search_result_SVR_RBF', 'SVR_RBF')
print_and_saveGrid(mlt2.fit(X, y), 'loss', 'mee', True, 'grid_search_result_SVR_POLY', 'SVR_POLY')
