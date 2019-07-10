features_train = RealFeatures(f_feats_train)
features_test = RealFeatures(f_feats_test)
labels_train = RegressionLabels(f_labels_train)
labels_test = RegressionLabels(f_labels_test)

mean_rule = MeanRule()

rand_forest = RandomForest(features_train, labels_train, 5)
rand_forest.set_combination_rule(mean_rule)


rand_forest.train()
labels_predict = rand_forest.apply_regression(features_test)

mse = MeanSquaredError()
oob = rand_forest.get_oob_error(mse)
mserror = mse.evaluate(labels_predict, labels_test)

'MA CHE E\' STA MERDA'