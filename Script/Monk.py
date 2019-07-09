from sklearn.preprocessing import StandardScaler

from utils import *

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

Model1 = {'layer': [2,3,4], 'hidden_units': [15,20,25,30], 'learning_rate': [0.5,0.1,0.6,0.01, 0.05], 'momentum': [0.9,0.6,0.8], 'decay': [0,0.001, 0.01],
          'batch': [32,64,124], 'activation': ['sigmoid','relu','tanh'], 'lamda': [0.001, 0.01, 0]}

Model2 = {'layer': [1], 'hidden_units': [7], 'learning_rate': [0.02,0.4,0.2,0.6], 'momentum': [0.99,0.8], 'decay': [0],
          'batch': [169], 'activation': ['sigmoid','tanh'], 'lamda': [0,0.001]}

Model3 ={'layer': [1, 2,3], 'hidden_units': [15,20,25,30], 'learning_rate': [0.5,0.1,0.6,0.01, 0.05], 'momentum': [0.9,0.6,0.8], 'decay': [0,0.0001],
          'batch': [64,122], 'activation': ['sigmoid','tanh'], 'lamda': [0.001, 0.01, 0]}

Models = [Model1, Model2, Model3]

epochs = [200, 400, 600, 800]


def main():
    XTS1, YTS1 = getTrainData(Monk1TS, '2:8', '1:2', ' ')
    XTR1, YTR1 = getTrainData(Monk1TR, '2:8', '1:2', ' ')

    XTS2, YTS2 = getTrainData(Monk2TS, '2:8', '1:2', ' ')
    XTR2, YTR2 = getTrainData(Monk2TR, '2:8', '1:2', ' ')


    saler = StandardScaler()
    XTR2 = saler.fit_transform(XTR2)

    XTS3, YTS3 = getTrainData(Monk3TS, '2:8', '1:2', ' ')
    XTR3, YTR3 = getTrainData(Monk3TR, '2:8', '1:2', ' ')

    # Model1['batch'] = int(XTR1.shape[0]/10)
    # Model2['batch'] = XTR2.shape[0]
    # Model3['batch'] = XTR3.shape[0]

    # model1 = create_model(XTR1.shape[1], YTR1.shape[1], Model1['learning_rate'], Model1['hidden_units'],
    #                       Model1['layer'], Model1['momentum'], Model1['decay'], Model1['activation'], Model1['lamda'])
    # model2 = create_model(XTR2.shape[1], YTR2.shape[1], Model2['learning_rate'], Model2['hidden_units'],
    #                       Model1['layer'], Model1['momentum'], Model1['decay'], Model1['activation'],Model1['lamda'])
    # model3 = create_model(XTR3.shape[1], YTR3.shape[1], Model3['learning_rate'], Model3['hidden_units'],
    #                       Model3['layer'], Model3['decay'] , Model1['lamda'])

    param_grid1 = dict(learn_rate=Model1['learning_rate'], units=Model1['hidden_units'], level=Model1['layer'],
                       batch_size=Model1['batch'],
                       activation=Model1['activation'], momentum=Model1['momentum'], lamda=Model1['lamda'],
                       decay=Model1['decay'], input_dim=[6], output_dim=[1], epochs=epochs)
    param_grid2 = dict(learn_rate=Model2['learning_rate'], units=Model2['hidden_units'], level=Model2['layer'],
                       batch_size=Model2['batch'],
                       activation=Model2['activation'], momentum=Model2['momentum'], lamda=Model2['lamda'],
                       decay=Model2['decay'], input_dim=[6], output_dim=[1], epochs=epochs)
    # param_grid3 = dict(learn_rate=Model3['learning_rate'], units=Model3['hidden_units'], level=Model3['layer'],
    #                    batch_size=Model3['batch'],
    #                    activation=Model3['activation'], momentum=Model3['momentum'], lamda=Model3['lamda'],
    #                    decay=Model3['decay'], input_dim=[6], output_dim=[1], epochs=epochs)
    #
    model1 = KerasRegressor(build_fn=create_model, verbose=2)
    model2 = KerasRegressor(build_fn=create_model, verbose=0)
    # model3 = KerasRegressor(build_fn=create_model, verbose=0)

    #grid1 = GridSearchCV(estimator=model1, param_grid=param_grid1, n_jobs=-1, refit=False, return_train_score=True,
     #                    cv=3, scoring=scoring)
    grid2 = GridSearchCV(estimator=model2, param_grid=param_grid2, n_jobs=-1, refit=False, return_train_score=True,
                         cv=3, scoring=scoring)
    # grid3 = GridSearchCV(estimator=model3, param_grid=param_grid3, n_jobs=-1, refit=False, return_train_score=True,
    #                      cv=3, scoring=scoring)

    #print_and_saveGrid(grid1.fit(XTR1, YTR1), True, False, "monk1", 'Monk')
    print("Parte1")
    print_and_saveGrid(grid2.fit(XTR2, YTR2), True, False, "monk2", 'Monk')
    # print(("Part2"))
    # print_and_saveGrid(grid3.fit(XTR3, YTR3), True, False, "monk3", 'Monk')
    # history1 = model1.fit(XTR1, YTR1, shuffle=True, epochs=Model1['epochs'], verbose=2, batch_size=XTR1.shape[0],
    #                       validation_data=[XTS1, YTS1])
    # validation_split=0.2)
    # history2 = model2.fit(XTR2, YTR2, shuffle=True, epochs=Model2['epochs'], verbose=2, batch_size=XTR2.shape[0],
    #                       validation_data=[XTS2,YTS2])
    # history3 = model3.fit(XTR3, YTR3, shuffle=True, epochs=Model3['epochs'], verbose=2, batch_size=XTR3.shape[0],
    #                       validation_split=0.2)

    # print("Min loss:", min(history1.history['val_loss']))
    # print("Min loss:", min(history2.history['val_loss']))
    # print("Min loss:", min(history3.history['val_loss']))

    # printPlots(history1, True, 'Monk1')
    # printPlots(history2, True, 'Monk2')
    # printPlots(history3, True, 'Monk3')

    # printCSV([history1, history2, history3])


def printPlots(history, save, name=None):
    fig = plt.figure()
    plt.plot(history.history['acc'], label="TR Acc")
    plt.plot(history.history['val_acc'], label="VL Acc", linestyle='dashed')
    plt.legend(loc='upper right')
    plt.show()
    if save:
        directory = "../Image/"
        t = strftime("%H_%M")
        file = name + "_Acc_" + t + ".png"
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(directory + file)

    fig = plt.figure()
    plt.plot(history.history['loss'], label="TR Loss")
    plt.plot(history.history['val_loss'], label="VL Loss", linestyle='dashed')
    plt.legend(loc='upper right')
    plt.show()
    if save:
        directory = "../Image/"
        t = strftime("%H_%M")
        file = name + "_Loss" + t + ".png"
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(directory + file)


def printCSV(monkS):
    results_records = {'Task': [], 'Model': [], 'MSE (TR/TS)': [], 'Accuracy (TR/TS)': []}
    for i in range(len(monkS)):
        results_records['Task'].append('Monk' + str(i + 1))
        results_records['Model'].append(getModel(Models[i]))
        results_records['MSE (TR/TS)'].append(
            '%.3f' % monkS[i].history['loss'][Models[i]['epochs'] - 1] + " / " + '%.3f' % monkS[i].history['val_loss'][
                Models[i]['epochs'] - 1])
        results_records['Accuracy (TR/TS)'].append(
            '%.3f' % monkS[i].history['acc'][Models[i]['epochs'] - 1] + " / " + '%.3f' % monkS[i].history['val_acc'][
                Models[i]['epochs'] - 1])
    saveOnCSV(results_records, 'MonkS_')


def getModel(model):
    string = ''
    for key, val in model.items():
        string = string + key + ":" + str(val) + " "
    return string


if __name__ == '__main__':
    main()
