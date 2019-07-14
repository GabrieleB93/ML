from sklearn.preprocessing import StandardScaler, OneHotEncoder
import category_encoders

from utils import *

Model1 = {'layer': 1, 'hidden_units': 7, 'learning_rate': 0.6, 'momentum': 0.9, 'decay': 0,
          'epochs': 1000, 'activation': 'sigmoid', 'lamda': 0}

Model2 = {'layer': 1, 'hidden_units': 7, 'learning_rate': 0.6, 'momentum': 0.9, 'decay': 0,
          'epochs': 500, 'activation': 'tanh', 'lamda': 0}

Model3 = {'layer': 1, 'hidden_units': 7, 'learning_rate': 0.6, 'momentum': 0.6, 'lamda': 0.003,
          'epochs': 1200, 'activation': 'tanh', 'decay': 0}

Models = [Model1, Model2, Model3]


def main():
    XTS1, YTS1 = getTrainData(Monk1TS, '2:8', '1:2', ' ')
    XTR1, YTR1 = getTrainData(Monk1TR, '2:8', '1:2', ' ')

    XTS2, YTS2 = getTrainData(Monk2TS, '2:8', '1:2', ' ')
    XTR2, YTR2 = getTrainData(Monk2TR, '2:8', '1:2', ' ')

    XTS3, YTS3 = getTrainData(Monk3TS, '2:8', '1:2', ' ')
    XTR3, YTR3 = getTrainData(Monk3TR, '2:8', '1:2', ' ')

    scaler = StandardScaler()

    XTR3 = scaler.fit_transform(XTR3)
    XTS3 = scaler.fit_transform(XTS3)

    scaler.fit(XTR2)
    XTR2 = scaler.transform(XTR2)
    XTR2 = one_of_k(XTR2)
    XTS2 = one_of_k(XTS2)

    XTR1 = scaler.fit_transform(XTR1)
    XTS1 = scaler.fit_transform(XTS1)

    Model1['batch'] = XTR1.shape[0]
    Model2['batch'] = XTR2.shape[0]
    Model3['batch'] = XTR3.shape[0]

    model1 = create_model(XTR1.shape[1], YTR1.shape[1], Model1['learning_rate'], Model1['hidden_units'],
                          Model1['layer'], Model1['momentum'], Model1['decay'], Model1['activation'], Model1['lamda'])
    model2 = create_model(XTR2.shape[1], YTR2.shape[1], Model2['learning_rate'], Model2['hidden_units'],
                          Model2['layer'], Model2['momentum'], Model2['decay'], Model2['activation'], Model2['lamda'])
    model3 = create_model(XTR3.shape[1], YTR3.shape[1], Model3['learning_rate'], Model3['hidden_units'],
                          Model3['layer'], Model3['decay'], Model3['lamda'])

    history1 = model1.fit(XTR1, YTR1, shuffle=True, epochs=Model1['epochs'], verbose=2, batch_size=Model1['batch'],
                          validation_data=[XTS1, YTS1])

    history2 = model2.fit(XTR2, YTR2, shuffle=True, epochs=Model2['epochs'], verbose=2, batch_size=Model2['batch'],
                          validation_data=[XTS2, YTS2])

    history3 = model3.fit(XTR3, YTR3, shuffle=True, epochs=Model3['epochs'], verbose=2, batch_size=XTR3.shape[0],
                          validation_data=[XTS3, YTS3])

    print("Min loss:", min(history1.history['val_loss']))
    print("Min loss:", min(history2.history['val_loss']))
    print("Min loss:", min(history3.history['val_loss']))

    printPlots(history1, True, 'Monk1')
    printPlots(history2, True, 'Monk2')
    printPlots(history3, True, 'Monk3')

    printCSV([history1, history2, history3])


def printPlots(history, save, name=None):
    fig = plt.figure()
    print(history.history.keys())
    plt.plot(history.history['acc'], label="TR Acc")
    plt.plot(history.history['val_acc'], label="VL Acc", linestyle='dashed')
    plt.legend(loc='upper right')
    plt.show()
    if save:
        print("Saving image")
        directory = "../../Image/MONK/"
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
        directory = "../../Image/MONK/"
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
    saveOnCSV(results_records, 'Monk_results', 'MONK')


def getModel(model):
    string = ''
    for key, val in model.items():
        string = string + key + ":" + str(val) + " "
    return string


if __name__ == '__main__':
    main()
