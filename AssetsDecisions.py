"""
This script reads prices of bitcoin and gold from files which contain closing values from 2016 to 2021. Gold can only be
traded on weekdays for 1% commission per transaction while bitcoin can be traded on any day for 2% commission per
transaction. Nothing is done for first 29 bitcoin and 21 gold values. For the 30th bitcoin value, an RNN model is
trained to predict the next bitcoin value using last 28 values and, for 22nd gold value, another RNN model is trained to
predict next gold value using last 20 gold values. These bitcoin and gold models are retrained after every 29 and 21
values respectively. Every day, I get predicted closing values for today and tomorrow from my models. In this case,
tomorrow's value is predicted using actual last 27 values and today's predicted value using the same model (a new model
is never trained to predict tomorrow's value). I then predict profit if 102%, in case of bitcoin, and 101%, in case of
gold, of today's predicted value is less than 98%, in case of bitcoin, and 99%, in case of gold, of tomorrow's predicted
value and the same relation holds between the value from 7, in case of bitcoin, and 5, in case of gold, trading days ago
and yesterday's respective value. I predict profits in terms of percentage with respect to today's predicted values. If
I have bitcoin and gold's predicted profit is greater than bitcoin, I sell bitcoin. Similarly, if I have gold and
bitcoin's predicted profit is greater than gold, I sell gold. I also sell if 98%, in case of bitcoin, and 99%, in case
of gold, of today's predicted value is greater than 102%, in case of bitcoin, and 101%, in case of gold, of tomorrow's
predicted value and the same relation holds between the value from 7, in case of bitcoin, and 5, in case of gold,
trading days ago and yesterday's respective value. I then buy the asset with greater predicted profit if I have cash. In
case predicted profit percentages turn out to be equal between gold and bitcoin, I randomly buy one of them. All
transactions are carried out at the actual values at end of the day.
"""

import random
import tensorflow as tf
import numpy as np
from keras_tuner import RandomSearch
import matplotlib.pyplot as plt


def readCSV(filePath):
    """
    Reads dates and corresponding values from a csv file and stores them in a 2-d array
    :param filePath: path of the csv file containing the data
    :return: a 2-d array containing the read data
    """
    with open(filePath) as file:
        file.readline()
        toReturn = file.readlines()

    toRemoveIndices = []

    for i in range(len(toReturn)):
        toReturn[i] = toReturn[i].split(',')
        if toReturn[i][1].strip() == "":
            toRemoveIndices.append(i)
            continue
        toReturn[i][1] = float(toReturn[i][1])

    for i in range(len(toRemoveIndices)):
        del toReturn[toRemoveIndices[i]]

        for j in range(i, len(toRemoveIndices)):
            toRemoveIndices[j] -= 1

    return toReturn


def makeWindowed(data, windowSize):
    """
    Transforms the received data into a tensorflow dataset where every entry contains an array of consecutive
    'windowSize' values and another array containing the next 1 value
    :param data: an array containing data to convert
    :param windowSize: the number of values that first array in each entry should contain
    :return: the corresponding windowed tensorflow dataset
    """
    return tf.data.Dataset.from_tensor_slices(tf.expand_dims(data, axis=-1)). \
        window(windowSize + 1, shift=1, drop_remainder=True).flat_map(lambda window: window.batch(windowSize + 1)) \
        .map(lambda window: (window[:-1], window[-1])).shuffle(len(data)).batch(windowSize).prefetch(windowSize)


def buildBTCModel(hp):
    """
    Builds the structure of a bitcoin prediction RNN model with 2 LSTM layers
    :param hp: usually used for hyperparameter variations; not used in this implementation
    :return: the built structure
    """
    modelStruct = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(100, input_shape=[None, 1], return_sequences=True),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(1)
    ])

    modelStruct.compile(loss="mse", optimizer="adam")

    return modelStruct


def buildGoldModel(hp):
    """
    Builds the structure of a gold prediction RNN model with 2 LSTM layers
    :param hp: usually used for hyperparameter variations; not used in this implementation
    :return: the built structure
    """
    modelStruct = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(50, input_shape=[None, 1], return_sequences=True),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])

    modelStruct.compile(loss="mse", optimizer="adam")

    return modelStruct


def predictNextVal(item, todayPredict=-1):
    """
    Predicts the next value for 'item'
    :param item: 'bitcoin', 'btc' or 'gold' in any case (lower, upper or mixed); the item/asset for which the next value
    is to be predicted
    :param todayPredict: if not -1, this value is used along with last 27, in case of bitcoin, and last 19, in case of
    gold, actual values to predict the next value using the existing model
    :return: next predicted value
    """
    global bTCModel
    global goldModel

    if item.lower() == "btc" or item.lower() == "bitcoin":
        originalValArr = bTCVals
        index = bTCIndex
        windowSize = 28
    elif item.lower() == "gold":
        originalValArr = goldVals
        index = goldIndex
        windowSize = 20

    data = []

    for entry in originalValArr[:(index + 1)]:
        data.append(entry[1])

    if todayPredict != -1:
        data.append(todayPredict)
        index += 1

    data = np.asarray(data)

    mean = data.mean(axis=0)
    stdDeviation = data.std(axis=0)

    data -= mean
    data /= stdDeviation

    windowedData = makeWindowed(data, windowSize)

    if item.lower() == "btc" or item.lower() == "bitcoin":
        if (index + 1) % (windowSize + 1) == 0 and todayPredict == -1:
            tuner = RandomSearch(buildBTCModel, objective='loss', directory='Models',
                                 project_name=('BTCUptoIndex' + str(bTCIndex)), max_trials=1)

            tuner.search(windowedData, epochs=300)

            bTCModel = tuner.get_best_models(1)[0]

        model = bTCModel
    elif item.lower() == "gold":
        if (index + 1) % (windowSize + 1) == 0 and todayPredict == -1:
            tuner = RandomSearch(buildGoldModel, objective='loss', directory='Models',
                                 project_name=('GoldUptoIndex' + str(goldIndex)), max_trials=1)

            tuner.search(windowedData, epochs=150)

            goldModel = tuner.get_best_models(1)[0]

        model = goldModel

    return model.predict(tf.data.Dataset
                         .from_tensor_slices(tf.expand_dims(data[(index - windowSize):(index + 1)], axis=-1))
                         .window(windowSize, shift=1, drop_remainder=True)
                         .flat_map(lambda window: window.batch(windowSize))
                         .batch(1), verbose=0)[0][0] * stdDeviation + mean


def buy(item):
    """
    Buys 'item' using all available cash
    :param item: 'bitcoin', 'btc' or 'gold' in any case (lower, upper or mixed); the item/asset to buy
    :return: nothing
    """
    global todayActions

    if item.lower() == "btc" or item.lower() == "bitcoin":
        state[2] += state[0] / (bTCVals[bTCIndex + 1][1] * 1.02)

        todayActions += "Buy Bitcoin; "
    elif item.lower() == "gold":
        state[1] += state[0] / (goldVals[goldIndex + 1][1] * 1.01)

        todayActions += "Buy Gold; "

    state[0] = 0


def sell(item):
    """
    Sells all of 'item' held
    :param item: 'bitcoin', 'btc' or 'gold' in any case (lower, upper or mixed); the item/asset to sell
    :return: nothing
    """
    global todayActions

    if item.lower() == "btc" or item.lower() == "bitcoin":
        state[0] += bTCVals[bTCIndex + 1][1] * 0.98 * state[2]
        state[2] = 0

        todayActions += "Sell Bitcoin; "
    elif item.lower() == "gold":
        state[0] += goldVals[goldIndex + 1][1] * 0.99 * state[1]
        state[1] = 0

        todayActions += "Sell Gold; "


state = [1000.0, 0.0, 0.0]  # [0]: cash; [1]: gold; [2]: BTC

bTCVals = readCSV("2022_Problem_C_DATA/BCHAIN-MKPRU.csv")  # [][0]: date string; [][1]: value
goldVals = readCSV("2022_Problem_C_DATA/LBMA-GOLD.csv")

bTCIndex = 28
goldIndex = 20

bTCModel = ''
goldModel = ''

bTCActualVals = []
bTCPredictVals = []
bTCIndices = []
goldActualVals = []
goldPredictVals = []
goldIndices = []

allOutFile = open("Output.txt", "w")
bTCPredictFile = open("BTCPredictions.txt", "w")
goldPredictFile = open("GoldPredictions.txt", "w")

while bTCIndex < len(bTCVals) - 1:
    todayActions = ""

    lastWeekBTCVal = bTCVals[bTCIndex - 7][1]
    lastWeekGoldVal = goldVals[goldIndex - 5][1]

    yestBTCVal = bTCVals[bTCIndex][1]
    yestGoldVal = goldVals[goldIndex][1]

    todayPredictBTCVal = predictNextVal("BTC")
    todayPredictGoldVal = predictNextVal("gold")

    tomPredictBTCVal = predictNextVal("BTC", todayPredictBTCVal)
    tomPredictGoldVal = predictNextVal("gold", todayPredictGoldVal)

    bTCPredictProfit = -1
    goldPredictProfit = -1

    if todayPredictBTCVal * 1.02 < tomPredictBTCVal * 0.98 and lastWeekBTCVal * 1.02 < yestBTCVal * 0.98:
        bTCPredictProfit = (tomPredictBTCVal * 0.98 - todayPredictBTCVal * 1.02) / (todayPredictBTCVal * 1.02)

    if goldVals[goldIndex][0] == bTCVals[bTCIndex][0] and todayPredictGoldVal * 1.01 < tomPredictGoldVal * 0.99 \
            and lastWeekGoldVal * 1.01 < yestGoldVal * 0.99:
        goldPredictProfit = (tomPredictGoldVal * 0.99 - todayPredictGoldVal * 1.01) / (todayPredictGoldVal * 1.01)

    if state[2] > 0 and (goldPredictProfit > bTCPredictProfit or (todayPredictBTCVal * 0.98 > tomPredictBTCVal * 1.02
                                                                  and lastWeekBTCVal * 0.98 > yestBTCVal * 1.02)):
        sell("BTC")

    if state[1] > 0 and goldVals[goldIndex][0] == bTCVals[bTCIndex][0] and \
            (bTCPredictProfit > goldPredictProfit or (todayPredictGoldVal * 0.99 > tomPredictGoldVal * 1.01 and
                                                      lastWeekGoldVal * 0.99 > yestGoldVal * 1.01)):
        sell("gold")

    if state[0] > 0:
        if bTCPredictProfit > goldPredictProfit:
            buy("BTC")
        elif goldPredictProfit > bTCPredictProfit:
            buy("gold")
        elif bTCPredictProfit > 0 and goldPredictProfit > 0:
            if random.randint(0, 1) == 0:
                buy("BTC")
            else:
                buy("gold")

    output = "{}: todayPredictGoldVal: {:.2f}; tomPredictGoldVal: {:.2f}; todayActualGoldVal: {:.2f}; " \
             "todayPredictBTCVal: {:.2f}; tomPredictBTCVal: {:.2f}; todayActualBTCVal: {:.2f}; {}" \
             "[{:.2f}, {:.2f}, {:.2f}]".format(bTCVals[bTCIndex + 1][0], todayPredictGoldVal, tomPredictGoldVal,
                                               goldVals[goldIndex + 1][1], todayPredictBTCVal, tomPredictBTCVal,
                                               bTCVals[bTCIndex + 1][1], todayActions, state[0], state[1], state[2])

    print(output)
    allOutFile.write(output + "\n")

    bTCIndex += 1

    if bTCIndex < len(bTCVals):
        bTCPredictFile.write("{},{:.2f}\n".format(bTCVals[bTCIndex][0], todayPredictBTCVal))

        bTCActualVals.append(bTCVals[bTCIndex][1])
        bTCPredictVals.append(todayPredictBTCVal)
        bTCIndices.append(bTCIndex)

    if goldIndex + 1 < len(goldVals) and bTCVals[bTCIndex][0] == goldVals[goldIndex + 1][0]:
        goldIndex += 1

        goldPredictFile.write("{},{:.2f}\n".format(goldVals[goldIndex][0], todayPredictGoldVal))

        goldActualVals.append(goldVals[goldIndex][1])
        goldPredictVals.append(todayPredictGoldVal)
        goldIndices.append(goldIndex)

bTCPredictFile.close()
goldPredictFile.close()
allOutFile.close()

print("\nTotal assets on {} in cash: {:.2f}".format(bTCVals[bTCIndex][0], state[0] +
                                                    bTCVals[bTCIndex][1] * 0.98 * state[2] +
                                                    goldVals[goldIndex][1] * 0.99 * state[1]))

plt.plot(bTCIndices, bTCActualVals)
plt.plot(bTCIndices, bTCPredictVals)
plt.show()

plt.plot(goldIndices, goldActualVals)
plt.plot(goldIndices, goldPredictVals)
plt.show()
