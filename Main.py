import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta
import tensorflow as tf
import matplotlib.pyplot as plt

import random
import pickle
import joblib


# creating chunk for data set
def dataKFold(foldNumber):
    data = pd.read_csv("data/SPY_30_AFTER2010.txt", date_parser=True, header=None) #file read
    dataToScale = data.copy()  #copy the file for minmax scale
    scaler = MinMaxScaler()
    dataToScale.pop(0)  #remove date for scaling
    scaler.fit(dataToScale)
    pickle.dump(scaler,  open("model/scalerfile", 'wb')) #saving scaled matric

    length = len(data)
    #print(length)
    chunkSize = int(length/foldNumber)
    dataRange = []
    rowNumber = []
    for x in range(1, foldNumber):
        rowNumber.append(chunkSize*x)
    for index in rowNumber:
        dataRange.append(data.iloc[index][0])
    dataFold = []
    prevDate = data.iloc[0][0]
    for d in dataRange:
        dataFold.append(data[(data[0] < d) & (data[0] >= prevDate)])
        prevDate = d

    dataFold.append(data[data[0] >= dataRange[foldNumber - 2]])
    foldNumber = 0
    for fold in dataFold:
        fold.to_csv("fold30min/fold_" + str(foldNumber) + ".txt", header=None, index=False)
        foldNumber += 1

#get the datelist of a chunk
def getDateList(dataFrame):
    dataFrame[0] = pd.to_datetime(dataFrame[0])
    return dataFrame[0].dt.strftime('%Y-%m-%d').unique()


def myDataFilter(row):
    if (row[0].hour >= 9) & (row[0].hour <= 17):
        return True
    return False

#parseInput for a specific date
def parseData(dataFrame, dateStart, dateEnd, scale):

    dateStart = dateStart + ' 08:00:00'
    dateE = dateEnd + " 15:00:00"
    dateOut = dateEnd + " 17:00:00"
    dateOutEnd = dateEnd + " 23:59:59"
    x = []
    dFInRange = dataFrame[(dataFrame[0] >= dateStart) & (dataFrame[0] <= dateE)]
    dFInRange = dFInRange[dFInRange.apply(myDataFilter, axis=1)]
    outData = dataFrame[(dataFrame[0] >= dateOut) & (dateOut[0] <= dateOutEnd)]
    if len(outData) < 1:
        return [], []
    for key, item in dFInRange.iterrows():
        #if item[0]
        x.append([item[1] * scale.scale_[0], item[2] * scale.scale_[1], item[3] * scale.scale_[2], item[4] * scale.scale_[3], item[5] * scale.scale_[4]])
    return x, [outData.iloc[0][1] * scale.scale_[0]]


#process the folded data
def processData(foldCount, selectedFoldIndex = None):
    foldData = []
    scale = pickle.load(open("model/scalerfile", 'rb'))
    for i in range(0, foldCount):
        foldData.append(pd.read_csv("fold30min/fold_" + str(i) + ".txt", date_parser=True, header=None))
    #print(len(foldData))
    if selectedFoldIndex is None:
        testIndex = random.randint(0, len(foldData)-1) #randomly selected a chunk for test set
    else:
        testIndex = selectedFoldIndex
    testSetData = foldData[testIndex]
    foldData.pop(testIndex)
    trainingData = foldData[0]
    for x in range(1, len(foldData)):
        trainingData = pd.concat([trainingData, foldData[x]])
    #print(len(testSetData))
    #print(trainingData.head)
    trainDates = getDateList(trainingData)
    testDates = getDateList(testSetData)
    trainX = []
    trainY = []
    testX = []
    testY = []
    l1 = len(trainDates)
    for i in range(10, l1):
        x, y = parseData(trainingData, trainDates[i-10], trainDates[i], scale)
        #print(len(x))
        if len(x) == 193:
            trainX.append(x)
            trainY.append(y)
    l2 = len(testDates)
    for j in range(10, l2):
        x, y = parseData(testSetData, testDates[j-10], testDates[j], scale)
        if len(x) == 193:
            testX.append(x)
            testY.append(y)
    pickle.dump(trainX, open("processedData/trainX_" + str(selectedFoldIndex), 'wb'))
    pickle.dump(trainY, open("processedData/trainY_" + str(selectedFoldIndex), 'wb'))
    pickle.dump(testX, open("processedData/testX_" + str(selectedFoldIndex), 'wb'))
    pickle.dump(testY, open("processedData/testY_" + str(selectedFoldIndex), 'wb'))
    return trainX, trainY, testX, testY

def trainModel(trainX, trainY,  testX, testY, modelPath=None):
    trainX, trainY = np.array(trainX), np.array(trainY)
    scale = pickle.load(open("model/scalerfile", 'rb'))
    print(trainX.shape)
    print(trainY.shape)
    testX, testY = np.array(testX), np.array(testY)
    print(testX.shape)
    print(testY.shape)
    if modelPath is None:
        regression = Sequential()
        regression.add(LSTM(units=200, return_sequences=True, input_shape=(193, 5)))
        regression.add(Dropout(0.2))
        regression.add(LSTM(units=250, return_sequences=True))
        regression.add(Dropout(0.3))
        # regression.add(LSTM(units=300, return_sequences=True))
        # regression.add(Dropout(0.3))
        regression.add(LSTM(units=300, return_sequences=True))
        regression.add(Dropout(0.3))
        regression.add(LSTM(units=350, return_sequences=True))
        regression.add(Dropout(0.4))
        # regression.add(LSTM(units=150, return_sequences=True, input_shape=(None, 5), dropout=0.3))
        # regression.add(LSTM(units=120, return_sequences=True, input_shape=(None, 5), dropout=0.3))
        # regression.add(LSTM(units=160, return_sequences=True, input_shape=(None, 5), dropout=0.3))
        # regression.add(LSTM(units=120, return_sequences=True, input_shape=(None, 5), dropout=0.4))
        # regression.add(LSTM(units=140, return_sequences=True, input_shape=(None, 5), dropout=0.4))
        # regression.add(LSTM(units=140, return_sequences=True, input_shape=(None, 5), dropout=0.5))
        regression.add(LSTM(units=400))
        regression.add(Dropout(0.5))
        regression.add(Dense(units=1))
        print(regression.summary())
        print(regression.summary())
        regression.compile(optimizer='adam', loss='mean_squared_error')
        regression.fit(trainX, trainY, epochs=50, batch_size=16)

    else:
        regression = load_model(modelPath)
    score = regression.evaluate(testX, testY)
    print(score)
    predictY = regression.predict(testX)
    s = 1/scale.scale_[0]
    predictY = predictY * s
    testY = testY * s
    #print(predictY)
    #print(TestY)
    plt.figure()
    plt.plot(testY, color='red', label="Real")
    plt.plot(predictY, color='blue', label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    save_model(regression, "model/saved_Model.hp5", save_format="h5")


def trainFromStoreData(foldIndex, modelPath=None):
    trainX = pickle.load(open("processedData/trainX_" + str(foldIndex), 'rb'))
    trainY = pickle.load(open("processedData/trainY_" + str(foldIndex), 'rb'))
    testX = pickle.load(open("processedData/testX_" + str(foldIndex), 'rb'))
    testY = pickle.load(open("processedData/testY_" + str(foldIndex), 'rb'))
    trainX, trainY = np.array(trainX), np.array(trainY)
    scale = pickle.load(open("model/scalerfile", 'rb'))
    print(trainX.shape)
    print(trainY.shape)
    testX, testY = np.array(testX), np.array(testY)
    print(testX.shape)
    print(testY.shape)
    if modelPath is None:
        regression = Sequential()
        regression.add(LSTM(units=200, return_sequences=True, input_shape=(193, 5)))
        regression.add(Dropout(0.2))
        regression.add(LSTM(units=250, return_sequences=True))
        regression.add(Dropout(0.3))
        regression.add(LSTM(units=300, return_sequences=True))
        regression.add(Dropout(0.3))
        regression.add(LSTM(units=350, return_sequences=True))
        regression.add(Dropout(0.4))
        regression.add(LSTM(units=400))
        regression.add(Dropout(0.5))
        regression.add(Dense(units=1))
        print(regression.summary())
        regression.compile(optimizer='adam', loss='mean_squared_error')
        regression.fit(trainX, trainY, epochs=50, batch_size=16)

    else:
        regression = load_model(modelPath)
    score = regression.evaluate(testX, testY)
    print(score)
    predictY = regression.predict(testX)
    s = 1 / scale.scale_[0]
    predictY = predictY * s
    predictY = predictY
    testY = testY * s
    # print(predictY)
    # print(TestY)
    plt.figure()
    plt.plot(testY, color='red', label="Real")
    plt.plot(predictY, color='blue', label="Predicted")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
    save_model(regression, "model/saved_Model.hp5", save_format="h5")


if __name__ == "__main__":
    foldNumber = 5
    # dataKFold(foldNumber)
    # trainX, trainY, testX, TestY = processData(foldNumber, 3)
    # trainModel(trainX, trainY, testX, TestY)
    trainFromStoreData(4)

