import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
import h5py
import random
import joblib

scaler = MinMaxScaler()
# creating chunk for data set
def dataKFold(foldNumber):
    data = pd.read_csv("data/SPY_1hour.txt", date_parser=True, header=None) #file read
    dataToScale = data.copy()  #copy the file for minmax scale
    dataToScale.pop(0)  #remove date for scaling
    scaler.fit(dataToScale)
    joblib.dump(scaler, "model\scalerfile") #saving scaled matric
    length = len(data)
    print(length)
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
    return dataFold

#get the datelist of a chunk
def getDateList(dataFrame):
    dataFrame[0] = pd.to_datetime(dataFrame[0])
    return dataFrame[0].dt.strftime('%Y-%m-%d').unique()

#parseInput for a specific date
def parseData(dataFrame, date):
    dateStart = date + ' 08:00:00'
    dateEnd = date + " 15:59:59"
    dateOut = date + " 17:00:00"
    dateOutEnd = date + " 23:59:59"
    x = []
    dFInRange = dataFrame[(dataFrame[0] >= dateStart) & (dataFrame[0] <= dateEnd)]
    outData = dataFrame[(dataFrame[0] >= dateOut) & (dateOut[0] <= dateOutEnd)]
    if len(outData) < 1:
        return [], []
    for key, item in dFInRange.iterrows():
        x.append((scaler.transform(np.array([item[1], item[2], item[3], item[4], item[5]]).reshape(1, -1))[0]).tolist())
    return x, scaler.transform(np.array([outData.iloc[0][1], outData.iloc[0][2], outData.iloc[0][3], outData.iloc[0][4], outData.iloc[0][5]]).reshape(1, -1))[0][0]


#process the folded data
def processData(foldData):
    print(len(foldData))
    testIndex = random.randint(0, len(foldData)-1) #randomly selected a chunk for test set
    testSetData = foldData[testIndex]
    foldData.pop(testIndex)
    trainingData = foldData[0]
    for x in range(1, len(foldData)):
        trainingData = pd.concat([trainingData, foldData[x]])
    print(len(testSetData))
    print(len(trainingData))
    trainDates = getDateList(trainingData)
    testDates = getDateList(testSetData)
    trainX = []
    trainY = []
    testX = []
    testY = []
    for d in trainDates:
        x, y = parseData(trainingData, d)
        if len(x) == 8:
            trainX.append(x)
            trainY.append(y)

    for d in testDates:
        x, y = parseData(testSetData, d)
        if len(x) == 8:
            testX.append(x)
            testY.append(y)

    return trainX, trainY, testX, testY

def trainModel(trainX, trainY,  testX, testY, modelPath=None):
    trainX, trainY = np.array(trainX), np.array(trainY)
    print(trainX.shape)
    print(trainY.shape)
    testX, testY = np.array(testX), np.array(testY)
    print(testX.shape)
    print(testY.shape)
    if modelPath == None:
        regression = Sequential()
        regression.add(LSTM(units=50, return_sequences=True, input_shape=(None, 5), dropout=0.1))
        regression.add(LSTM(units=80, return_sequences=True, input_shape=(None, 5), dropout=0.2))
        regression.add(LSTM(units=120, return_sequences=True, input_shape=(None, 5), dropout=0.3))
        regression.add(LSTM(units=160, input_shape=(None, 5), dropout=0.4))
        regression.add(Dense(units=1))
        print(regression.summary())
        regression.compile(optimizer='adam', loss='mean_squared_error')
        regression.fit(trainX, trainY, epochs=50, batch_size=16)
    else:
        regression = load_model(modelPath)
    score = regression.evaluate(testX, testY)
    print(score)
    predictY = regression.predict(testX)
    s = 1/scaler.scale_[0]
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
    save_model(regression, "model/saved_model_15min_1day_worked_beter.hp5", save_format="h5")


if __name__ == "__main__":
    foldNumber = 5
    foldData = dataKFold(foldNumber)
    trainX, trainY, testX, TestY = processData(foldData)
    trainModel(trainX, trainY, testX, TestY)