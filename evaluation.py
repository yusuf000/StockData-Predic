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

    for key, item in dFInRange.iterrows():
        #if item[0]
        x.append([item[1] * scale.scale_[0], item[2] * scale.scale_[1], item[3] * scale.scale_[2], item[4] * scale.scale_[3], item[5] * scale.scale_[4]])
    return x, dateEnd


def evaluate():
    scale = pickle.load(open("model/scalerfile", 'rb'))
    data = pd.read_csv("evaluationData/evaluate.txt", date_parser=True, header=None)  #file read
    dateList = getDateList(data)
    if len(dateList) < 11:
        print("Not enough date data for evaluate")
        return
    evaluateX = []
    evaluateDate = []
    l1 = len(dateList)
    for i in range(10, l1):
        x, z = parseData(data, dateList[i - 10], dateList[i], scale)
        # print(len(x))
        if len(x) == 193:
            evaluateX.append(x)
            evaluateDate.append(z)
    if len(evaluateX) < 1:
        print("Note enough valid date to create input data")
        return
    regression = load_model("model/saved_Model.hp5")
    predictY = regression.predict(evaluateX)
    s = 1 / scale.scale_[0]
    predictY = predictY * s
    for i in range(0, len(predictY)):
        print(evaluateDate[i] + " " + str(predictY[i][0]))


if __name__ == "__main__":
    evaluate()
