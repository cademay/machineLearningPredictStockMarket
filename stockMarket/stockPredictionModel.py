
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime

# stocks = ['AAPL', 'GOOGL', 'TTS', 'CLSN', 'CSCO', 'ATHX']
# stocks = ['TTS', 'CLSN', 'ATHX']
saveResults = True
daysBack = 3



#directoryName = 'ConfidenceModelOutputs'
#if not os.path.exists(directoryName):
#    os.makedirs(directoryName)


dataDirectory = 'datasets'

#stocks = os.listdir(dataDirectory)
#stocks = stocks[1:]
#stocks = removeCSVsuffix(stocks)


# Part 2 - Building the RNN
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from keras.layers.core import Activation


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Merge
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.layers.advanced_activations import *
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D
from keras.layers.recurrent import LSTM, GRU
from keras import regularizers
#date = getDatePath()

currentStockID = 'CRM'





X_train, y_train, trainingData, real_data, sc = preprocessData(currentStockID, daysBack=daysBack)

model = createModel()
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

# predict
predicted_stock_price = getPredictions(model, sc)

# real data
real_stock_price = invertRealPrices(real_data)

graphPath = ""
displayAndSaveResults(False, real_stock_price, predicted_stock_price, currentStockID, graphPath)




def createModel():
#    model = Sequential()
#    model.add(LSTM(units = 8,input_shape = (None, 4),  recurrent_dropout = 0.2))
#    model.add(Dense(units = 1))
#    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
#    return model
    
    model = Sequential()
    model.add(GRU(units = 128, batch_input_shape=(32,4,4), return_sequences = True, stateful = True))
    model.add(Dropout(0.25))
    model.add(GRU(units = 128, return_sequences = True, stateful = True))
    model.add(Dropout(0.25))
    model.add(GRU(units=128, stateful = True))
    model.add(Dropout(0.25))
    #model.add(Dense(units=64))
   # model.add(LeakyReLU())
#    model.add(Dense(units=64))
#    model.add(LeakyReLU())

    model.add(Dense(units = 1))
    
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    return model


def checkForNullData(currentStockID):
    dataFileName = dataDirectory + '/' + currentStockID + '.csv'
    dataset_train = pd.read_csv(dataFileName)
    training_set = dataset_train.iloc[:,1:5].values
    null = np.any(training_set[:, 0] == 'null')
    
    return null

def checkDataLength(currentStockID):
    dataFileName = dataDirectory + '/' + currentStockID + '.csv'
    dataset_train = pd.read_csv(dataFileName)
    training_set = dataset_train.iloc[:,1:5].values
    
    
    return len(training_set) > 80

def getPredictions(model, sc):
    inputs = []
    for i in range(daysBack, len(real_data) - 1):
        inputs.append(real_data[i-daysBack:i + 1, 0:])
        
    inputs = np.array(inputs)
    inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 4))
    
    predicted_stock_price = model.predict(inputs)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    return predicted_stock_price

def invertRealPrices(real_data):
    real_stock_price = sc.inverse_transform(real_data)
    real_stock_price = real_stock_price[daysBack + 1:, 0:1]
    
    return real_stock_price


def getDatePath():

    currentDateAndTime = datetime.datetime.now()
    date = currentDateAndTime.isoformat()
    
    microsecondIndex = date.find('.')
    date = date[0:microsecondIndex]
    date = date.replace('T', '---')
    
    return date


def createDirectory(path):
    if not os.path.exists(path):
        os.makedirs(path)

    
    
def preprocessData(currentStockID, daysBack):
    
    dataFileName = 'datasets/' + currentStockID + '.csv'
    dataset_train = pd.read_csv(dataFileName)
    
    
    training_set = dataset_train.iloc[:,1:5].values
    scaleData = dataset_train.iloc[:,1:5].values.flatten()
    scaleData = np.reshape(scaleData, (len(scaleData),1))
    

    
    trainBound = len(training_set) - 32
    
    """
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    """
    
        
    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    sc.fit(scaleData)
    training_set_scaled = sc.transform(training_set)
    
    
    real_data = training_set_scaled[trainBound:]
    training_set_scaled = training_set_scaled[0:trainBound]
    
    
    # Creating a data structure with 20 timesteps and t+1 output
    X_train = []
    y_train = []
    for i in range(5, trainBound - 1):
        X_train.append(training_set_scaled[i-3:i + 1, 0:])
        y_train.append(training_set_scaled[i + 1, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 4))
    
    return (X_train, y_train, training_set, real_data, sc)

def removeCSVsuffix(stocks):
    
    for i in range(len(stocks)):
        periodIndex = stocks[i].find('.')
        stocks[i] = (stocks[i])[0:periodIndex]
    return stocks



def test1Predicted():
    oneMeansUpZeroMeansDown1 = []
    count = 0
    for i in range(1, len(predicted_stock_price)):
        if predicted_stock_price[i] > (predicted_stock_price[i - 1] + (predicted_stock_price[i - 1] *0.015)):
            count += 1
            # print(str(predicted_stock_price[i]) + '   ' +  str((real_stock_price[i - 1])))
            oneMeansUpZeroMeansDown1.append(1)
        else:
            
            oneMeansUpZeroMeansDown1.append(0)
    
    print('count ' + str(count))
    
    #for i in range(1, len(predicted_stock_price)):
        # print(str(predicted_stock_price[i]) + '   ' +  str((real_stock_price[i])))
    
    oneMeansUpZeroMeansDown2 = []
    
    for i in range(1, len(real_stock_price)):
        if real_stock_price[i] > (real_stock_price[i - 1]  + (real_stock_price[i - 1] *0.015)):
            oneMeansUpZeroMeansDown2.append(1)
        else:
            oneMeansUpZeroMeansDown2.append(0)
    
    preds1 = 0
    for i in range(len(predicted_stock_price) - 1):
        num = oneMeansUpZeroMeansDown1[i]
        if num == 1.:
            preds1 += 1
    
            
    goodCount = 0.
    for i in range(0, len(real_stock_price) - 1):
        realChange = oneMeansUpZeroMeansDown2[i]
        predictedChange = oneMeansUpZeroMeansDown1[i]
        
        if realChange == predictedChange and predictedChange == 1 :
            goodCount += 1.
            
            
    print(goodCount)
    if preds1 != 0:
        percentageRight = goodCount / preds1
    else:
        percentageRight = -1
    print(percentageRight)
    
    return (percentageRight, goodCount)
    
    
def test2Real():
    oneMeansUpZeroMeansDown1 = []
    count = 0
    for i in range(1, len(predicted_stock_price)):
        if predicted_stock_price[i] > (real_stock_price[i - 1] + (real_stock_price[i - 1] *0.015)):
            count += 1
            # print(str(predicted_stock_price[i]) + '   ' +  str((real_stock_price[i - 1])))
            oneMeansUpZeroMeansDown1.append(1)
        else:
            
            oneMeansUpZeroMeansDown1.append(0)
    
    print('count ' + str(count))
    
    #for i in range(1, len(predicted_stock_price)):
        #print(str(predicted_stock_price[i]) + '   ' +  str((real_stock_price[i])))
    
    oneMeansUpZeroMeansDown2 = []
    
    for i in range(1, len(real_stock_price)):
        if real_stock_price[i] > (real_stock_price[i - 1]  + (real_stock_price[i - 1] *0)):
            oneMeansUpZeroMeansDown2.append(1)
        else:
            oneMeansUpZeroMeansDown2.append(0)
    
    preds1 = 0
    for i in range(len(predicted_stock_price) - 1):
        num = oneMeansUpZeroMeansDown1[i]
        if num == 1.:
            preds1 += 1
    
            
    goodCount = 0.
    for i in range(0, len(real_stock_price) - 1):
        realChange = oneMeansUpZeroMeansDown2[i]
        predictedChange = oneMeansUpZeroMeansDown1[i]
        
        if realChange == predictedChange and predictedChange == 1 :
            goodCount += 1.
            
            
    print(goodCount)
    if preds1 != 0:
        percentageRight = goodCount / preds1
    else:
        percentageRight = -1
    print(percentageRight)
    
    return (percentageRight, goodCount)


def saveMetrics(metricsDirectoryPath, currentStockID):
    metricsFilePath = metricsDirectoryPath + '/' + currentStockID + '.csv'

    metricsArray = []
    
    realPercentageRight, realCount = test2Real()
    predictedPercentageRight, predictedCount = test1Predicted()
    
    percentages = ['percentages', realPercentageRight, predictedPercentageRight]
    counts = ['numeratorCount',realCount, predictedCount]
    
    metricsArray.append(['  HEADERS  ','wrtReal', 'wrtPredicted'])
    metricsArray.append(percentages)
    metricsArray.append(counts)
    
    import csv
    with open(metricsFilePath, 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(metricsArray)

def displayAndSaveResults(save, realStockPrices, predictedStockPrice, currentStockID, graphPath):
    realLabel = 'Real ' +  currentStockID + ' Stock Price'
    predictedLabel = 'Predicted ' +  currentStockID + ' Stock Price'
    titleLabel = currentStockID + ' Stock Price Prediction'
    yLabel = currentStockID + ' Price' 
    plt.plot(realStockPrices, color = 'red', label = realLabel)
    plt.plot(predictedStockPrice, color = 'blue', label = predictedLabel)
    plt.title(titleLabel)
    plt.xlabel('Time')
    plt.ylabel(yLabel)
    plt.legend()
    
    if save:
        plt.savefig(graphPath)

    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
