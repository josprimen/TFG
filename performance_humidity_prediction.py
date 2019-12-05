import numpy as np
import math
import matplotlib.pyplot as plt
from pandas import read_csv
from datetime import datetime
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping



#Set random seed to make initial weights static.
np.random.seed(7)

#Load and represent the dataset
#conjunto = concatenate((performance_data, acidity_data), axis=1)
data = read_csv('files/datos_aceituna_gilena.csv', usecols=[3, 4], engine='python')
df = DataFrame(data)
df = df.loc[~(df==0).all(axis=1)]
dataset = df.values
dataset= dataset.astype('float32')

groups = [0,1]
columns = ['Rendimiento', 'Humedad']
features = 2
look_back = 1

aux = 1
pyplot.figure()
for group in groups:
    pyplot.subplot(features, 1, aux)
    pyplot.plot(dataset[:, group])
    pyplot.title(columns[aux-1], y=0.5, loc='right')
    aux = aux+1
pyplot.show()

#Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
normalize_data = scaler.fit_transform(dataset)

#Define a function to format the dataset
def dataX (dataset, features):
    df = DataFrame(dataset)
    aux = []
    for i in range (len(dataset)-1):
        a = []
        for n in range (features):
            a.append(df[n][i])
        aux.append([a])
    return np.array(aux)

def dataY (dataset):
    df = DataFrame(dataset)
    aux = []
    for i in range (len(dataset)-1):
        aux.append(df[0][i+1])
    return np.array(aux)

#Split into train and test sets
train_size = int(len(dataset) * 0.67)
train = normalize_data[:train_size, :]
test = normalize_data[train_size:, :]
trainX, trainY = dataX(train, features), dataY(train)
testX, testY = dataX(test, features), dataY(test)

print(dataset[:10])
print('train dataset')
print(train)
print('data trainX')
print(trainX)
print(len(trainX))
print('data trainY')
print(trainY)
print(len(trainY))
print('data testX')
print(testX)
print(len(testX))
print('data testY')
print(testY)
print(len(testY))

#Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, features)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#Fit network
history = model.fit(trainX, trainY, epochs=150, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)
#Plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#Make prediction
prediction_train = model.predict(trainX)
print('DATO prediction_train')
print(prediction_train)
prediction_test = model.predict(testX)
print('DATO prediction_test')
print(prediction_test)

#Format for concat
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((len(trainY), 1))
testX = testX.reshape((testX.shape[0], testX.shape[2]))
testY = testY.reshape((len(testY), 1))
#Concat
concat_train_real = concatenate((trainY, trainX[:, 1:]), axis=1)
concat_train_prediction = concatenate((prediction_train, trainX[:, 1:]), axis=1)
concat_test_real = concatenate((testY, testX[:, 1:]), axis=1)
concat_test_prediction = concatenate((prediction_test, testX[:, 1:]), axis=1)
##Invert the normalize
inversion_train_real = scaler.inverse_transform(concat_train_real)
inversion_train_prediction = scaler.inverse_transform(concat_train_prediction)
inversion_test_real = scaler.inverse_transform(concat_test_real)
inversion_test_prediction = scaler.inverse_transform(concat_test_prediction)
#Invert predictions
data_real_train = inversion_train_real[:, 0]
data_prediction_train = inversion_train_prediction[:, 0]
data_real_test = inversion_test_real[:, 0]
data_prediction_test = inversion_test_prediction[:, 0]
##Calculate the mean square error
trainScore = sqrt(mean_squared_error(data_real_train, data_prediction_train))
testScore = sqrt(mean_squared_error(data_real_test, data_prediction_test))
print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))

#Shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset[:,0])
trainPredictPlot[:] = np.nan
trainPredictPlot[:len(data_prediction_train)] = data_prediction_train
#Shift test predictions for plotting
testPredictPlot = np.empty_like(dataset[:,0])
testPredictPlot[:] = np.nan
testPredictPlot[len(data_prediction_train)+1:len(dataset)-1] = data_prediction_test
#Plot baseline and predictions
#data_prediction = concatenate((data_prediction_train, data_prediction_test))
plt.plot(dataset[:,0])
#plt.plot(data_prediction)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()