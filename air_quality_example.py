import numpy as np
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

#Set random seed to make initial weights static.
np.random.seed(7)

#Load and represent the dataset
data = read_csv('raw.csv', usecols=[5, 6, 7, 8, 9, 10, 11, 12], engine='python')
data['pm2.5'].fillna(0, inplace=True)
data = data[24:]
groups = [0, 1, 2, 3, 5, 6, 7]
columns = ['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']
dataset = data.values
features = 8

aux = 1
pyplot.figure()
for group in groups:
    pyplot.subplot(features, 1, aux)
    pyplot.plot(dataset[:, group])
    pyplot.title(columns[aux-1], y=0.5, loc='right')
    aux = aux+1
pyplot.show()

#Encode and normalize data
encoder = LabelEncoder()
dataset[:, 4] = encoder.fit_transform(dataset[:, 4])
dataset = dataset.astype('float32')
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
training_set_hours = 365*24
train = normalize_data[:training_set_hours+1, :]
test = normalize_data[training_set_hours:, :]
trainX, trainY = dataX(train, features), dataY(train)
testX, testY = dataX(test, features), dataY(test)

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
model.add(LSTM(50, input_shape=(1, 8)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#Fit network
history = model.fit(trainX, trainY, epochs=50, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)
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