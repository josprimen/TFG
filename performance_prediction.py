import pandas
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


"""
################################################################
datos=read_csv('files/datos_aceituna_2015_2016.csv', engine='python')
datos = datos.values
datos = datos.astype('float32')
df = DataFrame(datos)
df = df.loc[~(df==0).all(axis=1)]
df = df.loc[~(df==' ').all(axis=1)]
#df.to_csv('files/datos_aceituna_tratados_2015_2016.csv')

da1 = read_csv('files/datos_aceituna_tratados_2015_2016.csv', usecols=[1], engine='python')
da2 = read_csv('files/datos_aceituna_tratados_2016_2017.csv', usecols=[1], engine='python')
da3 = read_csv('files/datos_aceituna_tratados_2017_2018.csv', usecols=[1], engine='python')
da4 = read_csv('files/datos_aceituna_tratados_2018_2019.csv', usecols=[1], engine='python')
da1 = da1.values
da2 = da2.values
da3 = da3.values
da4 = da4.values
da_prueba = np.concatenate((da1,da2,da3,da4))
da_prueba_df = DataFrame(da_prueba)
da_prueba_df.to_csv('files/datos_aceituna_tratados.csv')
################################################################
"""

#TALARRUBIAS DATA
#datos=read_csv('files/datos_aceituna_tratados.csv', usecols=[1], engine='python')

#Set random seed to make initial weights static.
np.random.seed(7)

#Load and represent the dataset
data = read_csv('files/datos_aceituna_gilena.csv', usecols=[3], engine='python')
df = DataFrame(data)
df = df.loc[~(df==0).all(axis=1)]
dataset = df.values
dataset= dataset.astype('float32')
plt.plot(dataset)
plt.show()

#Normalize the dataset
minmax = MinMaxScaler(feature_range=(0,1))
dataset = minmax.fit_transform(dataset)

#Split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train = dataset[0:train_size]
test = dataset[train_size:len(dataset)]

#Define a function to format the dataset
def dataX (dataset, look_back):
    aux = []
    for i in range (0,(len(dataset)-(look_back+1))):
        a = []
        for n in range (look_back):
            a.append(dataset[i+n])
        aux.append(a)
    return np.array(aux)

def dataY (dataset, look_back):
    aux = []
    for i in range ((look_back-1),(len(dataset)-2)):
        aux.append(dataset[i+1,0])
    return np.array(aux)

#Format dataset
look_back = 5 #time steps
trainX = dataX(train, look_back)
trainY = dataY(train, look_back)

testX = dataX(test, look_back)
testY = dataY(test, look_back)



print('Conjunto de entrenamientoX t: \n' + str(trainX))
print('Con shape' + str(trainX.shape))
print('Conjunto de entrenamientoY t+1: \n' + str(trainY))
print('Con shape' + str(trainY.shape))
print('Conjunto pruebaX t: \n' + str(testX))
print('Conjunto pruebaY t+1: \n' + str(testY))



#Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, min_delta=0.001)

history = model.fit(trainX, trainY, epochs=20, validation_split=0.3, batch_size=1, verbose=2, callbacks=[es])

#Represent loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#Make prediction
trainPredict = model.predict(trainX)
print('La predicción de la red neuronal para los datos de entrenamiento: \n' + str(trainPredict))
testPredict = model.predict(testX)
print('La predicción de la red neuronal para los datos de test: \n' + str(testPredict))
#Invert the normalize
trainPredict = minmax.inverse_transform(trainPredict)
print('La predicción de la red neuronal para los datos de entrenamiento INVERTIDOS: \n' + str(trainPredict))
trainY = minmax.inverse_transform([trainY])
testPredict = minmax.inverse_transform(testPredict)
print('La predicción de la red neuronal para los datos de test INVERTLDOS: \n' + str(testPredict))
testY = minmax.inverse_transform([testY])
#Calculate the mean square error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#Shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#Shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#Plot baseline and predictions
plt.plot(minmax.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()