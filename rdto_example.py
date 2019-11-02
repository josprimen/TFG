import numpy as np
import math
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


datos=read_csv('datos_aceituna_tratados.csv', usecols=[1], engine='python')
datos = datos.values
datos = datos.astype('float32')
df = DataFrame(datos)
df = df.loc[~(df==0).all(axis=1)]
#df.to_csv('datos_aceituna_tratados_2015_2016.csv')
datos = df.values
print(datos)

pyplot.plot(datos[:])
pyplot.show()

"""
################################################################
da1 = read_csv('datos_aceituna_tratados_2015_2016.csv', usecols=[1], engine='python')
da2 = read_csv('datos_aceituna_tratados_2016_2017.csv', usecols=[1], engine='python')
da3 = read_csv('datos_aceituna_tratados_2017_2018.csv', usecols=[1], engine='python')
da4 = read_csv('datos_aceituna_tratados_2018_2019.csv', usecols=[1], engine='python')
da1 = da1.values
da2 = da2.values
da3 = da3.values
da4 = da4.values
da_prueba = np.concatenate((da1,da2,da3,da4))
da_prueba_df = DataFrame(da_prueba)
da_prueba_df.to_csv('datos_aceituna_tratados.csv')
################################################################
"""

minmax = MinMaxScaler(feature_range=(0,1))
conjunto = minmax.fit_transform(datos)

tamaño_entrenamiento = int(len(conjunto) * 0.67)
tamaño_test = len(conjunto) - tamaño_entrenamiento
entrenamiento = conjunto[0:tamaño_entrenamiento]
test = conjunto[tamaño_entrenamiento:len(conjunto)]

def datosX (conjunto):
    aux = []
    for i in range (len(conjunto)-2):
        a = [conjunto[i]]
        aux.append(a)
    return np.array(aux)

def datosY (conjunto):
    aux = []
    for i in range (len(conjunto)-2):
        aux.append(conjunto[i+1,0])
    return np.array(aux)


entrenamientoX = datosX(entrenamiento)
entrenamientoY = datosY(entrenamiento)

pruebasX = datosX(test)
pruebasY = datosY(test)


paso_atras = 1 #de cúantas unidades de tiempo son los pasos
model = Sequential()
model.add(LSTM(4, input_shape=(1, paso_atras)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(entrenamientoX, entrenamientoY, epochs=10, validation_split=0.3, batch_size=1, verbose=2)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

trainPredict = model.predict(entrenamientoX)
testPredict = model.predict(pruebasX)
trainPredict = minmax.inverse_transform(trainPredict)
trainY = minmax.inverse_transform([entrenamientoY])
testPredict = minmax.inverse_transform(testPredict)
testY = minmax.inverse_transform([pruebasY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(conjunto)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[paso_atras:len(trainPredict)+paso_atras, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(conjunto)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(paso_atras*2)+1:len(conjunto)-1, :] = testPredict
# plot baseline and predictions
pyplot.plot(minmax.inverse_transform(conjunto))
pyplot.plot(trainPredictPlot)
pyplot.plot(testPredictPlot)
pyplot.show()
