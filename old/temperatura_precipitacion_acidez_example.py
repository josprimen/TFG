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

datos_clima_2015 = read_csv('files/datos_clima_abril_mayo_2015.csv', usecols=[1,2], engine='python')
datos_clima_2015 = datos_clima_2015.values
datos_clima_2015df = DataFrame(datos_clima_2015)
datos_clima_2015c1 = datos_clima_2015df[0].values
datos_clima_2015c2 = datos_clima_2015df[1].values

datos_clima_2016 = read_csv('files/datos_clima_abril_mayo_2016.csv', usecols=[1,2], engine='python')
datos_clima_2016 = datos_clima_2016.values
datos_clima_2016df = DataFrame(datos_clima_2016)
datos_clima_2016c1 = datos_clima_2016df[0].values
datos_clima_2016c2 = datos_clima_2016df[1].values

datos_clima_2017 = read_csv('files/datos_clima_abril_mayo_2017.csv', usecols=[1,2], engine='python')
datos_clima_2017 = datos_clima_2017.values
datos_clima_2017df = DataFrame(datos_clima_2017)
datos_clima_2017c1 = datos_clima_2017df[0].values
datos_clima_2017c2 = datos_clima_2017df[1].values

datos_clima_2018 = read_csv('files/datos_clima_abril_mayo_2018.csv', usecols=[1,2], engine='python')
datos_clima_2018 = datos_clima_2018.values
datos_clima_2018df = DataFrame(datos_clima_2018)
datos_clima_2018c1 = datos_clima_2018df[0].values
datos_clima_2018c2 = datos_clima_2018df[1].values

datos_precipitacion = np.concatenate((datos_clima_2015c1, datos_clima_2016c1, datos_clima_2017c1, datos_clima_2018c1))
datos_precipitacion = DataFrame(datos_precipitacion)
datos_precipitacion = datos_precipitacion.values
datos_temperatura = np.concatenate((datos_clima_2015c2, datos_clima_2016c2, datos_clima_2017c2, datos_clima_2018c2))
datos_temperatura = DataFrame(datos_temperatura)
datos_temperatura = datos_temperatura.values


datos_acidez_2015 = read_csv('files/media_acidez_dias_2015.csv', usecols=[1], engine='python')
datos_acidez_2015 = datos_acidez_2015.values[0:61]

''' Salen los datos de manera distinta por eso aqui arriba pasamos a dataframe y de nuevo a values
datos_acidez_2015 = read_csv('files/media_acidez_dias_2015.csv', usecols=[1], engine='python')
datos_acidez_2015 = datos_acidez_2015.values
datos_acidez_2015df = DataFrame(datos_acidez_2015)
datos_acidez_2015 = datos_acidez_2015df[0].values

Con esto tenemos un array normal [] y con lo que hay tenemos un [[]]
'''

datos_acidez_2016 = read_csv('files/media_acidez_dias_2016.csv', usecols=[1], engine='python')
datos_acidez_2016 = datos_acidez_2016.values[0:61]

datos_acidez_2017 = read_csv('files/media_acidez_dias_2017.csv', usecols=[1], engine='python')
datos_acidez_2017 = datos_acidez_2017.values[0:61]

datos_acidez_2018 = read_csv('files/media_acidez_dias_2018.csv', usecols=[1], engine='python')
datos_acidez_2018 = datos_acidez_2018.values[0:61]

datos_acidez = np.concatenate((datos_acidez_2015, datos_acidez_2016, datos_acidez_2017, datos_acidez_2018))


groups = [datos_temperatura, datos_precipitacion, datos_acidez]
aux = 1
pyplot.figure()
for group in groups:
    pyplot.subplot(3, 1, aux)
    pyplot.plot(group)
    if group[0]==datos_temperatura[0]:
        pyplot.title('Temperatura abril-mayo')
    if group[0] == datos_precipitacion[0]:
        pyplot.title('Precipitacion abril-mayo')
    else:
        pyplot.title('acidez')
    aux = aux+1

pyplot.show()


conjunto = concatenate((datos_acidez, datos_temperatura, datos_precipitacion), axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
conjunto_normalizado = scaler.fit_transform(conjunto)


def datosX (conjunto):
    df = DataFrame(conjunto)
    aux = []
    for i in range (len(conjunto)-1):
        a = [[df[0][i], df[1][i], df[2][i]]]
        aux.append(a)
    return np.array(aux)

def datosY (conjunto):
    df = DataFrame(conjunto)
    aux = []
    for i in range (len(conjunto)-1):
        aux.append(df[0][i+1])
    return np.array(aux)


tamaño_entrenamiento = int(len(conjunto_normalizado) * 0.75)
tamaño_test = len(conjunto_normalizado) - tamaño_entrenamiento
entrenamiento = conjunto_normalizado[0:tamaño_entrenamiento]
test = conjunto_normalizado[tamaño_entrenamiento:len(conjunto_normalizado)]


entrenamientoX, entrenamientoY = datosX(entrenamiento), datosY(entrenamiento)
testX, testY = datosX(test), datosY(test)

print('DATOS entrenamientoX')
print(entrenamientoX)
print('DATOS entrenamientoY')
print(entrenamientoY)
print('DATOS testX')
print(testX)
print('DATOS testY')
print(testY)

"""
################################################################
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


model = Sequential()
model.add(LSTM(50, input_shape=(1, 3)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(entrenamientoX, entrenamientoY, epochs=25, validation_data=(testX, testY), verbose=2)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


#Hacer las predicciones sobre la acidez
prediccion_test = model.predict(testX)
print('DATO prediccion_test')
print(prediccion_test)

#Invertir el normalizado para tener los datos en la escala original

#1 Hacer el array del mismo tamaño que el de la salida para poder concatenar
testX = testX.reshape((testX.shape[0], testX.shape[2]))
testY = testY.reshape((len(testY), 1))

#2 Concatenar
concatenado_test_real = concatenate((testY, testX[:, 1:]), axis=1)
concatenado_test_prediccion = concatenate((prediccion_test, testX[:, 1:]), axis=1)

#3 Invertir el normalizado
inversion_test_real = scaler.inverse_transform(concatenado_test_real)
inversion_test_prediccion = scaler.inverse_transform(concatenado_test_prediccion)

#4 Obtener las predicciones invertidas
datos_real_test = inversion_test_real[:, 0]
print('Datos acidez test')
print(datos_real_test)

datos_prediccion_test = inversion_test_prediccion[:, 0]
print('datos prediccion test')
print(datos_prediccion_test)

#Calcular el error cuadrático medio
testScore = sqrt(mean_squared_error(datos_real_test, datos_prediccion_test))
print('Test Score: %.2f RMSE' % (testScore))

#Comparamos graficamente lo real y lo predicho
results = [[datos_real_test, datos_prediccion_test]]
aux = 1
pyplot.figure()
for result in results:
    pyplot.subplot(2, 1, aux)
    pyplot.plot(result[0])
    pyplot.plot(result[1])
    aux = aux+1

pyplot.show()