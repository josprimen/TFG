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


#datos=read_csv('files/datos_aceituna_gilena.csv', usecols=[3,5], engine='python')
#datos = datos.values

rendimientocsv = read_csv('files/csv_media_ponderada_rendimiento.csv', usecols=[1], engine='python')
acidezcsv = read_csv('files/csv_media_ponderada_acidez.csv', usecols=[1], engine='python')
datos = concatenate((acidezcsv, rendimientocsv), axis=1)

'''
#PARA HACER UNA PREDICCION DE LA ACIDEZ CAMBIAMOS LAS COLUMNAS DE POSICION (Hay que usar una copia, las asignaciones cambian si no)
rendimiento = np.copy(datos[:,0])
acidez = datos[:,1]
datos[:,0] = acidez
datos[:,1] = rendimiento
'''

print(datos)
df = DataFrame(datos)
#df.to_csv('files/datos_aceituna_gilena.csv')
columna_rendimiento = df[0].values
columna_acidez = df[1].values
groups = [columna_rendimiento, columna_acidez]
aux = 1
pyplot.figure()
for group in groups:
    pyplot.subplot(2, 1, aux)
    pyplot.plot(group)
    if group[0]==columna_rendimiento[0]:
        pyplot.title('rendimiento')
    else:
        pyplot.title('acidez')
    aux = aux+1

pyplot.show()

scaler = MinMaxScaler(feature_range=(0, 1))
conjunto = scaler.fit_transform(datos)

print(conjunto)

def datosX (conjunto):
    df = DataFrame(conjunto)
    aux = []
    for i in range (len(conjunto)-1):
        a = [[df[0][i], df[1][i]]]
        aux.append(a)
    return np.array(aux)

def datosY (conjunto):
    df = DataFrame(conjunto)
    aux = []
    for i in range (len(conjunto)-1):
        aux.append(df[0][i+1])
    return np.array(aux)

tamaño_entrenamiento = int(len(conjunto) * 0.67)
tamaño_test = len(conjunto) - tamaño_entrenamiento
entrenamiento = conjunto[0:tamaño_entrenamiento]
test = conjunto[tamaño_entrenamiento:len(conjunto)]

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


model = Sequential()
model.add(LSTM(50, input_shape=(1, 2)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(entrenamientoX, entrenamientoY, epochs=50, validation_data=(testX, testY), verbose=2)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


#Hacer predicciones
#prediccion_entrenamiento = model.predict(entrenamientoX)
#print('DATO prediccion_entrenamiento')
#print(prediccion_entrenamiento)
prediccion_test = model.predict(testX)
print('DATO prediccion_test')
print(prediccion_test)

#Invertir el normalizado para tener los datos en la escala original
#1 Hacer el array del mismo tamaño que el de la salida para poder concatenar
#entrenamientoX = entrenamientoX.reshape((entrenamientoX.shape[0], entrenamientoX.shape[2]))
#entrenamientoY = entrenamientoY.reshape((len(entrenamientoY), 1))
testX = testX.reshape((testX.shape[0], testX.shape[2]))
testY = testY.reshape((len(testY), 1))

#2 Concatenar
#concatenado_entrenamiento_real = concatenate((entrenamientoY, entrenamientoX[:, 1:]), axis=1)
#concatenado_entrenamiento_prediccion = concatenate((prediccion_entrenamiento, entrenamientoX[:, 1:]), axis=1)
concatenado_test_real = concatenate((testY, testX[:, 1:]), axis=1)
concatenado_test_prediccion = concatenate((prediccion_test, testX[:, 1:]), axis=1)

#3 Invertir el normalizado
#inversion_entrenamiento_real = scaler.inverse_transform(concatenado_entrenamiento_real)
#inversion_entrenamiento_prediccion = scaler.inverse_transform(concatenado_entrenamiento_prediccion)
inversion_test_real = scaler.inverse_transform(concatenado_test_real)
inversion_test_prediccion = scaler.inverse_transform(concatenado_test_prediccion)

#4 Obtener las predicciones invertidas

#datos_real_entrenamiento = inversion_entrenamiento_real[:, 0]

#datos_prediccion_entrenamiento = inversion_entrenamiento_prediccion[:, 0]

datos_real_test = inversion_test_real[:, 0]
print('Datos acidez test')
print(datos_real_test)

datos_prediccion_test = inversion_test_prediccion[:, 0]
print('datos prediccion test')
print(datos_prediccion_test)




#Calcular el error cuadrático medio
#trainScore = sqrt(mean_squared_error(datos_real_entrenamiento, datos_prediccion_entrenamiento))
testScore = sqrt(mean_squared_error(datos_real_test, datos_prediccion_test))
#print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))




#[datos_real_entrenamiento, datos_prediccion_entrenamiento],
results = [[datos_real_test, datos_prediccion_test]]
aux = 1
pyplot.figure()
for result in results:
    pyplot.subplot(2, 1, aux)
    pyplot.plot(result[0])
    pyplot.plot(result[1])
    aux = aux+1

pyplot.show()