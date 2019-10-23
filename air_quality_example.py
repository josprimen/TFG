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


datos = read_csv('raw.csv', usecols=[5, 6, 7, 8, 9, 10, 11, 12], engine='python')
datos['pm2.5'].fillna(0, inplace=True)
datos = datos[24:]
#datos.to_csv('datosmios.csv')
groups = [0, 1, 2, 3, 5, 6, 7]

datoss = datos.values

#Imprimimos una gráfica con los datos por pantalla (cada tipo de dato en un subgráfico)
aux = 1
pyplot.figure()
for group in groups:
    pyplot.subplot(7, 1, aux)
    pyplot.plot(datoss[:, group])
    aux = aux+1

pyplot.show()

#Encodeamos los datos de la dirección del viento para que tome un valor entero en lugar de ser categórico
encoder = LabelEncoder()
datoss[:, 4] = encoder.fit_transform(datoss[:, 4])

#Pasamos todos los datos a float y los normalizamos
datoss = datoss.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
datos_normalizados = scaler.fit_transform(datoss)
#datos_normalizados.shape[1]

#df = DataFrame(datos_normalizados)
#len(df[0])
#print('df[0][0]')

def datosX (conjunto):
    df = DataFrame(conjunto)
    aux = []
    for i in range (len(conjunto)-2):
        a = [[df[0][i], df[1][i], df[2][i], df[3][i], df[4][i], df[5][i], df[6][i], df[7][i]]]
        aux.append(a)
    return np.array(aux)

def datosY (conjunto):
    df = DataFrame(conjunto)
    aux = []
    for i in range (len(conjunto)-2):
        aux.append(df[0][i+1])
    return np.array(aux)

#Un año de entrenamiento
numero_horas_entrenamiento = 365*24
entrenamiento = datos_normalizados[:numero_horas_entrenamiento, :]
test = datos_normalizados[numero_horas_entrenamiento:, :]

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
model.add(LSTM(50, input_shape=(1, 8)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(entrenamientoX, entrenamientoY, epochs=50, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


#Hacer predicciones
prediccion_entrenamiento = model.predict(entrenamientoX)
print('DATO prediccion_entrenamiento')
print(prediccion_entrenamiento)
prediccion_test = model.predict(testX)
print('DATO prediccion_test')
print(prediccion_test)

#Invertir el normalizado para tener los datos en la escala original
#1 Hacer el array del mismo tamaño que el de la salida para poder concatenar
entrenamientoX = entrenamientoX.reshape((entrenamientoX.shape[0], entrenamientoX.shape[2]))
entrenamientoY = entrenamientoY.reshape((len(entrenamientoY), 1))
testX = testX.reshape((testX.shape[0], testX.shape[2]))
testY = testY.reshape((len(testY), 1))
#2 Concatenar
concatenado_entrenamiento_real = concatenate((entrenamientoY, entrenamientoX[:, 1:]), axis=1)
concatenado_entrenamiento_prediccion = concatenate((prediccion_entrenamiento, entrenamientoX[:, 1:]), axis=1)
concatenado_test_real = concatenate((testY, testX[:, 1:]), axis=1)
concatenado_test_prediccion = concatenate((prediccion_test, testX[:, 1:]), axis=1)
#3 Invertir el normalizado
inversion_entrenamiento_real = scaler.inverse_transform(concatenado_entrenamiento_real)
inversion_entrenamiento_prediccion = scaler.inverse_transform(concatenado_entrenamiento_prediccion)
inversion_test_real = scaler.inverse_transform(concatenado_test_real)
inversion_test_prediccion = scaler.inverse_transform(concatenado_test_prediccion)

#4 Obtener las predicciones invertidas
datos_real_entrenamiento = inversion_entrenamiento_real[:, 0]
datos_prediccion_entrenamiento = inversion_entrenamiento_prediccion[:, 0]
datos_real_test = inversion_test_real[:, 0]
datos_prediccion_test = inversion_test_prediccion[:, 0]



#Calcular el error cuadrático medio
trainScore = sqrt(mean_squared_error(datos_real_entrenamiento, datos_prediccion_entrenamiento))
testScore = sqrt(mean_squared_error(datos_real_test, datos_prediccion_test))
print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))


