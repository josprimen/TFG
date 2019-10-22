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
        a = [[df[0][i], df[1][i], df[2][i], df[3][i], df[4][i], df[5][i], df[6][i]]]
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
model.add(LSTM(50, input_shape=(1, 7)))
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
prediccion_entrenamiento =
prediccion_entrenamiento = scaler.inverse_transform(prediccion_entrenamiento)
entrenamientoY = scaler.inverse_transform([entrenamientoY])
prediccion_test = scaler.inverse_transform(prediccion_test)
testY = scaler.inverse_transform([testY])
#Calcular el error cuadrático medio
trainScore = sqrt(mean_squared_error(entrenamientoY[0], prediccion_entrenamiento[:, 0]))
testScore = sqrt(mean_squared_error(testY[0], prediccion_test[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))


