import pandas
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#Esto para representar los datos en una gráfica
#conjunto_datos = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
#plt.plot(conjunto_datos)
#plt.show()

#Para hacer que siempre salgan los mismos número aleatorios
np.random.seed(7)

#Vamos a reusar el código de representación de datos para extraer los datos del csv
datos = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
conjunto = datos.values
#pasamos los números a punto flotante porque las redes neuronales funcionan mejor con este tipo de datos
conjunto = conjunto.astype('float32')

#Las redes LSTM son sensible a cómo estén escalados los datos
#Funcionan mejor con datos de 0 a 1 y más al usar sigmoide de activación
#Normalizamos. Para ello creamos una variable con el rango en el que vamos a normalizar y normalizamos en ese rango
minmax = MinMaxScaler(feature_range=(0,1))
#max(conjunto) = 622, min(conjunto)=104.  112-104/622-104 = 0.015
conjunto = minmax.fit_transform(conjunto)
print('El conjunto normalizado: \n' + str(conjunto))

#Como hemos de tener un conjunto de entrenamiento y uno de test, vamos a separar el que tenemos en dos
#un 67% de los datos los usaremos para entrenar y el resto para el conjunto de prueba
tamaño_entrenamiento = int(len(conjunto) * 0.67)
tamaño_test = len(conjunto) - tamaño_entrenamiento
entrenamiento = conjunto[0:tamaño_entrenamiento]
test = conjunto[tamaño_entrenamiento:len(conjunto)]

print('Tamaño cojunto entrenamiento: ' +
      str(len(entrenamiento)), 'Tamaño conjunto pruebas: ' +
      str(len(test)), 'Tamaño total conjunto: ' +str(len(conjunto)))
print('Conjunto entrenamiento: \n' + str(entrenamiento))
print('Conjunto pruebas: \n' + str(test))


#Creamos los conjuntos de dato con el formato adecuado para LSTM

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



print('Conjunto de entrenamiento t mio bien: \n' + str(entrenamientoX))
print('Con shape')
print(entrenamientoX.shape)
print('Conjunto de entrenamiento t+1 mio: \n' + str(entrenamientoY))
print('Conjunto prueba t: \n' + str(pruebasX))
print('Conjunto prueba t+1: \n' + str(pruebasY))



# creamos y entrenamos la red LSTM
paso_atras = 1 #de cúantas unidades de tiempo son los pasos
model = Sequential()
model.add(LSTM(4, input_shape=(1, paso_atras)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(entrenamientoX, entrenamientoY, epochs=100, batch_size=1, verbose=2)


# Hacemos la predicción
trainPredict = model.predict(entrenamientoX)
print('La prediccion de la red neuronal para los datos de entrenamiento: \n' + str(trainPredict))
testPredict = model.predict(pruebasX)
print('La prediccion de la red neuronal para los datos de test: \n' + str(testPredict))
# Invertir el normalizado para obtener los datos de pasajeros en escala unidades de 1000
trainPredict = minmax.inverse_transform(trainPredict)
print('La prediccion de la red neuronal para los datos de entrenamiento INVERTIDOS: \n' + str(trainPredict))
trainY = minmax.inverse_transform([entrenamientoY])
testPredict = minmax.inverse_transform(testPredict)
print('La prediccion de la red neuronal para los datos de test INVERTLDOS: \n' + str(testPredict))
testY = minmax.inverse_transform([pruebasY])
# Calculamos el error cuadrático medio
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
plt.plot(minmax.inverse_transform(conjunto))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()