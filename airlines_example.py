import pandas
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#Esto para representar los datos y ver que funciona
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

#Convertir un array de valores en un conjunto en forma de matriz
#Lo único que estamos haciendo es ir de mes en mes y ver cuantos viajeros había. Viajeros será la x y la y será el mes
#Esto se hace creo para luego poder coger de dos en dos meses y cosas así
def crear_conjunto(conjunto, vista_atras=1):
    datosX, datosY = [],[]
    for i in range(len(conjunto)- vista_atras-1):
        #aux = conjunto[i:(i+vista_atras), 0] esto te devuelve un array por cada valor y haces return de un array de arrays
        aux = conjunto[i, 0]
        datosX.append(aux)
        datosY.append(conjunto[i + vista_atras, 0])
    return np.array(datosX), np.array(datosY)

entrenamientoX, entrenamientoY = crear_conjunto(entrenamiento)
testX, testY = crear_conjunto(test)

print('Conjunto entrenamiento t: \n' + str(entrenamientoX))
print('Conjunto entrenamiento t+1: \n' + str(entrenamientoY))
print('Conjunto pruebas t: \n' + str(testX))
print('Conjunto pruebas t+1: \n' + str(testY))

#Si tenemos Y que es una matriz de tamaño n filas y m columnas Y.shape devuelve (n,m)
#array = geek.arange(8) te crea un array del 0 al 7
#array = geek.arange(8).reshape(2, 4) te crea una matriz de 2 filas y 4 columnas con ese array [[0 1 2 3]
 #                                                                                             [4 5 6 7]]

entrenamientoX = np.reshape(entrenamientoX, (entrenamientoX.shape[0], 1, entrenamientoX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print('Conjunto entrenamiento reshapeado t: \n' + str(entrenamientoX))
print('Conjunto pruebas reshapeado t: \n' + str(testX))
