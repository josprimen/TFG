import numpy as np
import math
import pandas
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


df = pandas.read_json('datos_clima.json')
df.to_csv('datos_climatologia.csv')

datos=read_csv('datos_climatologia.csv', usecols=[11, 17], engine='python')
datos = datos.values
df = DataFrame(datos)
#df.to_csv('datos_clima_abril_mayo.csv')
#Los datos son de tipo String al venir de un JSON, modificamos la coma por un punto en el csv con ctrl+r

aux = []
aux2 = []

for number in df[0]:
    new = ''
    for letra in number:
        if letra == ',':
            new = new + '.'
        else:
            new = new + letra
    aux.append(new)


for number in df[1]:
    new = ''
    for letra in number:
        if letra == ',':
            new = new + '.'
        else:
            new = new + letra
    aux2.append(new)


datos[:,0] = aux
datos[:,1] = aux2

copia = np.copy(datos)
copia = copia.astype('float32')
copiadf = DataFrame(copia)
copiadf.to_csv('datos_clima_abril_mayo.csv')