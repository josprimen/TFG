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


datos=read_csv('datos_aceituna.csv', usecols=[0], engine='python')
datos = datos.values
datos = datos.astype('float32')
df = DataFrame(datos)
df = df.loc[~(df==0).all(axis=1)]
df.to_csv('datos_aceituna_tratados.csv')
datos = df.values
print(datos)

pyplot.plot(datos[:])
pyplot.show()