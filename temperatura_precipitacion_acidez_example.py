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

datos_clima_2015 = read_csv('datos_clima_abril_mayo_2015.csv', usecols=[1,2], engine='python')
datos_clima_2015 = datos_clima_2015.values
datos_clima_2015df = DataFrame(datos_clima_2015)
datos_clima_2015c1 = datos_clima_2015df[0].values
datos_clima_2015c2 = datos_clima_2015df[1].values

datos_clima_2016 = read_csv('datos_clima_abril_mayo_2016.csv', usecols=[1,2], engine='python')
datos_clima_2016 = datos_clima_2016.values
datos_clima_2016df = DataFrame(datos_clima_2016)
datos_clima_2016c1 = datos_clima_2016df[0].values
datos_clima_2016c2 = datos_clima_2016df[1].values

datos_clima_2017 = read_csv('datos_clima_abril_mayo_2017.csv', usecols=[1,2], engine='python')
datos_clima_2017 = datos_clima_2017.values
datos_clima_2017df = DataFrame(datos_clima_2017)
datos_clima_2017c1 = datos_clima_2017df[0].values
datos_clima_2017c2 = datos_clima_2017df[1].values

datos_clima_2018 = read_csv('datos_clima_abril_mayo_2018.csv', usecols=[1,2], engine='python')
datos_clima_2018 = datos_clima_2018.values
datos_clima_2018df = DataFrame(datos_clima_2018)
datos_clima_2018c1 = datos_clima_2018df[0].values
datos_clima_2018c2 = datos_clima_2018df[1].values

datos_precipitacion = np.concatenate((datos_clima_2015c1, datos_clima_2016c1, datos_clima_2017c1, datos_clima_2018c1))
datos_temperatura = np.concatenate((datos_clima_2015c2, datos_clima_2016c2, datos_clima_2017c2, datos_clima_2018c2))



datos_acidez_2015 = read_csv('media_acidez_dias_2015.csv', usecols=[1], engine='python')
datos_acidez_2015 = datos_acidez_2015.values[0:61]

'''
datos_acidez_2015 = read_csv('media_acidez_dias_2015.csv', usecols=[1], engine='python')
datos_acidez_2015 = datos_acidez_2015.values
datos_acidez_2015df = DataFrame(datos_acidez_2015)
datos_acidez_2015 = datos_acidez_2015df[0].values

Con esto tenemos un array normal [] y con lo que hay tenemos un [[]]
'''

datos_acidez_2016 = read_csv('media_acidez_dias_2016.csv', usecols=[1], engine='python')
datos_acidez_2016 = datos_acidez_2016.values[0:61]

datos_acidez_2017 = read_csv('media_acidez_dias_2017.csv', usecols=[1], engine='python')
datos_acidez_2017 = datos_acidez_2017.values[0:61]

datos_acidez_2018 = read_csv('media_acidez_dias_2018.csv', usecols=[1], engine='python')
datos_acidez_2018 = datos_acidez_2018.values[0:61]

datos_acidez = np.concatenate((datos_acidez_2015, datos_acidez_2016, datos_acidez_2017, datos_acidez_2018))

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
