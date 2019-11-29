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

years = ['2015', '2016', '2017', '2018']

for year in years:
    df = pandas.read_json('files/datos_clima_'+year+'.json')
    df.to_csv('files/files/datos_climatologia_'+year+'.csv')
    data = read_csv('files/files/datos_climatologia_'+year+'.csv', usecols=[11, 17], engine='python')
    data.fillna('0,0', inplace=True)
    data = data.values
    df = DataFrame(data)
    # df.to_csv('files/files/datos_clima_abril_mayo_2014.csv')
    # Los datos son de tipo String al venir de un JSON, modificamos la coma por un punto en el csv con ctrl+r
    # errata 03/05/2016 la roda de andalucia        2017-04-11

    aux = []
    aux2 = []

    for number in df[0]:
        new = ''
        for letter in number:
            if letter == ',':
                new = new + '.'
            else:
                new = new + letter
        aux.append(new)

    for number in df[1]:
        new = ''
        for letter in number:
            if letter == ',':
                new = new + '.'
            else:
                new = new + letter
        aux2.append(new)

    data[:, 0] = aux
    data[:, 1] = aux2

    copy = np.copy(data)
    copy = copy.astype('float32')
    copydf = DataFrame(copy)
    copydf.to_csv('files/files/datos_clima_abril_mayo_' + year + '.csv')
