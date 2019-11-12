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

'''
def parse(x):
	return datetime.strptime(x, '%Y %m %d %H')

dataset = read_csv('raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)


datos=read_csv('datos_aceituna_gilena.csv', engine='python')
datos = datos.values
df = DataFrame(datos)

aa = df[0][0]
aa.__contains__('21/10')
dia = 12
mes = 11
barra = '/'
fecha = str(dia) + barra + str(mes)
'''


datos=read_csv('datos_aceituna_gilena.csv', usecols=[0,5], engine='python')
datoss = datos.values
meses = ['11','12']
dias = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
suma = []

for mes in meses:
    for dia in dias:
        media_dia = 0
        #print(str(i)+'/'+mes)
        for d in datoss:
            if (dia+'/'+mes) in d[0]:
            #if d[0].__contains__(str(i)+'/'+mes):
                print('Fecha: ' + d[0] + ' Acidez: ' + str(d[1]))
                media_dia = media_dia + d[1]
        suma.append(media_dia)

print('Suma: ')
print(suma)