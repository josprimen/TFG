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


datos=read_csv('datos_aceituna_gilena.csv', usecols=[0,5], engine='python')
datoss = datos.values
anyos= ['2015', '2016', '2017', '2018']
meses = ['11','12']
dias = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
suma2015 = []
suma2016 = []
suma2017 = []
suma2018 = []
sumaall = []

for anyo in anyos:
    for mes in meses:
        for dia in dias:
            media_dia = 0
            numero_almazaras = 0
            #print(str(i)+'/'+mes)
            for d in datoss:
                if (dia+'/'+mes+'/'+anyo) in d[0]:
                #if d[0].__contains__(str(i)+'/'+mes):
                    print('Fecha: ' + d[0] + ' Acidez: ' + str(d[1]))
                    media_dia = media_dia + d[1]
                    numero_almazaras = numero_almazaras +1
            sumaall.append(media_dia)
            if anyo == '2015':
                if numero_almazaras==0:
                    suma2015.append(media_dia)
                else:
                    suma2015.append(media_dia/numero_almazaras)
            if anyo == '2016':
                if numero_almazaras==0:
                    suma2016.append(media_dia)
                else:
                    suma2016.append(media_dia/numero_almazaras)
            if anyo == '2017':
                if numero_almazaras==0:
                    suma2017.append(media_dia)
                else:
                    suma2017.append(media_dia/numero_almazaras)
            if anyo == '2018':
                if numero_almazaras==0:
                    suma2018.append(media_dia)
                else:
                    suma2018.append(media_dia/numero_almazaras)


print('Suma All: ')
print(sumaall)
print('len')
print(len(sumaall))
print('\n')

print('Suma 2015: ')
print(suma2015)
print('len')
print(len(suma2015))
print('\n')

print('Suma 2016: ')
print(suma2016)
print('len')
print(len(suma2016))
print('\n')

print('Suma 2017: ')
print(suma2017)
print('len')
print(len(suma2017))
print('\n')

print('Suma 2018: ')
print(suma2018)
print('len')
print(len(suma2018))
print('\n')


suma_anyos = [suma2015, suma2016, suma2017, suma2018]
aux = 1
pyplot.figure()
for result in suma_anyos:
    pyplot.subplot(4, 1, aux)
    pyplot.plot(result)
    aux = aux+1

pyplot.show()

suma2015df = DataFrame(suma2015)
suma2015df = suma2015df.loc[~(suma2015df==0).all(axis=1)]
suma2016df = DataFrame(suma2016)
suma2016df = suma2016df.loc[~(suma2016df==0).all(axis=1)]
suma2017df = DataFrame(suma2017)
suma2017df = suma2017df.loc[~(suma2017df==0).all(axis=1)]
suma2018df = DataFrame(suma2018)
suma2018df = suma2018df.loc[~(suma2018df==0).all(axis=1)]

suma_anyos_df = [suma2015df, suma2016df, suma2017df, suma2018df]
aux = 1
pyplot.figure()
for result in suma_anyos_df:
    pyplot.subplot(4, 1, aux)
    pyplot.plot(result.values)
    aux = aux+1

pyplot.show()