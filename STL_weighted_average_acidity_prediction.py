import numpy as np
import math
import matplotlib.pyplot as plt
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
from keras.callbacks import EarlyStopping
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib import pyplot
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping

#Load and process data
data=read_csv('files/datos_aceituna_gilena.csv', usecols=[0,2,5], engine='python')
dataset = data.values
kilograms = read_csv('files/datos_aceituna_gilena.csv', usecols=[2], engine='python')
kilograms = kilograms.values
minmax = MinMaxScaler(feature_range=(0,1))
normalize_kilograms = minmax.fit_transform(kilograms)
dataset[:,1] = normalize_kilograms[:,0]

years= ['2015', '2016', '2017', '2018', '2019']
months = ['01','10','11','12']
days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
average = []
aver2015 = []
aver2016 = []
aver2017 = []
aver2018 = []
aver2019 = []
averageall = []

#Weigthed average acidity (days)
for year in years:
    for month in months:
        for day in days:
            average_day = 0
            delivery_note_number = 0
            for d in dataset:
                if (day+'/'+month+'/'+year) in d[0]:
                    print('Fecha: ' + d[0] + ' Acidez: ' + str(d[2]) + ' Kilos (en escala 0-1): '+ str(d[1]))
                    average_day = average_day + d[1]*d[2]
                    delivery_note_number = delivery_note_number +1
            averageall.append(average_day)
            if year == '2015':
              if delivery_note_number==0:
                aver2015.append(average_day)
              else:
                aver2015.append(average_day/delivery_note_number)
            if year == '2016':
              if delivery_note_number==0:
                aver2016.append(average_day)
              else:
                aver2016.append(average_day/delivery_note_number)
            if year == '2017':
              if delivery_note_number==0:
                aver2017.append(average_day)
              else:
                aver2017.append(average_day/delivery_note_number)
            if year == '2018':
              if delivery_note_number==0:
                aver2018.append(average_day)
              else:
                aver2018.append(average_day/delivery_note_number)
            if year == '2019':
              if delivery_note_number==0:
                aver2019.append(average_day)
              else:
                aver2019.append(average_day/delivery_note_number)


#Drop zeros
aver2015df = DataFrame(aver2015)
aver2015df = aver2015df.loc[~(aver2015df==0).all(axis=1)]
aver2015df = aver2015df.values
aver2016df = DataFrame(aver2016)
aver2016df = aver2016df.loc[~(aver2016df==0).all(axis=1)]
aver2016df = aver2016df.values
aver2017df = DataFrame(aver2017)
aver2017df = aver2017df.loc[~(aver2017df==0).all(axis=1)]
aver2017df = aver2017df.values
aver2018df = DataFrame(aver2018)
aver2018df = aver2018df.loc[~(aver2018df==0).all(axis=1)]
aver2018df = aver2018df.values
aver2019df = DataFrame(aver2019)
aver2019df = aver2019df.loc[~(aver2019df==0).all(axis=1)]
aver2019df = aver2019df.values

#Concatenate all years data
acidity_data = np.concatenate((aver2015df, aver2016df, aver2017df, aver2018df, aver2019df))
acidity_datadf = DataFrame(acidity_data)

serie = np.array(acidity_datadf)

result = seasonal_decompose(serie, freq=30)
result.plot()
pyplot.show()