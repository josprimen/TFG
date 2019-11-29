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

#Load and process data
data=read_csv('files/datos_aceituna_gilena.csv', usecols=[0,2,3], engine='python')
dataset = data.values
kilograms = read_csv('files/datos_aceituna_gilena.csv', usecols=[2], engine='python')
kilograms = kilograms.values
minmax = MinMaxScaler(feature_range=(0,1))
normalize_kilograms = minmax.fit_transform(kilograms)
dataset[:,1] = normalize_kilograms[:,0]

years= ['2015', '2016', '2017', '2018', '2019']
months = ['01','10','11','12']
days = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
sum = []
sum2015 = []
sum2016 = []
sum2017 = []
sum2018 = []
sum2019 = []
sumall = []

#Weigthed average performance (days)
for year in years:
    for month in months:
        for day in days:
            sum_day = 0
            delivery_note_number = 0
            for d in dataset:
                if (day+'/'+month+'/'+year) in d[0]:
                    print('Fecha: ' + d[0] + ' Rendimiento: ' + str(d[2]))
                    sum_day = sum_day + d[1]*d[2]
                    delivery_note_number = delivery_note_number +1
            sumall.append(sum_day)
            if year == '2015':
              if delivery_note_number==0:
                sum2015.append(sum_day)
              else:
                sum2015.append(sum_day/delivery_note_number)
            if year == '2016':
              if delivery_note_number==0:
                sum2016.append(sum_day)
              else:
                sum2016.append(sum_day/delivery_note_number)
            if year == '2017':
              if delivery_note_number==0:
                sum2017.append(sum_day)
              else:
                sum2017.append(sum_day/delivery_note_number)
            if year == '2018':
              if delivery_note_number==0:
                sum2018.append(sum_day)
              else:
                sum2018.append(sum_day/delivery_note_number)
            if year == '2019':
              if delivery_note_number==0:
                sum2019.append(sum_day)
              else:
                sum2019.append(sum_day/delivery_note_number)

print('sum All: ')
print(sumall)
print('len')
print(len(sumall))
print('\n')

print('sum 2015: ')
print(sum2015)
print('len')
print(len(sum2015))
print('\n')

print('sum 2016: ')
print(sum2016)
print('len')
print(len(sum2016))
print('\n')

print('sum 2017: ')
print(sum2017)
print('len')
print(len(sum2017))
print('\n')

print('sum 2018: ')
print(sum2018)
print('len')
print(len(sum2018))
print('\n')


sum_years = [sum2015, sum2016, sum2017, sum2018, sum2019]
aux = 1
pyplot.figure()
for result in sum_years:
    pyplot.subplot(5, 1, aux)
    pyplot.plot(result)
    aux = aux+1
pyplot.show()

#Drop zeros
sum2015df = DataFrame(sum2015)
sum2015df = sum2015df.loc[~(sum2015df==0).all(axis=1)]
#sum2015df.to_csv('files/meday_rendimiento_days_2015.csv')
sum2015df = sum2015df.values
sum2016df = DataFrame(sum2016)
sum2016df = sum2016df.loc[~(sum2016df==0).all(axis=1)]
#sum2016df.to_csv('files/meday_rendimiento_days_2016.csv')
sum2016df = sum2016df.values
sum2017df = DataFrame(sum2017)
sum2017df = sum2017df.loc[~(sum2017df==0).all(axis=1)]
#sum2017df.to_csv('files/meday_rendimiento_days_2017.csv')
sum2017df = sum2017df.values
sum2018df = DataFrame(sum2018)
sum2018df = sum2018df.loc[~(sum2018df==0).all(axis=1)]
#sum2018df.to_csv('files/meday_rendimiento_days_2018.csv')
sum2018df = sum2018df.values
sum2019df = DataFrame(sum2019)
sum2019df = sum2019df.loc[~(sum2019df==0).all(axis=1)]
#sum2019df.to_csv('files/meday_rendimiento_days_2019.csv')
sum2019df = sum2019df.values


print('sum 2015 Drop zeros: ')
print(sum2015df)
print('len')
print(len(sum2015df))
print('\n')

print('sum 2016 Drop zeros: ')
print(sum2016df)
print('len')
print(len(sum2016df))
print('\n')

print('sum 2017 Drop zeros: ')
print(sum2017df)
print('len')
print(len(sum2017df))
print('\n')

print('sum 2018 Drop zeros: ')
print(sum2018df)
print('len')
print(len(sum2018df))
print('\n')

print('sum 2019 Drop zeros: ')
print(sum2019df)
print('len')
print(len(sum2019df))
print('\n')

sum_years_df = [sum2015df, sum2016df, sum2017df, sum2018df, sum2019df]
aux = 1
pyplot.figure()
for result in sum_years_df:
    pyplot.subplot(5, 1, aux)
    pyplot.plot(result)
    aux = aux+1

pyplot.show()

#Concatenate all years data
performance_data = np.concatenate((sum2015df, sum2016df, sum2017df, sum2018df, sum2019df))
print('datos rendimiento:')
print(performance_data)

performance_datadf = DataFrame(performance_data)
performance_datadf.to_csv('files/csv_media_ponderada_rendimiento.csv')

#Set random seed to make initial weights static.
np.random.seed(7)

#Load and represent the dataset
dataset = performance_data
dataset= dataset.astype('float32')
plt.plot(dataset)
plt.show()

#Normalize the dataset
minmax = MinMaxScaler(feature_range=(0,1))
dataset = minmax.fit_transform(dataset)

#Split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train = dataset[0:train_size]
test = dataset[train_size:len(dataset)]

#Define a function to format the dataset
def dataX (dataset, look_back):
    aux = []
    for i in range (0,(len(dataset)-(look_back+1))):
        a = []
        for n in range (look_back):
            a.append(dataset[i+n])
        aux.append(a)
    return np.array(aux)

def dataY (dataset, look_back):
    aux = []
    for i in range ((look_back-1),(len(dataset)-2)):
        aux.append(dataset[i+1,0])
    return np.array(aux)

#Format dataset
look_back = 5 #time steps
trainX = dataX(train, look_back)
trainY = dataY(train, look_back)

testX = dataX(test, look_back)
testY = dataY(test, look_back)



print('Conjunto de entrenamientoX t: \n' + str(trainX))
print('Con shape' + str(trainX.shape))
print('Conjunto de entrenamientoY t+1: \n' + str(trainY))
print('Con shape' + str(trainY.shape))
print('Conjunto pruebaX t: \n' + str(testX))
print('Conjunto pruebaY t+1: \n' + str(testY))



#Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back,1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, min_delta=0.001)

history = model.fit(trainX, trainY, epochs=50, validation_split=0.3, batch_size=1, verbose=2, callbacks=[es])

#Represent loss
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#Make prediction
trainPredict = model.predict(trainX)
print('La predicción de la red neuronal para los data de entrenamiento: \n' + str(trainPredict))
testPredict = model.predict(testX)
print('La predicción de la red neuronal para los data de test: \n' + str(testPredict))
#Invert the normalize
trainPredict = minmax.inverse_transform(trainPredict)
print('La predicción de la red neuronal para los data de entrenamiento INVERTIDOS: \n' + str(trainPredict))
trainY = minmax.inverse_transform([trainY])
testPredict = minmax.inverse_transform(testPredict)
print('La predicción de la red neuronal para los data de test INVERTLDOS: \n' + str(testPredict))
testY = minmax.inverse_transform([testY])
#Calculate the mean square error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#Shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#Shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
#Plot baseline and predictions
plt.plot(minmax.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()