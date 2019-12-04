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

cli_data_2015 = read_csv('files/datos_clima_abril_mayo_2015.csv', usecols=[1,2], engine='python')
cli_data_2015 = cli_data_2015.values
cli_data_2015df = DataFrame(cli_data_2015)
cli_data_2015c1 = cli_data_2015df[0].values
cli_data_2015c2 = cli_data_2015df[1].values

cli_data_2016 = read_csv('files/datos_clima_abril_mayo_2016.csv', usecols=[1,2], engine='python')
cli_data_2016 = cli_data_2016.values
cli_data_2016df = DataFrame(cli_data_2016)
cli_data_2016c1 = cli_data_2016df[0].values
cli_data_2016c2 = cli_data_2016df[1].values

cli_data_2017 = read_csv('files/datos_clima_abril_mayo_2017.csv', usecols=[1,2], engine='python')
cli_data_2017 = cli_data_2017.values
cli_data_2017df = DataFrame(cli_data_2017)
cli_data_2017c1 = cli_data_2017df[0].values
cli_data_2017c2 = cli_data_2017df[1].values

cli_data_2018 = read_csv('files/datos_clima_abril_mayo_2018.csv', usecols=[1,2], engine='python')
cli_data_2018 = cli_data_2018.values
cli_data_2018df = DataFrame(cli_data_2018)
cli_data_2018c1 = cli_data_2018df[0].values
cli_data_2018c2 = cli_data_2018df[1].values

precip_data = np.concatenate((cli_data_2015c1, cli_data_2016c1, cli_data_2017c1, cli_data_2018c1))
precip_data = DataFrame(precip_data)
precip_data = precip_data.values
temp_data = np.concatenate((cli_data_2015c2, cli_data_2016c2, cli_data_2017c2, cli_data_2018c2))
temp_data = DataFrame(temp_data)
temp_data = temp_data.values







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
sum = []
sum2015 = []
sum2016 = []
sum2017 = []
sum2018 = []
sumall = []

#Weigthed average acidity (days)
for year in years:
    for month in months:
        for day in days:
            sum_day = 0
            delivery_note_number = 0
            for d in dataset:
                if (day+'/'+month+'/'+year) in d[0]:
                    print('Fecha: ' + d[0] + ' Acidez: ' + str(d[2]))
                    sum_day = sum_day + d[1]*d[2]
                    delivery_note_number = delivery_note_number +1
            sumall.append(sum_day)
            if year == '2015':
              if delivery_note_number==0:
                sum2015.append(sum_day)
              else:
                sum2015.append(sum_day/delivery_note_number)
            if year == '2016':
                if month in ['10', '11', '12']:
                    if delivery_note_number == 0:
                        sum2016.append(sum_day)
                    else:
                        sum2016.append(sum_day / delivery_note_number)
                if month in ['01']:
                    if delivery_note_number == 0:
                        sum2015.append(sum_day)
                    else:
                        sum2015.append(sum_day / delivery_note_number)
            if year == '2017':
                if month in ['10', '11', '12']:
                    if delivery_note_number == 0:
                        sum2017.append(sum_day)
                    else:
                        sum2017.append(sum_day / delivery_note_number)
                if month in ['01']:
                    if delivery_note_number == 0:
                        sum2016.append(sum_day)
                    else:
                        sum2016.append(sum_day / delivery_note_number)
            if year == '2018':
                if month in ['10', '11', '12']:
                    if delivery_note_number == 0:
                        sum2018.append(sum_day)
                    else:
                        sum2018.append(sum_day / delivery_note_number)
                if month in ['01']:
                    if delivery_note_number == 0:
                        sum2017.append(sum_day)
                    else:
                        sum2017.append(sum_day / delivery_note_number)
            if year == '2019':
              if delivery_note_number==0:
                sum2018.append(sum_day)
              else:
                sum2018.append(sum_day/delivery_note_number)


#Drop zeros
acidity_data_2015 = DataFrame(sum2015)
acidity_data_2015 = acidity_data_2015.loc[~(acidity_data_2015==0).all(axis=1)]
#acidity_data_2015.to_csv('files/media_acidez_dias_2015.csv')
acidity_data_2016 = DataFrame(sum2016)
acidity_data_2016 = acidity_data_2016.loc[~(acidity_data_2016==0).all(axis=1)]
#acidity_data_2016.to_csv('files/media_acidez_dias_2016.csv')
acidity_data_2017 = DataFrame(sum2017)
acidity_data_2017 = acidity_data_2017.loc[~(acidity_data_2017==0).all(axis=1)]
#acidity_data_2017.to_csv('files/media_acidez_dias_2017.csv')
acidity_data_2018 = DataFrame(sum2018)
acidity_data_2018 = acidity_data_2018.loc[~(acidity_data_2018==0).all(axis=1)]
#acidity_data_2018.to_csv('files/media_acidez_dias_2018.csv')

acidity_data_2015 = acidity_data_2015.values[0:61]
acidity_data_2016 = acidity_data_2016.values[0:61]
acidity_data_2017 = acidity_data_2017.values[0:61]
acidity_data_2018 = acidity_data_2018.values[0:61]
acidity_data = np.concatenate((acidity_data_2015, acidity_data_2016, acidity_data_2017, acidity_data_2018))



#Set random seed to make initial weights static.
np.random.seed(7)

#Load and represent the dataset
#conjunto = concatenate((performance_data, acidity_data), axis=1)
dataset = concatenate((acidity_data, temp_data, precip_data), axis=1)
groups = [0,1,2]
columns = ['Acidez', 'Temperatura', 'Precipitación']
features = 3
look_back = 1

aux = 1
pyplot.figure()
for group in groups:
    pyplot.subplot(features, 1, aux)
    pyplot.plot(dataset[:, group])
    pyplot.title(columns[aux-1], y=0.5, loc='right')
    aux = aux+1
pyplot.show()

#Normalize data
dataset = dataset.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
normalize_data = scaler.fit_transform(dataset)

#Define a function to format the dataset
def dataX (dataset, features):
    df = DataFrame(dataset)
    aux = []
    for i in range (len(dataset)-1):
        a = []
        for n in range (features):
            a.append(df[n][i])
        aux.append([a])
    return np.array(aux)

def dataY (dataset):
    df = DataFrame(dataset)
    aux = []
    for i in range (len(dataset)-1):
        aux.append(df[0][i+1])
    return np.array(aux)

#Split into train and test sets
train_size = int(len(dataset) * 0.67)
train = normalize_data[:train_size, :]
test = normalize_data[train_size:, :]
trainX, trainY = dataX(train, features), dataY(train)
testX, testY = dataX(test, features), dataY(test)

print('train dataset')
print(train)
print('data trainX')
print(trainX)
print(len(trainX))
print('data trainY')
print(trainY)
print(len(trainY))
print('data testX')
print(testX)
print(len(testX))
print('data testY')
print(testY)
print(len(testY))

#Create and fit the LSTM network
model = Sequential()
model.add(LSTM(50, input_shape=(look_back, features)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
#Fit network
history = model.fit(trainX, trainY, epochs=50, batch_size=72, validation_data=(testX, testY), verbose=2, shuffle=False)
#Plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

#Make prediction
prediction_train = model.predict(trainX)
print('DATO prediction_train')
print(prediction_train)
prediction_test = model.predict(testX)
print('DATO prediction_test')
print(prediction_test)

#Format for concat
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))
trainY = trainY.reshape((len(trainY), 1))
testX = testX.reshape((testX.shape[0], testX.shape[2]))
testY = testY.reshape((len(testY), 1))
#Concat
concat_train_real = concatenate((trainY, trainX[:, 1:]), axis=1)
concat_train_prediction = concatenate((prediction_train, trainX[:, 1:]), axis=1)
concat_test_real = concatenate((testY, testX[:, 1:]), axis=1)
concat_test_prediction = concatenate((prediction_test, testX[:, 1:]), axis=1)
##Invert the normalize
inversion_train_real = scaler.inverse_transform(concat_train_real)
inversion_train_prediction = scaler.inverse_transform(concat_train_prediction)
inversion_test_real = scaler.inverse_transform(concat_test_real)
inversion_test_prediction = scaler.inverse_transform(concat_test_prediction)
#Invert predictions
data_real_train = inversion_train_real[:, 0]
data_prediction_train = inversion_train_prediction[:, 0]
data_real_test = inversion_test_real[:, 0]
data_prediction_test = inversion_test_prediction[:, 0]
##Calculate the mean square error
trainScore = sqrt(mean_squared_error(data_real_train, data_prediction_train))
testScore = sqrt(mean_squared_error(data_real_test, data_prediction_test))
print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))

#Shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset[:,0])
trainPredictPlot[:] = np.nan
trainPredictPlot[:len(data_prediction_train)] = data_prediction_train
#Shift test predictions for plotting
testPredictPlot = np.empty_like(dataset[:,0])
testPredictPlot[:] = np.nan
testPredictPlot[len(data_prediction_train)+1:len(dataset)-1] = data_prediction_test
#Plot baseline and predictions
#data_prediction = concatenate((data_prediction_train, data_prediction_test))
plt.plot(dataset[:,0])
#plt.plot(data_prediction)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()