import pandas
import matplotlib.pyplot as plt
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


#Set random seed to make initial weights static.
np.random.seed(7)

#Load and represent the dataset
data = pandas.read_csv('files/files/airline-passengers.csv', usecols=[1], engine='python')
plt.plot(data)
plt.show()
dataset = data.values
dataset = dataset.astype('float32')

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

model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


#Make prediction
trainPredict = model.predict(trainX)
print('La predicción de la red neuronal para los datos de entrenamiento: \n' + str(trainPredict))
testPredict = model.predict(testX)
print('La predicción de la red neuronal para los datos de test: \n' + str(testPredict))
#Invert the normalize
trainPredict = minmax.inverse_transform(trainPredict)
print('La predicción de la red neuronal para los datos de entrenamiento INVERTIDOS: \n' + str(trainPredict))
trainY = minmax.inverse_transform([trainY])
testPredict = minmax.inverse_transform(testPredict)
print('La predicción de la red neuronal para los datos de test INVERTLDOS: \n' + str(testPredict))
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