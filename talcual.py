import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error




# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataframe = pandas.read_csv('airline-passengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')

print('Dataset: ' + str(dataset))

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

print('Dataset normalizado: ' + str(dataset))

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))
print('entrenamiento: ' + str(train))
print('test: ' + str(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		print(i)
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)


# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX_timestep, trainY_timestep = create_dataset(train, look_back)
testX_timestep, testY_timestep = create_dataset(test, look_back)

print('trainX: ' + str(trainX))
print('trainY: ' + str(trainY))
print('testX: ' + str(testX))
print('testY: ' + str(testY))

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print('reshape entrenamientox : ' + str(trainX))
print('reshape testx: ' + str(testX))

# reshape using time_step
trainX_timestep = numpy.reshape(trainX_timestep, (trainX_timestep.shape[0], trainX_timestep.shape[1], 1))
testX_timestep = numpy.reshape(testX_timestep, (testX_timestep.shape[0], testX_timestep.shape[1], 1))

print('reshape entrenamientox time_step : ' + str(trainX_timestep))
print('Con shape')
print(trainX_timestep.shape)
print('reshape testx time_step: ' + str(testX_timestep))

# create and fit the LSTM network
model = Sequential()
#model.add(LSTM(4, input_shape=(look_back,1)))

# first model layer code with memory
batch_size = 1
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print('Datos que va a usar la red neuronal: \n')
print('trainX: \n')
#print(trainX)
print(trainX_timestep)
print('trainY: \n')
#print(trainY)
print(trainY_timestep)
#model.fit(trainX_timestep, trainY_timestep, epochs=100, batch_size=1, verbose=2)

# fit code with memory
for i in range(300):
    model.fit(trainX_timestep, trainY_timestep, epochs=1, batch_size=batch_size, verbose=2, shuffle=False)
    model.reset_states()

# make predictions
#trainPredict = model.predict(trainX_timestep)
#print('La prediccion de la red neuronal para los datos de entrenamiento: \n' + str(trainPredict))
#testPredict = model.predict(testX_timestep)
#print('La prediccion de la red neuronal para los datos de test: \n' + str(testPredict))

# make predictions with memory
trainPredict = model.predict(trainX_timestep, batch_size=batch_size)
print('La prediccion de la red neuronal para los datos de entrenamiento: \n' + str(trainPredict))
model.reset_states()
testPredict = model.predict(testX_timestep, batch_size=batch_size)
print('La prediccion de la red neuronal para los datos de test: \n' + str(testPredict))


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
print('La prediccion de la red neuronal para los datos de entrenamiento INVERTIDOS: \n' + str(trainPredict))
trainY_timestep = scaler.inverse_transform([trainY_timestep])
testPredict = scaler.inverse_transform(testPredict)
print('La prediccion de la red neuronal para los datos de test INVERTLDOS: \n' + str(testPredict))
testY_timestep = scaler.inverse_transform([testY_timestep])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY_timestep[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY_timestep[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()