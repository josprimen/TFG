'''import numpy as np
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





datos=read_csv('files/datos_aceituna_gilena.csv', usecols=[0,3], engine='python')
datoss = datos.values
anyos= ['2015', '2016', '2017', '2018', '2019']
meses = ['01','10','11','12']
dias = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
suma = []
suma2015 = []
suma2016 = []
suma2017 = []
suma2018 = []
suma2019 = []
sumaall = []


for anyo in anyos:
    for mes in meses:
        for dia in dias:
            suma_dia = 0
            for d in datoss:
                if (dia+'/'+mes+'/'+anyo) in d[0]:
                    print('Fecha: ' + d[0] + ' Rendimiento: ' + str(d[1]))
                    suma_dia = suma_dia + d[1]
            sumaall.append(suma_dia)
            if anyo == '2015':
                suma2015.append(suma_dia)
            if anyo == '2016':
                suma2016.append(suma_dia)
            if anyo == '2017':
                suma2017.append(suma_dia)
            if anyo == '2018':
                suma2018.append(suma_dia)
            if anyo == '2019':
                suma2019.append(suma_dia)

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


suma_anyos = [suma2015, suma2016, suma2017, suma2018, suma2019]
aux = 1
pyplot.figure()
for result in suma_anyos:
    pyplot.subplot(5, 1, aux)
    pyplot.plot(result)
    aux = aux+1

pyplot.show()




suma2015df = DataFrame(suma2015)
suma2015df = suma2015df.loc[~(suma2015df==0).all(axis=1)]
#suma2015df.to_csv('files/media_rendimiento_dias_2015.csv')
suma2015df = suma2015df.values
suma2016df = DataFrame(suma2016)
suma2016df = suma2016df.loc[~(suma2016df==0).all(axis=1)]
#suma2016df.to_csv('files/media_rendimiento_dias_2016.csv')
suma2016df = suma2016df.values
suma2017df = DataFrame(suma2017)
suma2017df = suma2017df.loc[~(suma2017df==0).all(axis=1)]
#suma2017df.to_csv('files/media_rendimiento_dias_2017.csv')
suma2017df = suma2017df.values
suma2018df = DataFrame(suma2018)
suma2018df = suma2018df.loc[~(suma2018df==0).all(axis=1)]
#suma2018df.to_csv('files/media_rendimiento_dias_2018.csv')
suma2018df = suma2018df.values
suma2019df = DataFrame(suma2019)
suma2019df = suma2019df.loc[~(suma2019df==0).all(axis=1)]
#suma2019df.to_csv('files/media_rendimiento_dias_2019.csv')
suma2019df = suma2019df.values



print('Suma 2015 Drop zeros: ')
print(suma2015df)
print('len')
print(len(suma2015df))
print('\n')

print('Suma 2016 Drop zeros: ')
print(suma2016df)
print('len')
print(len(suma2016df))
print('\n')

print('Suma 2017 Drop zeros: ')
print(suma2017df)
print('len')
print(len(suma2017df))
print('\n')

print('Suma 2018 Drop zeros: ')
print(suma2018df)
print('len')
print(len(suma2018df))
print('\n')

print('Suma 2019 Drop zeros: ')
print(suma2019df)
print('len')
print(len(suma2019df))
print('\n')

suma_anyos_df = [suma2015df, suma2016df, suma2017df, suma2018df, suma2019df]
aux = 1
pyplot.figure()
for result in suma_anyos_df:
    pyplot.subplot(5, 1, aux)
    pyplot.plot(result)
    aux = aux+1

pyplot.show()



datos_rendimiento = np.concatenate((suma2015df, suma2016df, suma2017df, suma2018df, suma2019df))
print('Datos rendimiento:')
print(datos_rendimiento)



datos = datos_rendimiento

pyplot.plot(datos[:])
pyplot.show()

minmax = MinMaxScaler(feature_range=(0,1))
conjunto = minmax.fit_transform(datos)

tamaño_entrenamiento = int(len(conjunto) * 0.67)
tamaño_test = len(conjunto) - tamaño_entrenamiento
entrenamiento = conjunto[0:tamaño_entrenamiento]
test = conjunto[tamaño_entrenamiento:len(conjunto)]

def datosX (conjunto):
    aux = []
    for i in range (len(conjunto)-2):
        a = [conjunto[i]]
        aux.append(a)
    return np.array(aux)

def datosY (conjunto):
    aux = []
    for i in range (len(conjunto)-2):
        aux.append(conjunto[i+1,0])
    return np.array(aux)


entrenamientoX = datosX(entrenamiento)
entrenamientoY = datosY(entrenamiento)

pruebasX = datosX(test)
pruebasY = datosY(test)


paso_atras = 1 #de cúantas unidades de tiempo son los pasos
model = Sequential()
model.add(LSTM(4, input_shape=(1, paso_atras)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

history = model.fit(entrenamientoX, entrenamientoY, epochs=25, validation_split=0.3, batch_size=1, verbose=2)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

trainPredict = model.predict(entrenamientoX)
testPredict = model.predict(pruebasX)
trainPredict = minmax.inverse_transform(trainPredict)
trainY = minmax.inverse_transform([entrenamientoY])
testPredict = minmax.inverse_transform(testPredict)
testY = minmax.inverse_transform([pruebasY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(conjunto)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[paso_atras:len(trainPredict)+paso_atras, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(conjunto)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(paso_atras*2)+1:len(conjunto)-1, :] = testPredict
# plot baseline and predictions
pyplot.plot(minmax.inverse_transform(conjunto))
pyplot.plot(trainPredictPlot)
pyplot.plot(testPredictPlot)
pyplot.show()












'''
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
from keras.callbacks import EarlyStopping





datos=read_csv('files/datos_aceituna_gilena.csv', usecols=[0,2,3], engine='python')
datoss = datos.values
kilos = read_csv('files/datos_aceituna_gilena.csv', usecols=[2], engine='python')
kilos = kilos.values
#MOD? RANGO DE 1 A 2
minmax = MinMaxScaler(feature_range=(0,1))
kilos_normalizados = minmax.fit_transform(kilos)
datoss[:,1] = kilos_normalizados[:,0]

anyos= ['2015', '2016', '2017', '2018', '2019']
meses = ['01','10','11','12']
dias = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
suma = []
suma2015 = []
suma2016 = []
suma2017 = []
suma2018 = []
suma2019 = []
sumaall = []


for anyo in anyos:
    for mes in meses:
        for dia in dias:
            suma_dia = 0
            numero_albaran = 0
            for d in datoss:
                if (dia+'/'+mes+'/'+anyo) in d[0]:
                    print('Fecha: ' + d[0] + ' Rendimiento: ' + str(d[2]))
                    suma_dia = suma_dia + d[1]*d[2]
            sumaall.append(suma_dia)
            if anyo == '2015':
              if numero_albaran==0:
                suma2015.append(suma_dia)
              else:
                suma2015.append(suma_dia/numero_albaran)
            if anyo == '2016':
              if numero_albaran==0:
                suma2016.append(suma_dia)
              else:
                suma2016.append(suma_dia/numero_albaran)
            if anyo == '2017':
              if numero_albaran==0:
                suma2017.append(suma_dia)
              else:
                suma2017.append(suma_dia/numero_albaran)
            if anyo == '2018':
              if numero_albaran==0:
                suma2018.append(suma_dia)
              else:
                suma2018.append(suma_dia/numero_albaran)
            if anyo == '2019':
              if numero_albaran==0:
                suma2019.append(suma_dia)
              else:
                suma2019.append(suma_dia/numero_albaran)

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


suma_anyos = [suma2015, suma2016, suma2017, suma2018, suma2019]
aux = 1
pyplot.figure()
for result in suma_anyos:
    pyplot.subplot(5, 1, aux)
    pyplot.plot(result)
    aux = aux+1

pyplot.show()




suma2015df = DataFrame(suma2015)
suma2015df = suma2015df.loc[~(suma2015df==0).all(axis=1)]
#suma2015df.to_csv('files/media_rendimiento_dias_2015.csv')
suma2015df = suma2015df.values
suma2016df = DataFrame(suma2016)
suma2016df = suma2016df.loc[~(suma2016df==0).all(axis=1)]
#suma2016df.to_csv('files/media_rendimiento_dias_2016.csv')
suma2016df = suma2016df.values
suma2017df = DataFrame(suma2017)
suma2017df = suma2017df.loc[~(suma2017df==0).all(axis=1)]
#suma2017df.to_csv('files/media_rendimiento_dias_2017.csv')
suma2017df = suma2017df.values
suma2018df = DataFrame(suma2018)
suma2018df = suma2018df.loc[~(suma2018df==0).all(axis=1)]
#suma2018df.to_csv('files/media_rendimiento_dias_2018.csv')
suma2018df = suma2018df.values
suma2019df = DataFrame(suma2019)
suma2019df = suma2019df.loc[~(suma2019df==0).all(axis=1)]
#suma2019df.to_csv('files/media_rendimiento_dias_2019.csv')
suma2019df = suma2019df.values



print('Suma 2015 Drop zeros: ')
print(suma2015df)
print('len')
print(len(suma2015df))
print('\n')

print('Suma 2016 Drop zeros: ')
print(suma2016df)
print('len')
print(len(suma2016df))
print('\n')

print('Suma 2017 Drop zeros: ')
print(suma2017df)
print('len')
print(len(suma2017df))
print('\n')

print('Suma 2018 Drop zeros: ')
print(suma2018df)
print('len')
print(len(suma2018df))
print('\n')

print('Suma 2019 Drop zeros: ')
print(suma2019df)
print('len')
print(len(suma2019df))
print('\n')

suma_anyos_df = [suma2015df, suma2016df, suma2017df, suma2018df, suma2019df]
aux = 1
pyplot.figure()
for result in suma_anyos_df:
    pyplot.subplot(5, 1, aux)
    pyplot.plot(result)
    aux = aux+1

pyplot.show()



datos_rendimiento = np.concatenate((suma2015df, suma2016df, suma2017df, suma2018df, suma2019df))
print('Datos rendimiento:')
print(datos_rendimiento)


csv_media_ponderada_rendimiento = DataFrame(datos_rendimiento)
csv_media_ponderada_rendimiento.to_csv('files/csv_media_ponderada_rendimiento.csv')
datos = datos_rendimiento

pyplot.plot(datos[:])
pyplot.show()

minmax = MinMaxScaler(feature_range=(0,1))
conjunto = minmax.fit_transform(datos)

tamaño_entrenamiento = int(len(conjunto) * 0.67)
tamaño_test = len(conjunto) - tamaño_entrenamiento
entrenamiento = conjunto[0:tamaño_entrenamiento]
test = conjunto[tamaño_entrenamiento:len(conjunto)]

def datosX (conjunto):
    aux = []
    for i in range (len(conjunto)-2):
        a = [conjunto[i]]
        aux.append(a)
    return np.array(aux)

def datosY (conjunto):
    aux = []
    for i in range (len(conjunto)-2):
        aux.append(conjunto[i+1,0])
    return np.array(aux)


entrenamientoX = datosX(entrenamiento)
entrenamientoY = datosY(entrenamiento)

pruebasX = datosX(test)
pruebasY = datosY(test)


paso_atras = 1 #de cúantas unidades de tiempo son los pasos
model = Sequential()
model.add(LSTM(4, input_shape=(1, paso_atras)))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, min_delta=0.001)

history = model.fit(entrenamientoX, entrenamientoY, epochs=50, validation_split=0.3, batch_size=1, verbose=2, callbacks=[es])

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

trainPredict = model.predict(entrenamientoX)
testPredict = model.predict(pruebasX)
trainPredict = minmax.inverse_transform(trainPredict)
trainY = minmax.inverse_transform([entrenamientoY])
testPredict = minmax.inverse_transform(testPredict)
testY = minmax.inverse_transform([pruebasY])

trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(conjunto)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[paso_atras:len(trainPredict)+paso_atras, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(conjunto)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(paso_atras*2)+1:len(conjunto)-1, :] = testPredict
# plot baseline and predictions
pyplot.plot(minmax.inverse_transform(conjunto))
pyplot.plot(trainPredictPlot)
pyplot.plot(testPredictPlot)
pyplot.show()

