import pandas as pd
from pandas import DataFrame
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = read_csv('files/STL_prueba_acidez.csv',header=0, index_col=0)
#series = read_csv('files/datos_aceituna_gilena.csv', usecols=[0,3], engine='python', index_col=0)
seriesdf = DataFrame(series)
seriesdf.reset_index(inplace=True)
seriesdf['FECHA'] = pd.to_datetime(seriesdf['FECHA'])
seriesdf = seriesdf.set_index('FECHA')

result = seasonal_decompose(seriesdf.values, model='multiplicative', freq=30)
#result = seasonal_decompose(seriesdf, model='multiplicative') freq no necesario si se le pasa un pandas
result.plot()
pyplot.show()


'''
series = read_csv('files/datos_aceituna_gilena.csv', usecols=[0,3], engine='python', index_col=0)
seriesdf = DataFrame(series)
seriesdf.reset_index(inplace=True)
seriesdf['FECHA'] = pd.to_datetime(seriesdf['FECHA'])
seriesdf = seriesdf.set_index('FECHA')
#Añadir comillas y borrar hora con crtl+r


result = seasonal_decompose(series, model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)


aa = result.trend
#na
aaaadf = DataFrame(aa)
aaaadf.to_csv('files/trend_STL_rendimiento.csv')
'''


