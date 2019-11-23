from pandas import read_csv
from matplotlib import pyplot



import pandas as pd
from pandas import DataFrame
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = read_csv('STL_prueba_rendimiento.csv',header=0, index_col=0)
#series = read_csv('datos_aceituna_gilena.csv', usecols=[0,3], engine='python', index_col=0)
seriesdf = DataFrame(series)
seriesdf.reset_index(inplace=True)
seriesdf['FECHA'] = pd.to_datetime(seriesdf['FECHA'])
seriesdf = seriesdf.set_index('FECHA')

result = seasonal_decompose(seriesdf.values, model='multiplicative', freq=30)
#result = seasonal_decompose(seriesdf, model='multiplicative') freq no necesario si se le pasa un pandas| probar multiplicative con freq
result.plot()
pyplot.show()


'''
from statsmodels.tsa.seasonal import seasonal_decompose
series = ...
result = seasonal_decompose(series, model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)


aa = result.trend
aaa = aa[15:]
aaaa = aaa[0:7537]
aaaadf = DataFrame(aaaa)
aaaadf.to_csv('trend_STL_rendimiento.csv')
'''


