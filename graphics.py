import matplotlib.pyplot as plt
import pandas






rmse_train = pandas.read_csv('RMSE-TRAIN.csv', usecols=[1], engine='python')
rmse_test = pandas.read_csv('RMSE-TEST.csv', usecols=[1], engine='python')

print(rmse_train.values)
plt.plot(rmse_train.values)
plt.plot(rmse_test.values)
plt.show()