import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

# Red neuronal básica para la predicción de salidas en compuertas XOR.
# 2 nodos de entrada, 16 capa intermedia y 1 nodo de salida.

# Cargamos las 4 combinaciones de las compuertas XOR
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# y estos son los reultados que se obtienen, en el mismo orden
target_data = np.array([[0],[1],[1],[0]], "float32")

# Crea el modelo vacío y va añadiendo capas.
# En la primera indicamos el número de nodos y las entradas, así como su función de activación.
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Se compila indicando función de pérdida, optimización de pesos y métrica.
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

# Número de iteraciónes.
model.fit(training_data, target_data, epochs=1000)

# Evaluamos el modelo.
scores = model.evaluate(training_data, target_data)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(model.predict(training_data).round())


# Para guardar el modelo:

# Serializar el modelo a JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
        json_file.write(model_json)

# Serializar los pesos a HDF5
model.save_weights("model.h5")
print("Modelo Guardado!")

# Vemos en otro archivo cómo se carga este modelo guardado
