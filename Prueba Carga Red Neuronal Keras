import numpy as np
from keras.models import model_from_json

# cargar json y crear el modelo
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# cargar pesos al nuevo modelo
loaded_model.load_weights("model.h5")
print("Cargado modelo desde disco.")

# Compilar modelo cargado y listo para usar.
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['binary_accuracy'])

# Y usarlo.
# Simplemente habría que crear el cojunto de entrenamiento y el objetivo, como en la creación original.
# Lo cual nos obliga a importar numpy
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")

loaded_model.fit(training_data, target_data, epochs=1000)

# evaluamos el modelo
scores = loaded_model.evaluate(training_data, target_data)

# Ya que es un modelo que habíamos entrenado previamente,
# los resultados serán perfectos desde la primera iteración
print("\n%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1] * 100))
print(loaded_model.predict(training_data).round())
