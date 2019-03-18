import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers

# num_words=10000 para quedarnos solo con las 10000 palabras que más aparezcan en las reviews (+ comunes)
# train_data y test_data son listas de reviews ya pasadas a enteros.
# train_labels y test_labels son listas de 0s y 1s que se corresponden con la valoración de la película
# 0 mala, 1 buena
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# probar train_data[0] y train_labels[0] para ver la forma

# El diccionario que traduce de palabras a enteros
word_index = imdb.get_word_index()
# Lo invertimos para hacer la traducción opuesta
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# En este diccionario en concreto hay que meter un offset de 3 porque 0,1 y 2 son índices reservados.
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[1]])

# probar decoded_review para ver una review tras hacerle la inversión


def vectorize_sequences(sequences, dimension=10000):
    # Creamos una matriz de tamaño (tamaño de sequences, dimension) inicializada con ceros
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        print(sequence)
        results[i, sequence] = 1. # cambia el indice especificado en results[i] a 1
    return results


# Usamos la función de arriba para pasar las reviews a una matriz de unos y ceros procesable para una red neuronal
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Pasamos también los resultados de las reviews de una lista a un array
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels, 'float32')


# Creamos el modelo vacío y añadimos las capas
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Datos de 'validación' para comprobar la eficacia de la red con nuevos ejemplos
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#history = model.fit(partial_x_train,
#                    partial_y_train,
 #                   epochs=20,
  #                  batch_size=512,
   #                 validation_data=(x_val, y_val))
model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=512)
history = model.evaluate(x_val, y_val)



