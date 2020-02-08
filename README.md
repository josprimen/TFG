# Aplicación de redes Long Short-Term Memory para predicción de la calidad de la oliva en campaña
realizado por José Enrique Prieto Menacho como trabajo de fin de grado en I.Informática- TI en la Universidad de Sevilla

## INSTALANDO EL ENTORNO

-Instalar anaconda desde la web.<br />
-Instalar pycharm desde la web.<br />
-Ejecutar el prompt de anaconda. <br />
-Crear un entorno: conda create -n redes_neuronales python=3.6.8 (3.7 da conflicto con keras y tensorflow).<br />
-Activar el entorno: conda activate redes_neuronales.<br />
-Instalar tensorflow en el entorno: conda install -c conda-forge tensorflow.<br />
-Instalar keras en el entorno: conda install -c conda-forge keras.<br />
-Abrir un proyecto en pycharm, file>settings>proyect interpreter>add interpeter>C:\Users\jose9\Anaconda3\envs\redes_neuronales\python.exe (o ruta correspondiente)
<br /><br />

## PROBLEMAS/SOLUCIONES.<br />
-SI SE QUEDA EN SOLVING ENVIROMENT USAR conda update conda o mirar el path.<br />
-Al instalar keras error, se cierra el prompt-> [solución](https://stackoverflow.com/questions/53483685/keras-breaks-anaconda-prompt)<br />
-En Anaconda3 editar keras_activate.bat cambiar donde ponga <br />
`>nul`<br />
por<br />
`1>`
<br />


## ARCHIVOS Y FUNCIONAMIENTO
Una vez importado el proyecto a nuestro entorno simplemente hemos de ejecutar el archivo python que queramos.<br />
En "files" se encuentran los datos usados, tanto en bruto como una vez hecho el tratamiento de estos. <br /><br />
### ¿Qué hace cada fichero?<br />
#### Predicción con una variable<br />
- performance_prediction.py -> Predicción del rendimiento de la oliva usando rendimientos observados.
- acidity_prediction.py -> Predicción de la acidez de la oliva usando acideces observadas.
#### Predicción con múltiples variables: Dos variables<br />
- performance_acidity_prediction.py -> Predicción del rendimiento de la oliva usando rendimientos y acideces observadas.
- performance_humidity_prediction.py -> Predicción del rendimiento de la oliva usando rendimientos y humedad observadas.
- temp_precip_acidity.py -> Predicción de la acidez de la oliva usando datos de temperatura y precipitación.
#### Predicción usando la media ponderada de rendimiento y acidez de las cargas recibidas por días.<br />
- weighted_average_acidity_prediction.py -> Predicción de la acidez de la oliva usando las medias de las acideces observadas.
- weighted_average_performance_prediction.py -> Predicción del rendimiento de la oliva usando las medias de los rendimientos observados.
- weighted_average_performance_acidity_prediction.py -> Predicción del rendimiento de la oliva usando las medias de los rendimientos y acideces observadas.
#### Pruebas de predicción usando STL (Seasonal and Trend decomposition using Loess)<br />
- STL_prediction_acidity.py -> Predicción de la acidez usando los datos de la descomposición STL de la acidez.
- STL_prediction_performance.py -> Preddicción del rendimiento usando los datos de la descomposición STL del rendimiento.
- STL_weighted_average_acidity_prediction.py -> Descomposición STL de la acidez.
- STL_weighted_average_performance_prediction.py -> Descomposición STL del rendimiento.
#### Otros<br />
- json_to_csv.py -> Pasar datos de json (AEMET) a csv para su tratamiento.
- air_quality_example.py -> Ejemplo realizado para aprender.
### ¿En qué orden ejecuto?<br />
- 1º performance_prediction.py y acidity_prediction.py
- 2º performance_acidity_prediction.py, performance_humidity_prediction.py y temp_precip_acidity.py.
- 3º weighted_average_acidity_prediction.py y weighted_average_performance_prediction.py (Mejores resultados).
- 4º weighted_average_performance_acidity_prediction.py
- 5º STL_weighted_average_acidity_prediction.py y STL_weighted_average_performance_prediction.py
- 6º STL_prediction_acidity.py y STL_prediction_performance.py
