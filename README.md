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
- performance_prediction.py
- acidity_prediction.py

<br />

### ¿En qué orden ejecuto?<br />
