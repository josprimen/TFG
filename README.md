#Instalando el entorno

Instalar anaconda desde la web.
Instalar pycharm desde la web
Ejecutar el prompt de anaconda
Crear un entorno: conda create -n redes_neuronales python=3.6.8 (3.7 da conflicto con keras y tensorflow)
Activar el entorno: conda activate redes_neuronales
Instalar tensorflow en el entorno: conda install -c conda-forge tensorflow
Instalar keras en el entorno: conda install -c conda-forge keras
Abrir un proyecto en pycharm, file>settings>proyect interpreter>add interpeter>C:\Users\jose9\Anaconda3\envs\redes_neuronales\python.exe
<br />
PROBLEMAS/SOLUCIONES
SI SE QUEDA EN SOLVING ENVIROMENT USAR conda update conda o mirar el path
Al instalar keras error, se cierra el prompt
https://stackoverflow.com/questions/53483685/keras-breaks-anaconda-prompt
En Anaconda3 editar keras_activate.bat cambiar donde ponga 
>nul
por
1>
<br />
