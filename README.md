# Proyecto facenet

## Requerimientos técnicos.

Los requerimientos para trabajar con facenet son:

1. Python
2. opencv-python==3.4.0.12

Puede utilizar las siguientes guias de referencia:
*[Instalar python y pip](https://github.com/BurntSushi/nfldb/wiki/Python-&-pip-Windows-installation)


## 1. Antes de utilizar Facenet

Para reconocer rostros debemos alimentar Facenet con un conjunto de imagenes con el siguiente formato.

```
Folder
	Imagen_1.jpg
	Imagen_2.jpg
	Imagen_n.jpg
```

### Formato de carpetas y nombres de imagenes

#### Formato para el nombre de la carpeta: 

Para identificar una carpeta se deben utilizar dos identificadores, para nuestro ejemplo usamos el nombre y el apellido de una persona.  El nombre y el apellido deben iniciar con mayuscula y deben estar unidos por un guion bajo.  Por Ejemplo:

```
	Jorge_Santos
	JorgeSalvador_SantosNeill
```

Los dos ejemplos anteriores son validos porque inician con un caracterer en mayuscula y estan separados por guion bajo.


#### Formato para el nombre de cada imagen:

Cada imagen debe tener un formato al igual que el nombre de las carpetas.   Si las imagenes son generadas con un proceso manual, el nombre de la imagen debe coincidir con el el nombre de la carpeta.  Es decir, si el nombre de la carpeta es Jorge_Santos, el nombre de las imagenes deberia de ser:

```
	Jorge_Santos_1.jpg
	Jorge_Santos_2.jpg
	Jorge_Santos_3.jpg
```

Puede notar que el el nombre se compone de tres partes, las primeras dos partes es el nombre y apellido de la persona y la tercera parte representa el numero de la imagen.   Todas las imagenes deben tener el formato "jpg"

Si las imagenes son generadas generadas automaticamente desde una camara, debe asegurarse de que las imagenes se guarden con el formato especificado.

#### Un ejemplo de una carpeta con sus imagenes

El resultado de un ejemplo completo de una carpeta y sus imagenes se describe en esta seccion.   Dado un ejemplo que se requiere guardar 5 imagenes de la persona jorge santos, el resultado deberia ser el siguiente:

```
Jorge_Santos
	Jorge_Santos_1.jpg
	Jorge_Santos_2.jpg
	Jorge_Santos_3.jpg
```



### Pasos para capturar imagenes por medio de la camara.

1. Crear una carpeta con nombre "imagenesDeEntrada" en el directorio "facenet/files/lfw/"

2. Abrir el archivo openCamera.py con un editor de texto, esta ubicado en el directorio "facenet/files/data"

3.  Cambiar el contenido de la variable "name" con el nombre de la persona de la que queremos capturar un conjunto de imagenes. El contenido de la variable debe tener el formato "Nombre_Apellido"

4. Cambiar el contenido de la variable "directorioDeImagenes" por "..\lfw\imagenesDeEntrada"

5. Ejecutar el archivo "openCamera.py".  Para ejecutar el archivo debemos estar ubicados en la consola de comandos en el directorio "facenet/files/data" y ejecutar el comando siguiente

```
	py openCamera.py
```






# Install

# Run

## 1. Set environment variable to use facenet

Ubuntu
```
export PYTHONPATH=/notebooks/src
```

Windows
```
set PYTHONPATH=C:\Users\jorge\repository\facenet\files\src
```

## 2. Align data

### Original Example

Ubuntu 
```
for N in {1..4}; do python src/align/align_dataset_mtcnn.py lfw/raw lfw/lfw_mtcnnpy_160 --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done
```
Note: remove for N in {1..4}; do and & done in the before code example


### Custom Example

Ubuntu
```
for N in {1..4}; do python src/align/align_dataset_mtcnn.py lfw/input lfw/output --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25 & done
```

Note: remove for N in {1..4}; do and & done in the before code example


## 2. Test Example

### Original Example

Ubuntu & Windows
```
python src/validate_on_lfw.py lfw/output models/20180402-114759 --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization --lfw_pair data/pairs190.txt  --lfw_batch_size 236
```

## 3. Train a classifier on LFW (Custom Example)

### Custom Example

Ubuntu & Windows
```
python src/validate_on_lfw.py lfw/output models/20180402-114759 --distance_metric 1 --use_flipped_images --subtract_mean --use_fixed_image_standardization --lfw_pair data/pairs.txt  --lfw_batch_size 44
```

# Documents

* [Google news](https://www.unocero.com/noticias/ciencia/google-nuestro-sistema-de-reconocimiento-de-rostros-es-el-mejor/)
* [facenet documentarion](https://arxiv.org/pdf/1503.03832.pdf)
* [KNN Algoritm](https://www.analiticaweb.es/algoritmo-knn-modelado-datos/)
* [K-means algoritm](https://es.wikipedia.org/wiki/K-means)

