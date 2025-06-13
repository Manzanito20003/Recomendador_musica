# 🎧 Extracción de Características de Audio con MFCC

Este proyecto en Python utiliza la biblioteca <code>librosa</code> para procesar archivos de audio y extraer características relevantes mediante los coeficientes cepstrales en las frecuencias de Mel (MFCC). Esta representación es ampliamente utilizada en tareas de clasificación de audio, recomendación musical, análisis acústico, entre otras aplicaciones en el campo del audio digital.

<hr/>

## 🎯 Objetivo

Obtener una representación numérica y compacta de archivos de audio, basada en MFCC, que pueda ser usada como entrada en modelos de Machine Learning o análisis exploratorio de datos musicales.

<hr/>

## 🧠 Función Principal

<h4><code>extract_mfcc(file_path, n_mfcc=13)</code></h4>

Extrae los coeficientes MFCC de los primeros 30 segundos de un archivo de audio.

<ul>
  <li><strong>file_path</strong>: ruta al archivo de audio.</li>
  <li><strong>n_mfcc</strong>: número de coeficientes por frame (por defecto 13).</li>
</ul>

<p>Este valor ofrece un equilibrio entre compacidad y nivel de detalle. Puedes ajustarlo según el objetivo del análisis:</p>

<ul>
  <li><code>1–10</code>: muy compacto, pierde detalles finos.</li>
  <li><code>13</code>: recomendado por su balance.</li>
  <li><code>20+</code>: más detallado pero más ruidoso y costoso computacionalmente.</li>
</ul>

<hr/>

## 📂 Estructura del Proyecto

<pre>
ANACONDA/
├── audio/                # Archivos de audio (.mpeg, .wav, etc.)
├── proyect_audio.ipynb   # Notebook principal con lógica de extracción
├── requirements.txt      # Lista de dependencias necesarias
└── .gitignore            # Archivos ignorados por Git
</pre>

<hr/>

## ⚙️ Instalación de Dependencias

Ejecuta el siguiente comando:

```bash
pip install -r requirements.txt
```
Paquetes necesarios
<ul> <li><code>librosa</code></li> <li><code>numpy</code></li> <li><code>jupyter</code> (para ejecutar el notebook)</li> </ul> <hr/>
Uso Básico
<ol> <li>Coloca tus archivos de audio dentro de la carpeta <code>audio/</code>.</li> <li>Abre <code>proyect_audio.ipynb</code> con Jupyter Notebook o VSCode.</li> <li>Ejecuta las celdas para cargar el audio, procesar y visualizar los coeficientes MFCC.</li> </ol> <hr/>
Recomendaciones
<ul> <li>Asegúrate de trabajar con archivos de audio de buena calidad para evitar errores de lectura.</li> <li>Si planeas usar los MFCC en modelos de aprendizaje automático, puedes normalizar los resultados o aplicar técnicas de reducción de dimensionalidad.</li> </ul> <hr/>
Créditos
Este proyecto fue desarrollado como parte de una práctica de análisis de audio para sistemas de recomendación musical, aprendizaje automático y procesamiento de señales.



### 📦 Descarga automática del dataset

Este proyecto usa el dataset `Feanix/gtzan-10-sec` desde Hugging Face.

No es necesario subirlo al repositorio. Para descargarlo automáticamente:

```python
from datasets import load_dataset
dataset = load_dataset("Feanix/gtzan-10-sec", cache_dir="./audio")



