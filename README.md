Sistema de Recomendación Musical basado en MFCC

Este proyecto implementa un sistema de recomendación musical utilizando características acústicas extraídas de archivos de audio mediante MFCC y técnicas de agrupamiento con K-Means.

Funcionalidades

- Extracción automática de MFCC desde archivos `.mp3` o `.mpeg`.
- Clustering acústico con KMeans (scikit-learn).
- Inserción de nuevos audios y procesamiento automático.
- Recomendaciones usando distancias: coseno, manhattan o euclidiana.
- Unión con metadatos musicales desde un archivo CSV extendido.

Estructura del Proyecto

<pre>
.
├── dataset/
│   └── spotify_songs_download.csv
├── utils/
│   ├── scaler.joblib
│   ├── Kmeans.joblib
│   └── histogramas_acusticos.json
├── audios_temp/
├── audios_wav/
├── process/
│   └── extract_audio_functions.py
├── proyect_audio.ipynb
└── requirements.txt
</pre>

Instalación de Dependencias

Instalar todas las librerías necesarias:
```bash
pip install -r requirements.txt
```
Dependencias clave:

- librosa
- numpy
- pandas
- scikit-learn
- joblib

Uso

Verificación del entorno

Antes de iniciar, asegúrese de tener:

- ./dataset/spotify_songs_download.csv
- ./utils/scaler.joblib
- ./utils/Kmeans.joblib
- ./utils/histogramas_acusticos.json
- ./process/extract_audio_functions.py

El sistema validará estos recursos al inicio.

Recomendación por archivo de audio

audio = "C:/ruta/archivo.mpeg"
recomendaciones = obtener_recomendaciones_por_audio_mp3(audio, k=5, tipo="coseno")

Recomendación por ID

recomendaciones = obtener_recomendaciones_por_song_id(2913, tipo="manhattan", k=5)

Insertar nuevo audio

id = max_key("./utils/histogramas_acusticos.json") + 1
insert_audio("C:/ruta/archivo.mpeg", id)

Este proceso:
- Convierte el archivo a .wav.
- Extrae MFCC.
- Genera el histograma acústico.
- Inserta en el JSON y el CSV base.

Unir con metadatos extendidos

df = pd.read_csv("./dataset/spotify_songs_download.csv")
df['id'] = df['id'].astype(str)
filas = df[df['id'].isin(recomendaciones)]
filas['score'] = filas['id'].map(recomendaciones)

df_genre = pd.read_csv('./dataset/spotify_songs_download_FINAL.csv')
df_genre['id'] = df_genre['id'].astype(str)
join_data = pd.merge(filas, df_genre, on='id', how='left')

Consideraciones

- Se recomienda usar audios de buena calidad (mínimo 30 segundos).
- El sistema está preparado para insertar nuevos audios sin afectar los registros previos.
- La métrica de similitud puede cambiarse según el enfoque deseado.

Créditos

Este proyecto fue desarrollado como parte del curso de Base de Datos 2 (BD2).

Trabajo realizado en conjunto por el equipo de estudiantes, en colaboración y coordinación activa durante todo el proceso.
