# FUNCIONES PARA EXTRACTION DE AUDIO
import matplotlib.pyplot as plt
import librosa.display 
import librosa
import numpy as np
from sklearn.cluster import KMeans
import os
import json

# Convertir a input los audios
from pydub import AudioSegment

# calculos matematicos
import math
import heapq

def transform_mp3_to_wav(file_path_import, file_path_export,tiempo_recorte=30):
    """
    Convierte los primeros 30 segundos de un archivo MP3 a WAV mono 22050 Hz.
    
    Parámetros:
      file_path_import: Ruta completa del archivo MP3.
      file_path_export: Ruta completa del archivo WAV. (para guardar)
      tiempo_recorte  : Tiempo de recorte en (Segundos)
    
    """
    
    # Construir la ruta de salida
    input_path = file_path_import
    output_path = file_path_export

    # Cargar archivo MP3
    audio = AudioSegment.from_file(input_path, format="mp3")

    # Cortar los primeros 30 segundos (30 * 1000 milisegundos)
    audio = audio[:tiempo_recorte*1000]  

    # Convertir a WAV (mono, 22050 Hz)
    audio = audio.set_frame_rate(22050).set_channels(1)
    audio.export(output_path, format="wav")

    print(f"Convertido: {input_path} → {output_path}")

def mp3_to_wav(file_path_import,tiempo_recorte=30):
    """
    Convierte los primeros 30 segundos de un archivo MP3 a WAV mono 22050 Hz.
    
    Parámetros:
      file_path_import: Ruta completa del archivo MP3.
      tiempo_recorte  : Tiempo de recorte en (Segundos)
    Return    :
      aduio
    """
    
    # Construir la ruta de salida
    input_path = file_path_import

    # Cargar archivo MP3
    audio = AudioSegment.from_file(input_path, format="mp3")

    # Cortar los primeros 30 segundos (30 * 1000 milisegundos)
    audio = audio[:tiempo_recorte*1000]  

    # Convertir a WAV (mono, 22050 Hz)
    audio = audio.set_frame_rate(22050).set_channels(1)
    return audio



# Obtener el mfcc
def extract_mfcc(file_path_wav, n_mfcc=13,ventana=0.5,duracion=30,out=False):
    """
    Extrae las características MFCC de un archivo de audio.

    Parámetros:
        file_path_wave: Ruta al archivo .wav
        n_mfcc        : Número de coeficientes MFCC a extraer por frame 
        ventana       : tomar de captura del frame en el tiempo en una ventana s(Segundos) 
        duracion      : cuanto tiempo del audio se procesara
        out           : salida para ver el debug (FALSE)

    Retorna:
        np.ndarray    : Matriz (frames, n_mfcc)
    """

    y, sr = librosa.load(file_path_wav, duration=duracion) 
    if out:
        print("Forma de onda (y):", y)
        print("Frecuencia de muestreo (sr):", sr)

    # hop_length y n_fft ajustados a una ventana de 0.5 segundos
    hop = int(sr * ventana)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop, n_fft=hop)

    return mfcc.T  # Transpuesta: filas = frames, columnas = coeficientes MFCC



# Sacar los codewords -> kmeans y el centroide es cada codeword 
def construir_codebook(features, n_clusters=256,out=False):
    """
    Aplica K-Means para construir el diccionario acústico (codebook).

    Parámetros:
        features     : vector de caracteristicas del mfcc apilados.
        n_clusters   : numero de clusteres que se usara para entrenar
    Retorna   :
        Kmeans       : el modelo entrenado con los features
    """
    if out:
        print(f"Entrenando K-Means con {n_clusters} clusters sobre {features[0].shape[0]} vectores...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, verbose=1)
    kmeans.fit(features)
    if out:
        print("Entrenamiento completado.")
    return kmeans



def histogram_audio(mfcc_features_audio, kmeans_model):
    """
    Representa solo un audio como histograma de frecuencias de palabras acústicas.
    
    Parámetros:
        mfcc_features_audio: vector de caracteristicas del mfcc por audio
        kmeans_model       : el modelo Kmeans entrenado con el mfcc

    Retorna:
        hist         : el histograma normalizado de cada audio
    """
    #cada frame es asignado a un cluster
    labels = kmeans_model.predict(mfcc_features_audio) 
    #Cuenta cuántas veces aparece cada número (label/cluster)
    hist, _ = np.histogram(labels, bins=np.arange(kmeans_model.n_clusters + 1))
    hist = hist / np.linalg.norm(hist)  # Normalizar histograma
    return hist



def guardar_json(diccionario, ruta_salida="histogramas_acusticos.json"):
    """
    Guarda un diccionario como archivo JSON.
    """
    with open(ruta_salida, "w") as f:
        json.dump(diccionario, f, indent=4)
    print(f"Histogramas guardados en: {ruta_salida}")

from joblib import dump,load

def guardar_objeto(objeto, ruta):
    """
    Guarda cualquier objeto serializable con joblib.
    
    Parámetros:
        objeto : objeto Python a guardar (modelo, scaler, vector, etc.)
        ruta   : ruta del archivo a guardar, por ejemplo 'modelo.joblib'
    """
    dump(objeto, ruta)

def cargar_objeto(ruta_base):
    """
    Carga el objeto

    Parámetros:
        ruta_base: prefijo del archivo 
    """
    objeto=load(ruta_base)
    return objeto

# -----------GRAFICAS ------------
def show_mfcc(mfcc_features,title="Coeficiente MFCC 1 a lo largo del tiempo"):
    # Supón que ya tienes `mfcc` con shape (1292, 13)
    plt.plot(mfcc_features[:, 0])  # Graficar solo el coeficiente 1
    plt.title(title)
    plt.xlabel("Frames (1s por frame)")
    plt.ylabel("Valor del coeficiente")
    plt.show()

## --------------------------------------------------------------------------------
# Testing 

if __name__ == "__main__":
    output_path = "/home/jazmin/Escritorio/Database II/Recomendador_musica/fma_small_wave/000"
    name_file = "000002"
    #transform_mp3_to_wav("/home/jazmin/Escritorio/Database II/Recomendador_musica/fma_small/000", name_file, output_path)

    # Extraer
    output_wave = os.path.join(output_path, f"{name_file}.wav")
    mfcc_matrix = extract_mfcc(output_wave)

    print(mfcc_matrix.shape) # Para ver que cumple con el tamaño de ventanas x numero de coeficientes 

    # Mostrar
    show_mfcc(mfcc_matrix)




# Aplicar knn con similitud de coseno

import numpy as np

def distancia_euclidiana(val1, val2):
    
    return np.linalg.norm(np.array(val1) - np.array(val2))

def knn_lineal(query, k):
    """
        fijo: ruta del codebook en formato json -> dict: {nombre_archivo_id: [valores del histograma]}
        input: query-> en formato hist o mp3
                k -> cuantos similares retorno
        output: retorna los  k mas cercanos

        PASOS: 
            1. Cargar el codebook
            2. Tener la distancia 
            
    """
    
    with open("histogramas_acusticos.json") as f:
        codebook = json.load(f)

     # 2. Cola de prioridad (simulamos max-heap con -distancia)
    heap = []  # formato: (-distancia, audio_id)
    resultados = []

    for audio_id, hist in codebook.items():

        hist=list(map(float,hist)) # casteo a list float
        dist = distancia_euclidiana(query, hist)
        print("dist:",dist)
        heapq.heappush(heap, (dist, audio_id))

    
    for _ in range(k):
        dist, audio_id = heapq.heappop(heap)
        resultados.append((audio_id, dist))

    return resultados

# aplicar knn con tf idf

def cosine_similarity(v1, v2):
    """
    Retorna la similitud de coseno entre dos vectores.
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def knn_cosine(query, k):
    """
    Retorna los k elementos más similares al query usando similitud de coseno,
    solo con heap (sin ordenar al final).
    """
    with open("histogramas_acusticos.json") as f:
        codebook = json.load(f)

    heap = []  # max-heap

    for audio_id, hist in codebook.items():    #track_id :[ hist ]
        hist=list(map(float,hist))
        sim = cosine_similarity(query, hist)
        heapq.heappush(heap, (-sim, audio_id))  # max-heap simulada con -sim

    resultados = []
    for _ in range(min(k, len(heap))):
        neg_sim, audio_id = heapq.heappop(heap)
        sim = -neg_sim  # revertimos el negativo
        resultados.append((audio_id, sim))

    return resultados

# Con los resultados del knn hacemos calculo de como conectamos para obtener musica lyric, etc etc. 

import heapq
import json

def distancia_manhattan(v1, v2):
    """
    Calcula la distancia Manhattan (L1) entre dos vectores.
    """
    return sum(abs(a - b) for a, b in zip(v1, v2))

def knn_manhattan(query, k):
    """
    Retorna los k elementos más cercanos al query usando distancia Manhattan (L1).
    """
    with open("histogramas_acusticos.json") as f:
        codebook = json.load(f)

    heap = []  # min-heap por defecto en Python

    for audio_id, hist in codebook.items():
        hist = list(map(float, hist))  # Asegurarse de que sean floats
        dist = distancia_manhattan(query, hist)
        heapq.heappush(heap, (dist, audio_id))

    resultados = []
    for _ in range(min(k, len(heap))):
        dist, audio_id = heapq.heappop(heap)
        resultados.append((audio_id, dist))

    return resultados




