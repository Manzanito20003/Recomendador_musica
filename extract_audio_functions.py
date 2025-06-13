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

def transform_mp3_to_wav(file_path_import, name_file, file_path_export):
    """
    Input:
        - file_path_import: recibe la ruta donde se encuentra el audio -> ejemplo: /home/jazmin/Escritorio/Database II/Recomendador_musica/fma_small/000 
        - name_file : es el nombre del archivo sin ninguna extension
        -- file_path_export: recibe la ruta donde se va a guardar el audio en .wav -> ejemplo: /home/jazmin/Escritorio/Database II/Recomendador_musica/fma_small_output/000 
    """

     # Construir rutas completas
    input_path = os.path.join(file_path_import, f"{name_file}.mpeg")
    output_path = os.path.join(file_path_export, f"{name_file}.wav")
    
    # Cargar archivo MP3
    audio = AudioSegment.from_file(input_path, format="mp3")

    # Convertir a WAV (mono, 22050 Hz)
    audio = audio.set_frame_rate(22050).set_channels(1)
    audio.export(output_path, format="wav")
    print(f"Convertido: de lna ruta {input_path} →  to : {output_path}")

def recortar_si_posible(input_path, output_path, inicio_s=60, duracion_s=10):
    audio = AudioSegment.from_wav(input_path)
    total_duracion = len(audio) / 1000  # en segundos

    # Si no alcanza para cortar desde 60s, recorta desde el inicio
    if total_duracion >= inicio_s + duracion_s:
        segmento = audio[inicio_s * 1000 : (inicio_s + duracion_s) * 1000]
    elif total_duracion >= duracion_s:
        segmento = audio[:duracion_s * 1000]
    else:
        segmento = audio  # recorta lo que haya (menos de 30s)
    
    segmento.export(output_path, format="wav")
    print(f"✅ Recortado: {output_path} ({len(segmento)/1000:.1f}s)")

def all_audio_transform_mp3_to_wav(dir_input, dir_output):
    """
    Recorre todos los .mp3 en el directorio_entrada y los convierte a .wav
    usando la función transform_mp3_to_wav.
    """
    os.makedirs(dir_output, exist_ok=True)
    nombres_convertidos = []

    for archivo in os.listdir(dir_input):
        if archivo.lower().endswith(".mp3"):
            nombre_sin_ext = os.path.splitext(archivo)[0]
            transform_mp3_to_wav(dir_input, nombre_sin_ext, dir_output)
            nombres_convertidos.append(nombre_sin_ext)
    
    return nombres_convertidos  # Por si luego quieres usar los .wav resultantes
    


# Obtener el mfcc


def extract_mfcc(file_path_wave, n_mfcc=13):
    """
    Extrae las características MFCC de un archivo de audio.

    Parámetros:
        file_path_wave (str): Ruta al archivo .wav
        n_mfcc (int): Número de coeficientes MFCC a extraer por frame

    Retorna:
        np.ndarray: Matriz (frames, n_mfcc) de coeficientes MFCC
    """
    y, sr = librosa.load(file_path_wave, duration=30)  # Carga los primeros 30s

    print("Forma de onda (y):", y)
    print("Frecuencia de muestreo (sr):", sr)

    # hop_length y n_fft ajustados a una ventana de 0.5 segundos
    hop = int(sr * 0.5)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop, n_fft=hop)

    return mfcc.T  # Transpuesta: filas = frames, columnas = coeficientes MFCC



def extract_mfcc_from_array(array_audio, sr=22050, n_mfcc=13 ,tiempo=0.5):
    """Extrae MFCCs desde un array de audio 1D."""
    
    # hop_length y n_fft ajustados a una ventana de 0.5 segundos
    hop = int(sr * tiempo)
    mfcc = librosa.feature.mfcc(y=array_audio, sr=sr, n_mfcc=n_mfcc,
                                hop_length=hop, n_fft=hop)
    return mfcc.T



# Unir los mfcc's de cada audio

def union_descriptors(lista_rutas_audio):
    """
        Input: Una lista con las rutas de audio (con todo y extension)
        Output : Conjunto de descriptores sacados del mfcc's
    """


    all_descriptors = []  # Lista de todos los frames MFCC de todos los audios

    for path in lista_rutas_audio:
        mfcc = extract_mfcc(path)  # (frames, n_mfcc)
        all_descriptors.append(mfcc)

    all_descriptors = np.vstack(all_descriptors)
    return all_descriptors 



# Sacar los codewords -> kmeans y el centroide es cada codeword 

def construir_codebook(features, n_clusters=256):
    """
    Aplica K-Means para construir el diccionario acústico (codebook).
    """
    print(f"Entrenando K-Means con {n_clusters} clusters sobre {features[0].shape[0]} vectores...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, verbose=1)
    kmeans.fit(features)
    print("Entrenamiento completado.")
    return kmeans



def histogram_audio(mfcc_features, kmeans_model):
    """
    Representa solo un audio como histograma de frecuencias de palabras acústicas.
    """
    labels = kmeans_model.predict(mfcc_features)  # Cluster asignado por frame
    print("labels:",labels)
    hist, _ = np.histogram(labels, bins=np.arange(kmeans_model.n_clusters + 1))
    print("hist_sin_nomrm",hist)
    hist = hist / np.linalg.norm(hist)  # Normalizar histograma
    print("hist_norm",hist)
    return hist



# FALTA TESTEAR (ESTA FUNCION)
def calcular_histogramas_todos_audios(folder_wav, kmeans_model, n_mfcc=13):
    """
    Procesa todos los archivos .wav y crea un diccionario de histogramas acústicos.
    
    Retorna:
        dict: {nombre_archivo: [valores del histograma]}
    """
    diccionario_histogramas = {}

    for filename in os.listdir(folder_wav):
        if filename.endswith(".wav"):
            path = os.path.join(folder_wav, filename)
            nombre = os.path.splitext(filename)[0]

            mfcc = extract_mfcc(path, n_mfcc=n_mfcc)
            hist = histogram_audio(mfcc, kmeans_model)
            diccionario_histogramas[nombre] = hist.tolist()  # Convertir a lista para JSON

            print(f"Procesado: {nombre}")

    return diccionario_histogramas
# FALTA TESTEAR (ESTA FUNCION)
def calcular_histogramas_todos_audios(folder_wav, kmeans_model, n_mfcc=13):
    """
    Procesa todos los archivos .wav y crea un diccionario de histogramas acústicos.
    
    Retorna:
        dict: {nombre_archivo: [valores del histograma]}
    """
    diccionario_histogramas = {}

    for filename in os.listdir(folder_wav):
        if filename.endswith(".wav"):
            path = os.path.join(folder_wav, filename)
            nombre = os.path.splitext(filename)[0]

            mfcc = extract_mfcc(path, n_mfcc=n_mfcc)
            hist = histogram_audio(mfcc, kmeans_model)
            diccionario_histogramas[nombre] = hist.tolist()  # Convertir a lista para JSON

            print(f"Procesado: {nombre}")

    return diccionario_histogramas

def guardar_json(diccionario, ruta_salida="histogramas_acusticos.json"):
    """
    Guarda un diccionario como archivo JSON.
    """
    with open(ruta_salida, "w") as f:
        json.dump(diccionario, f, indent=4)
    print(f"Histogramas guardados en: {ruta_salida}")



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



