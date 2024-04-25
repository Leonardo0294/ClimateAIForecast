import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def load_data_from_csv(file_path):
    # Cargar datos desde el archivo CSV con punto y coma como delimitador
    df = pd.read_csv(file_path, delimiter=';')
    
    # Convertir la columna de fecha al formato de fecha de pandas (d/m/año)
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)
    
    # Ordenar el DataFrame por fecha (si no está ordenado ya)
    df.sort_values(by='fecha', inplace=True)
    
    # Extraer la serie de tiempo y el tiempo (días)
    time = np.arange(len(df))  # Tiempo como índices
    series_temp = df['temp_ambiente'].values  # Temperatura ambiente como serie de tiempo
    series_precip = df['precipitacion'].values  # Precipitación como serie de tiempo
    
    return time, series_temp, series_precip

# Cargar datos desde el archivo CSV
file_path = './dataSET_ESTACION_ESPAM.csv'

time, series_temp, series_precip = load_data_from_csv(file_path)

# Definir una función para preparar datos de entrenamiento y validación
def prepare_data_for_classification(series_temp, series_precip, window_size, threshold=0):
    X, Y = [], []
    for i in range(len(series_temp) - window_size - 1):
        # Obtener la secuencia de temperatura para la ventana actual
        temp_window = series_temp[i:i+window_size]
        
        # Definir la etiqueta (1 si va a llover, 0 si no va a llover)
        precip_next_day = 1 if series_precip[i+window_size+1] > threshold else 0
        
        X.append(temp_window)
        Y.append(precip_next_day)
    
    X, Y = np.array(X), np.array(Y)
    return X, Y

# Parámetros para el modelo LSTM
window_size = 10
batch_size = 16

# Preparar datos de entrenamiento y validación para clasificación de precipitación
X_train, Y_train = prepare_data_for_classification(series_temp, series_precip, window_size)
X_valid, Y_valid = prepare_data_for_classification(series_temp, series_precip, window_size)

# Crear y compilar el modelo LSTM para clasificación binaria
model = Sequential([
    LSTM(64, activation='relu', input_shape=(window_size, 1)),
    Dense(1, activation='sigmoid')  # Capa de salida con activación sigmoide para clasificación binaria
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, Y_train, epochs=20, batch_size=batch_size, validation_data=(X_valid, Y_valid))

# Hacer predicciones para un día específico
def predict_next_day_precipitation(model, series_temp, window_size, threshold=0):
    # Obtener la última ventana de temperatura
    temp_window = series_temp[-window_size:]
    
    # Preparar datos de entrada para el modelo
    X = np.array([temp_window])
    
    # Realizar la predicción (probabilidad de precipitación)
    probability_of_precipitation = model.predict(X)[0, 0]
    
    # Determinar si va a llover o no según el umbral
    predicted_precipitation = 1 if probability_of_precipitation > threshold else 0
    
    return predicted_precipitation, probability_of_precipitation

# Hacer una predicción para el siguiente día
predicted_precipitation, probability_of_precipitation = predict_next_day_precipitation(model, series_temp, window_size)

# Imprimir resultados
if predicted_precipitation == 1:
    print("Predicción: Va a llover con probabilidad", probability_of_precipitation)
else:
    print("Predicción: No va a llover con probabilidad", probability_of_precipitation)
