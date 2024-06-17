import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def load_data_from_csv(file_path):
    """
    Carga los datos desde un archivo CSV y procesa las fechas y las columnas de interés.

    Args:
    file_path (str): Ruta al archivo CSV.

    Returns:
    df (pandas.DataFrame): DataFrame con los datos del archivo CSV.
    """
    # Leer el archivo CSV usando pandas
    df = pd.read_csv(file_path, delimiter=';')

    # Convertir la columna de fechas a formato datetime (considerando formato día-mes-año)
    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)

    # Ordenar el DataFrame por fecha
    df.sort_values(by='fecha', inplace=True)

    return df

def prepare_data_for_lstm(df, window_size):
    """
    Prepara los datos para el entrenamiento del modelo LSTM.

    Args:
    df (pandas.DataFrame): DataFrame con los datos del tiempo.
    window_size (int): Tamaño de la ventana (número de días a considerar como entrada).

    Returns:
    X (numpy.array): Array 3D de características para el modelo LSTM.
    Y (numpy.array): Array 1D de etiquetas para el modelo LSTM.
    """
    # Extraer la serie de temperatura ambiente
    series_temp = df['temp_ambiente'].values

    X, Y = [], []

    # Iterar sobre los datos para crear las secuencias de entrada y las etiquetas
    for i in range(len(series_temp) - window_size - 1):
        # Extraer la ventana de temperatura
        temp_window = series_temp[i:i+window_size]

        # Determinar si va a llover al día siguiente
        precip_next_day = 1 if df.loc[i+window_size+1, 'precipitacion'] > 0 else 0

        # Agregar la ventana de temperatura a X y la etiqueta a Y
        X.append(temp_window)
        Y.append(precip_next_day)

    # Convertir las listas a arrays numpy
    X, Y = np.array(X), np.array(Y)
    # Reshape para que sea compatible con LSTM (n_samples, window_size, n_features)
    X = np.reshape(X, (X.shape[0], window_size, 1))

    return X, Y

def load_weather_data_for_date(df, target_date):
    """
    Carga los detalles del tiempo para una fecha específica desde un DataFrame.

    Args:
    df (pandas.DataFrame): DataFrame con los datos del tiempo.
    target_date (str): Fecha objetivo en formato 'dd/mm/yyyy' (por ejemplo, '10/10/2018').

    Returns:
    weather_details (dict): Diccionario con los detalles del tiempo para la fecha objetivo.
    """
    # Convertir la fecha objetivo a formato datetime
    target_date = pd.to_datetime(target_date, dayfirst=True)

    # Filtrar los datos para la fecha objetivo
    filtered_data = df[df['fecha'] == target_date]

    # Verificar si se encontraron datos para la fecha objetivo
    if len(filtered_data) == 0:
        print(f"No se encontraron datos para la fecha {target_date}")
        return None

    # Convertir los detalles del tiempo a un diccionario
    weather_details = filtered_data.iloc[0].to_dict()

    return weather_details

def predict_next_day_precipitation(model, temp_window):
    """
    Realiza una predicción de precipitación para el próximo día usando el modelo LSTM.

    Args:
    model (tensorflow.keras.Model): Modelo LSTM entrenado.
    temp_window (numpy.array): Ventana de temperatura para la predicción.

    Returns:
    predicted_precipitation (int): Predicción binaria de precipitación (0 o 1).
    probability_of_precipitation (float): Probabilidad de precipitación.
    """
    # Preparar los datos de entrada para el modelo LSTM (reshape necesario)
    X = np.reshape(temp_window, (1, temp_window.shape[0], 1))

    # Realizar la predicción (obtener la probabilidad de precipitación)
    probability_of_precipitation = model.predict(X)[0, 0]

    # Determinar si va a llover según el umbral
    predicted_precipitation = 1 if probability_of_precipitation > 0.5 else 0

    return predicted_precipitation, probability_of_precipitation

# Ruta al archivo CSV con los datos del tiempo
file_path = './dataset.csv'

# Cargar los datos desde el archivo CSV
df = load_data_from_csv(file_path)

# Parámetros del modelo LSTM
window_size = 10

# Preparar los datos para el modelo LSTM
X, Y = prepare_data_for_lstm(df, window_size)

# Definir el modelo LSTM
model = Sequential([
    LSTM(64, activation='relu', input_shape=(window_size, 1)),  # Capa LSTM con 64 unidades
    Dense(1, activation='sigmoid')  # Capa densa para la salida binaria (0 o 1)
])

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar el modelo (usando todos los datos disponibles)
model.fit(X, Y, epochs=20, batch_size=16)

# Entrada interactiva para ingresar la fecha objetivo
target_date = input("Ingrese la fecha para la predicción (formato dd/mm/yyyy): ")

# Cargar los detalles del tiempo para la fecha objetivo
weather_details = load_weather_data_for_date(df, target_date)

if weather_details:
    # Extraer la temperatura ambiente para la fecha objetivo
    temp_for_prediction = np.array([weather_details['temp_ambiente']])

    # Realizar la predicción de precipitación para el próximo día
    predicted_precipitation, probability_of_precipitation = predict_next_day_precipitation(model, temp_for_prediction)

    # Imprimir los resultados de la predicción
    if predicted_precipitation == 1:
        print("Predicción: Va a llover con probabilidad", probability_of_precipitation)
    else:
        print("Predicción: No va a llover con probabilidad", probability_of_precipitation)
else:
    print("No se pueden realizar predicciones para la fecha especificada.")
