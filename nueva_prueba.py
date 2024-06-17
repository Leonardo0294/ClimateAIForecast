import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime
import numpy as np

def load_data_from_csv(file_path):
    """
    Carga los datos desde un archivo CSV y procesa las fechas y las columnas de interés.

    Args:
    file_path (str): Ruta al archivo CSV.

    Returns:
    df (pandas.DataFrame): DataFrame con los datos del archivo CSV.
    """
    # Leer el archivo CSV especificando el encoding como utf-8 y manejo de decimales
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8', decimal=',')

    # Convertir la columna 'Fecha / Hora' a datetime
    df['Fecha / Hora'] = pd.to_datetime(df['Fecha / Hora'], dayfirst=True, errors='coerce')

    # Eliminar filas con fechas incorrectas (NaT)
    df = df.dropna(subset=['Fecha / Hora'])

    return df

def preprocess_data(df):
    """
    Realiza el preprocesamiento de los datos para entrenar el modelo de predicción.

    Args:
    df (pandas.DataFrame): DataFrame con los datos del archivo CSV.

    Returns:
    X (numpy.array): Datos de entrada para el modelo.
    y (numpy.array): Datos de salida para el modelo (target).
    """
    # Seleccionar características para el modelo
    features = ['Temperatura del aire HC [°C] - promedio', 'Punto de Rocío [°C] - promedio',
                'Radiación solar [W/m2] - promedio', 'DPV [kPa] - promedio',
                'Humedad relativa HC [%] - promedio', 'Precipitación [mm]',
                'Velocidad de Viento [m/s] - promedio', 'Dirección de Viento [deg]']

    X = df[features].values
    y = df['Temperatura del aire HC [°C] - máx'].values  # Predecir la temperatura máxima

    return X, y

def train_model(X_train, y_train):
    """
    Entrena un modelo de regresión RandomForest.

    Args:
    X_train (numpy.array): Datos de entrada de entrenamiento.
    y_train (numpy.array): Datos de salida de entrenamiento (target).

    Returns:
    model (sklearn.ensemble.RandomForestRegressor): Modelo entrenado.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa un modelo de regresión RandomForest en el conjunto de prueba.

    Args:
    model (sklearn.ensemble.RandomForestRegressor): Modelo entrenado.
    X_test (numpy.array): Datos de entrada de prueba.
    y_test (numpy.array): Datos de salida de prueba (target).

    Returns:
    rmse (float): Error cuadrático medio en el conjunto de prueba.
    """
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def main():
    # Ruta al archivo CSV con los datos meteorológicos
    file_path = './station_data.csv'

    # Cargar los datos desde el archivo CSV
    df = load_data_from_csv(file_path)

    # Preprocesar los datos
    X, y = preprocess_data(df)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    model = train_model(X_train, y_train)

    # Evaluar el modelo
    rmse = evaluate_model(model, X_test, y_test)
    print(f'Error cuadrático medio en el conjunto de prueba: {rmse:.2f} °C')

if __name__ == "__main__":
    main()
