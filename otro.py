import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tkinter import *
from datetime import datetime
import numpy as np

# Definir 'features' globalmente
features = ['Temperatura del aire HC [°C] - promedio', 'Punto de Rocío [°C] - promedio',
            'Radiación solar [W/m2] - promedio', 'DPV [kPa] - promedio',
            'Humedad relativa HC [%] - promedio', 'Precipitación [mm]',
            'Velocidad de Viento [m/s] - promedio', 'Dirección de Viento [deg]']

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

def predict_future_temperature(model, df, future_date):
    """
    Predice la temperatura futura para una fecha especificada.

    Args:
    model (sklearn.ensemble.RandomForestRegressor): Modelo entrenado.
    df (pandas.DataFrame): DataFrame con los datos históricos.
    future_date (str): Fecha futura en formato 'YYYY-MM-DD'.

    Returns:
    pred_temp (float): Predicción de temperatura para la fecha futura.
    """
    # Preprocesar la fecha futura
    future_date = pd.to_datetime(future_date, format='%Y-%m-%d')

    # Obtener los últimos datos disponibles para la fecha futura
    last_data = df[df['Fecha / Hora'] <= future_date].tail(1)

    if last_data.empty:
        raise ValueError("No hay datos disponibles para la fecha seleccionada.")

    # Extraer las características de los últimos datos disponibles
    last_features = last_data[features].values

    # Realizar la predicción
    pred_temp = model.predict(last_features)[0]

    return pred_temp

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

    # Crear una interfaz gráfica simple para predecir la temperatura futura
    root = Tk()
    root.title("Predicción de Temperatura Futura")

    # Función para manejar el evento del botón de predicción
    def predict_button_clicked():
        future_date = entry_date.get()
        try:
            pred_temp = predict_future_temperature(model, df, future_date)
            result_label.config(text=f"Predicción para {future_date}:\nTemperatura predicha: {pred_temp:.2f} °C")
        except ValueError as e:
            result_label.config(text=str(e))

    # Etiqueta y entrada para la fecha futura
    Label(root, text="Ingrese una fecha futura (YYYY-MM-DD):").pack()
    entry_date = Entry(root)
    entry_date.pack()

    # Botón para realizar la predicción
    Button(root, text="Predecir", command=predict_button_clicked).pack()

    # Etiqueta para mostrar el resultado de la predicción
    result_label = Label(root, text="")
    result_label.pack()

    root.mainloop()

if __name__ == "__main__":
    main()
