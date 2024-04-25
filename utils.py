import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def load_data_from_csv(file_path):
   
    df = pd.read_csv(file_path, delimiter=';')
    

    df['fecha'] = pd.to_datetime(df['fecha'], dayfirst=True)
    
   
    df.sort_values(by='fecha', inplace=True)
    
  
    series_temp = df['temp_ambiente'].values
    series_precip = df['precipitacion'].values
    
    return series_temp, series_precip


def prepare_data_for_classification(series_temp, series_precip, window_size, threshold=0):
    X, Y = [], []
    for i in range(len(series_temp) - window_size - 1):
        
        temp_window = series_temp[i:i+window_size]
        
       
        precip_next_day = 1 if series_precip[i+window_size+1] > threshold else 0
        
        X.append(temp_window)
        Y.append(precip_next_day)
    
    X, Y = np.array(X), np.array(Y)
    return X, Y


window_size = 10
batch_size = 16


file_path = './dataset.csv'
series_temp, series_precip = load_data_from_csv(file_path)


X_train, Y_train = prepare_data_for_classification(series_temp, series_precip, window_size)
X_valid, Y_valid = prepare_data_for_classification(series_temp, series_precip, window_size)


model = Sequential([
    LSTM(64, activation='relu', input_shape=(window_size, 1)),
    Dense(1, activation='sigmoid')  
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

# Función para generar un archivo HTML con los resultados de la predicción
def generate_html_output(predicted_precipitation, probability_of_precipitation):
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Predicción de Precipitación</title>
    </head>
    <body>
        <h1>Predicción de Precipitación</h1>
        <p><strong>Predicción:</strong> {'Va a llover' if predicted_precipitation == 1 else 'No va a llover'}</p>
        <p><strong>Probabilidad de Precipitación:</strong> {probability_of_precipitation:.2f}</p>
    </body>
    </html>
    """

    # Guardar el contenido HTML en un archivo
    with open('prediction_output.html', 'w') as file:
        file.write(html_content)

# Llamar a la función para generar el archivo HTML con los resultados de la predicción
generate_html_output(predicted_precipitation, probability_of_precipitation)
