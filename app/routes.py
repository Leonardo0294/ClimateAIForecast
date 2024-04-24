from flask import Blueprint, request, jsonify, render_template
from .models import entrenar_modelo, hacer_prediccion
import pandas as pd

bp = Blueprint('routes', __name__)

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/train_model', methods=['POST'])
def train_model():
    df = pd.read_csv('datos_climaticos.csv')
    X_train = df[['meantemp', 'humidity', 'wind_speed', 'meanpressure']]
    y_train = df['precipitation']
    modelo_entrenado = entrenar_modelo(X_train, y_train)
    return jsonify({'message': 'Modelo entrenado correctamente'}), 200

@bp.route('/predict', methods=['POST'])
def predict():
    datos = request.json
    modelo = crear_modelo()  # Cargar el modelo entrenado o crear uno nuevo si es necesario
    resultado = hacer_prediccion(modelo, datos)
    return jsonify({'prediction': str(resultado[0])}), 200
