from flask import Flask

def create_app():
    app = Flask(__name__)

    # Configuración de la aplicación
    app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui'

    # Importar y registrar blueprints (rutas)
    from .routes import bp as routes_bp
    app.register_blueprint(routes_bp)

    return app
