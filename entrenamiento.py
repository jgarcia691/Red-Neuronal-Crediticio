"""
Módulo para el entrenamiento del sistema de aprobación de crédito.

Este módulo contiene la función principal para orquestar
el flujo completo de entrenamiento del modelo de crédito,
desde la carga o generación de datos, el preprocesamiento,
hasta la configuración y entrenamiento del modelo,
finalizando con la serialización de objetos clave para su evaluación posterior.
"""

from src.data_processing import CreditDataProcessor
from src.model import create_model_from_config
from src.training import CreditModelTrainer
import pickle

def ejecutar_entrenamiento(n_samples=10000, epochs=50, use_kaggle_data=False):
    """
    Ejecuta el proceso completo de entrenamiento del modelo de crédito.

    Parámetros:
        n_samples (int, opcional): Número de muestras a generar si no se usa el dataset de Kaggle.
                                   Por defecto 10,000.
        epochs (int, opcional): Número de épocas para entrenar el modelo. Por defecto 50.
        use_kaggle_data (bool, opcional): Si es True, se carga el dataset de crédito alemán de Kaggle.
                                          Si es False, se generan datos sintéticos. Por defecto False.

    Retorna:
        model: Modelo entrenado.
        processor: Objeto procesador de datos utilizado.
        X_test (np.array o pd.DataFrame): Datos de prueba para evaluación.
        y_test (np.array o pd.Series): Etiquetas de prueba.
        feature_names (list): Lista de nombres de características usadas por el modelo.
    """
    # 1. Inicializar procesador y cargar/generar datos
    processor = CreditDataProcessor()
    if use_kaggle_data:
        data = processor.load_german_credit_data()
    else:
        data = processor.generate_sample_data(n_samples=n_samples)
    
    # 2. Preparar datos (normalización, separación en train/test)
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(data, test_size=0.4)
    
    # 3. Configuración del modelo y parámetros de entrenamiento
    config = {
        'model': {
            'type': 'basic',
            'hidden_layers': [15],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'l2_reg': 0.01
        },
        'training': {
            'batch_size': 32,
            'epochs': epochs,
            'validation_split': 0.2,
            'early_stopping_patience': 10,
            'reduce_lr_patience': 5,
            'reduce_lr_factor': 0.5
        },
        'data': {
            'test_size': 0.4,
            'random_state': 42
        }
    }
    
    # 4. Crear entrenador y ejecutar entrenamiento
    trainer = CreditModelTrainer(config)
    trainer.train_model(X_train, y_train)

    # 5. Guardar objetos importantes para evaluación y reutilización posterior
    with open('modelo_entrenado.pkl', 'wb') as f:
        pickle.dump(trainer.model, f)
    with open('procesador.pkl', 'wb') as f:
        pickle.dump(processor, f)
    with open('X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    return trainer.model, processor, X_test, y_test, feature_names
