"""
Módulo para el entrenamiento del sistema de aprobación de crédito.

Este módulo contiene la función principal para orquestar
el flujo completo de entrenamiento del modelo de crédito,
desde la carga o generación de datos, el preprocesamiento,
hasta la configuración y entrenamiento del modelo,
finalizando con la serialización de objetos clave para su evaluación posterior.
"""
# Importa las clases y funciones necesarias para procesar datos, crear y entrenar el modelo
from src.data_processing import CreditDataProcessor
from src.model import create_model_from_config
from src.training import CreditModelTrainer
import pickle

# Función principal que ejecuta todo el proceso de entrenamiento
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
    # Si se indica, carga datos reales; si no, genera datos sintéticos
    if use_kaggle_data:
        data = processor.load_german_credit_data()
    else:
        data = processor.generate_sample_data(n_samples=n_samples)
    
    # 2. Preparar datos (normalización, separación en train/test)
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(data, test_size=0.4)
    
    # 3. Configuración del modelo y parámetros de entrenamiento
    config = {
        'model': {
            'type': 'basic',  # Tipo de modelo
            'hidden_layers': [35],  # Una capa oculta de 35 neuronas
            'dropout_rate': 0.3,    # Tasa de dropout para regularización
            'learning_rate': 0.001, # Tasa de aprendizaje
            'l2_reg': 0.01          # Regularización L2
        },
        'training': {
            'batch_size': 32,           # Tamaño de lote para entrenamiento
            'epochs': epochs,           # Número de épocas
            'validation_split': 0.6,    # 40% de los datos de entrenamiento se usan para validación
            'early_stopping_patience': 10, # Paciencia para early stopping
            'reduce_lr_patience': 5,    # Paciencia para reducir learning rate
            'reduce_lr_factor': 0.5     # Factor para reducir learning rate
        },
        'data': {
            'test_size': 0.4,           # (No se usa aquí, pero está en la config)
            'random_state': 42          # Semilla para reproducibilidad
        }
    }
    
    # 4. Crear entrenador y ejecutar entrenamiento
    trainer = CreditModelTrainer(config)
    # Entrena el modelo con los datos de entrenamiento
    trainer.train_model(X_train, y_train)

    # Guardar objetos para evaluación posterior
    # Serializa y guarda el modelo, el procesador y los datos de test para usarlos después
    import pickle
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

    # Devuelve los objetos principales para su uso posterior
    return trainer.model, processor, X_test, y_test, feature_names
