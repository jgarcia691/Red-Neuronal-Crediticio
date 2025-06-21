"""
Módulo para el entrenamiento del sistema de aprobación de crédito.
"""
from src.data_processing import CreditDataProcessor
from src.model import create_model_from_config
from src.training import CreditModelTrainer

def ejecutar_entrenamiento(n_samples=10000, epochs=50, use_kaggle_data=False):
    # 1. Procesamiento de datos
    processor = CreditDataProcessor()
    if use_kaggle_data:
        data = processor.load_german_credit_data()
    else:
        data = processor.generate_sample_data(n_samples=n_samples)
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(data, test_size=0.4)
    # 2. Entrenamiento
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
    trainer = CreditModelTrainer(config)
    trainer.train_model(X_train, y_train)

    # Guardar objetos para evaluación posterior
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

    return trainer.model, processor, X_test, y_test, feature_names