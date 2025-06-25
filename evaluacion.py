"""
Módulo para la evaluación del sistema de aprobación de crédito.

Este módulo proporciona la función principal para evaluar
un modelo previamente entrenado utilizando un conjunto de
datos de prueba, calcular métricas relevantes y visualizar
resultados como la importancia de características.
"""

from src.evaluation import CreditModelEvaluator
from utils.visualization import CreditVisualizer

def ejecutar_evaluacion(modelo, procesador, X_test, y_test, feature_names):
    """
    Ejecuta la evaluación completa del modelo de crédito.

    Parámetros:
        modelo: Modelo entrenado a evaluar.
        procesador: Objeto procesador de datos usado para preprocesar datos de entrada.
        X_test (np.array o pd.DataFrame): Datos de prueba.
        y_test (np.array o pd.Series): Etiquetas verdaderas para los datos de prueba.
        feature_names (list): Lista con nombres de las características.

    Retorna:
        metrics (dict): Diccionario con métricas básicas y avanzadas de evaluación.
    """
    # Inicializar evaluador con el modelo y procesador
    evaluator = CreditModelEvaluator(modelo, procesador)
    
    # Definir los datos de prueba para la evaluación
    evaluator.set_test_data(X_test, y_test, feature_names)
    
    # Calcular métricas básicas y avanzadas
    metrics = evaluator.comprehensive_metrics()
    
    # Mostrar resumen de métricas principales
    print("\nMétricas de evaluación:")
    print(f"Accuracy: {metrics['basic']['accuracy']:.4f}")
    print(f"Precision: {metrics['basic']['precision']:.4f}")
    print(f"Recall: {metrics['basic']['recall']:.4f}")
    print(f"F1-Score: {metrics['basic']['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['advanced']['roc_auc']:.4f}")
    
    # Realizar análisis de importancia de características y visualizar si está disponible
    importance_analysis = evaluator.analyze_feature_importance(method='permutation')
    if importance_analysis:
        evaluator.plot_feature_importance(importance_analysis)
    
    return metrics
