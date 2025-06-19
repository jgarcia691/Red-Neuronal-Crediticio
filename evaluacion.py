"""
Módulo para la evaluación del sistema de aprobación de crédito.
"""
from src.evaluation import CreditModelEvaluator
from utils.visualization import CreditVisualizer

def ejecutar_evaluacion(modelo, procesador, X_test, y_test, feature_names):
    evaluator = CreditModelEvaluator(modelo, procesador)
    evaluator.set_test_data(X_test, y_test, feature_names)
    metrics = evaluator.comprehensive_metrics()
    print("\nMétricas de evaluación:")
    print(f"Accuracy: {metrics['basic']['accuracy']:.4f}")
    print(f"Precision: {metrics['basic']['precision']:.4f}")
    print(f"Recall: {metrics['basic']['recall']:.4f}")
    print(f"F1-Score: {metrics['basic']['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['advanced']['roc_auc']:.4f}")
    # Importancia de características
    importance_analysis = evaluator.analyze_feature_importance(method='permutation')
    if importance_analysis:
        evaluator.plot_feature_importance(importance_analysis) 