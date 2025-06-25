"""
Módulo de evaluación y análisis del modelo de clasificación de crédito.
Incluye análisis de interpretabilidad, sesgos y métricas avanzadas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)
from sklearn.inspection import permutation_importance
import shap
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

class CreditModelEvaluator:
    """
    Clase para evaluar y analizar el modelo de clasificación de crédito.

    Proporciona métodos para:
    - Establecer datos de prueba.
    - Obtener predicciones con un umbral configurable.
    - Calcular métricas comprehensivas (precisión, recall, F1, AUC, etc.).
    - Graficar resultados de evaluación.
    - Analizar la importancia de características mediante diferentes métodos (SHAP, permutación, gradientes).
    - Analizar sesgos y justicia en subgrupos definidos por características sensibles.
    - Generar reportes de explicabilidad basados en LIME para muestras individuales.
    """
    
    def __init__(self, model, processor=None):
        """
        Inicializa el evaluador del modelo.

        Args:
            model: Modelo entrenado compatible con método predict.
            processor: Procesador opcional de datos (e.g., para transformar características).
        """
        self.model = model
        self.processor = processor
        self.feature_names = None
        self.X_test = None
        self.y_test = None
        
    def set_test_data(self, X_test, y_test, feature_names=None):
        """
        Establece los datos de prueba para evaluación.

        Args:
            X_test (np.array): Matriz de características de prueba.
            y_test (np.array): Vector de etiquetas verdaderas.
            feature_names (list): Lista de nombres de características (opcional).
        """
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_test.shape[1])]
    
    def get_predictions(self, threshold=0.5):
        """
        Obtiene las predicciones del modelo aplicando un umbral binario.

        Args:
            threshold (float): Umbral para convertir probabilidades en etiquetas binarias.

        Returns:
            tuple: Probabilidades predichas y etiquetas binarias.
        """
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > threshold).astype(int)
        return y_pred_proba, y_pred
    
    def comprehensive_metrics(self, threshold=0.5):
        """
        Calcula un conjunto amplio de métricas de evaluación para el modelo.

        Incluye métricas clásicas, curvas ROC y PR, y métricas adicionales de balance y robustez.

        Args:
            threshold (float): Umbral para clasificación binaria.

        Returns:
            dict: Diccionario con métricas básicas, avanzadas, matriz de confusión,
                  reporte de clasificación y predicciones.
        """
        y_pred_proba, y_pred = self.get_predictions(threshold)
        
        # Cálculo de matriz de confusión y métricas derivadas
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall  # Equivalente a recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Curvas ROC y Precision-Recall
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = average_precision_score(self.y_test, y_pred_proba)
        
        balanced_accuracy = (sensitivity + specificity) / 2
        g_mean = np.sqrt(sensitivity * specificity)
        
        class_report = classification_report(self.y_test, y_pred, output_dict=True)
        
        metrics = {
            'basic': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1_score': f1_score,
                'balanced_accuracy': balanced_accuracy,
                'g_mean': g_mean
            },
            'advanced': {
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'sensitivity': recall,
                'false_positive_rate': fpr,
                'true_positive_rate': tpr,
                'precision_curve': precision_curve,
                'recall_curve': recall_curve
            },
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': {
                'probabilities': y_pred_proba,
                'predictions': y_pred
            }
        }
        
        return metrics
    
    def plot_comprehensive_evaluation(self, metrics, save_path=None):
        """
        Genera una figura con múltiples gráficos para visualizar la evaluación del modelo.

        Incluye matriz de confusión, curvas ROC y PR, histogramas y barras de métricas.

        Args:
            metrics (dict): Métricas calculadas por `comprehensive_metrics`.
            save_path (str): Ruta para guardar la imagen (opcional).
        """
        # (Aquí sigue el código existente para graficar)
        ...
    
    def analyze_feature_importance(self, method='shap', n_samples=1000):
        """
        Realiza un análisis de importancia de características con el método indicado.

        Args:
            method (str): Método a usar ('shap', 'permutation', 'gradient').
            n_samples (int): Número de muestras para SHAP (si aplica).

        Returns:
            dict: Resultados del análisis de importancia.
        """
        # (Implementación existente)
        ...
    
    def _shap_analysis(self, n_samples):
        """
        Análisis de importancia basado en SHAP.

        Args:
            n_samples (int): Número de muestras a analizar.

        Returns:
            dict: Valores SHAP y DataFrame con importancias.
        """
        # (Implementación existente)
        ...
    
    def _permutation_importance_analysis(self):
        """
        Análisis de importancia usando permutación.

        Returns:
            dict: Resultados con importancias medias y desviaciones estándar.
        """
        # (Implementación existente)
        ...
    
    def _gradient_importance_analysis(self):
        """
        Análisis basado en gradientes (requiere TensorFlow).

        Returns:
            dict: Importancias basadas en gradientes y gradientes calculados.
        """
        # (Implementación existente)
        ...
    
    def plot_feature_importance(self, importance_analysis, save_path=None):
        """
        Grafica la importancia de características según el análisis proporcionado.

        Args:
            importance_analysis (dict): Resultado del análisis de importancia.
            save_path (str): Ruta para guardar el gráfico (opcional).
        """
        # (Implementación existente)
        ...
    
    def analyze_bias_and_fairness(self, sensitive_features=None):
        """
        Evalúa sesgos y justicia del modelo según características sensibles.

        Args:
            sensitive_features (dict): Diccionario con nombre y valores de características sensibles.

        Returns:
            dict: Métricas por grupo para cada característica sensible.
        """
        # (Implementación existente)
        ...
    
    def generate_explanation_report(self, sample_idx=0, save_path=None):
        """
        Genera un reporte explicativo usando LIME para una muestra específica.

        Args:
            sample_idx (int): Índice de la muestra a explicar.
            save_path (str): Ruta para guardar el reporte en formato JSON (opcional).

        Returns:
            dict: Reporte con explicación y datos de la muestra.
        """
        # (Implementación existente)
        ...

def main():
    """
    Función demostrativa para indicar el uso esperado del módulo.

    No ejecuta análisis; solo muestra instrucciones para uso posterior.
    """
    print("Este módulo debe usarse después de entrenar el modelo")
    print("Ejemplo de uso:")
    print("1. Entrenar modelo usando training.py")
    print("2. Cargar modelo entrenado")
    print("3. Usar CreditModelEvaluator para análisis")

if __name__ == "__main__":
    main()
