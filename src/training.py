"""
Módulo de entrenamiento del modelo MLP para clasificación de crédito implementado desde cero.
"""

# Importación de librerías necesarias para manejo de datos, visualización y métricas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
import json
from datetime import datetime
from imblearn.over_sampling import SMOTE

# Importa el procesador de datos y el modelo MLP definidos en otros módulos
from src.data_processing import CreditDataProcessor
from src.model import CreditMLP, create_model_from_config

class CreditModelTrainer:
    """
    Clase para entrenar y evaluar el modelo de clasificación de crédito.
    """
    
    def __init__(self, config=None):
        # Inicializa la clase con una configuración (o usa la predeterminada)
        self.config = config or self._get_default_config()
        self.model = None
        self.processor = CreditDataProcessor()
        self.feature_names = None
        self.input_dim = None
    
    def _get_default_config(self):
        # Configuración por defecto del modelo y entrenamiento
        return {
            'model': {
                'hidden_layers': [15],
                'activation': 'relu',
                'learning_rate': 0.001,
                'max_iter': 200,
                'random_state': 42
            },
            'training': {},
            'data': {
                'test_size': 0.4,
                'random_state': 42
            }
        }
    
    def load_and_prepare_data(self, data_path=None):
        # Carga y prepara los datos para entrenamiento y prueba
        print("Cargando y preparando datos...")
        if data_path and os.path.exists(data_path):
            data = self.processor.load_data(data_path)
        else:
            data = self.processor.load_data()
        X_train, X_test, y_train, y_test, feature_names = self.processor.prepare_data(
            data,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        self.feature_names = feature_names
        self.input_dim = len(feature_names)
        return X_train, X_test, y_train, y_test, feature_names
    
    def create_model(self):
        # Crea el modelo MLP según la configuración
        print("Creando modelo...")
        model_config = self.config['model'].copy()
        model_config['input_dim'] = self.input_dim
        self.model = create_model_from_config(model_config)
        print("Modelo creado:")
        print(self.model.get_model_summary())
        return self.model
    
    def train_model(self, X_train, y_train):
        # Aplica SMOTE para balancear las clases y entrena el modelo
        print("Aplicando SMOTE para balancear las clases...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"Clases después de SMOTE: {np.bincount(y_res)}")
        print("Iniciando entrenamiento...")
        self.input_dim = X_res.shape[1]
        if self.model is None:
            self.create_model()
        self.model.fit(X_res, y_res)
        print("Entrenamiento completado!")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        # Evalúa el modelo en el conjunto de prueba y calcula métricas
        print("Evaluando modelo...")
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        metrics = {
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'f1_score': f1_score,
            'roc_auc': roc_auc,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'roc_curve': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            }
        }
        return metrics
    
    def plot_training_history(self, save_path=None):
        """
        Grafica el historial de entrenamiento del modelo manual.
        """
        if hasattr(self.model, 'training_loss') and self.model.training_loss:
            plt.figure(figsize=(12, 5))
            
            # Gráfico de pérdida
            plt.subplot(1, 2, 1)
            plt.plot(self.model.training_loss, label='Training Loss', color='blue')
            if self.model.validation_loss:
                plt.plot(self.model.validation_loss, label='Validation Loss', color='red')
            plt.title('Historial de Entrenamiento')
            plt.xlabel('Época')
            plt.ylabel('Pérdida')
            plt.legend()
            plt.grid(True)
            
            # Gráfico de importancia de características
            if self.feature_names:
                plt.subplot(1, 2, 2)
                importance = self.model.get_feature_importance(self.feature_names)
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                features, importances = zip(*sorted_importance[:10])  # Top 10
                
                plt.barh(range(len(features)), importances)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Importancia')
                plt.title('Top 10 Características Más Importantes')
                plt.gca().invert_yaxis()
            
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Historial guardado en: {save_path}")
            plt.show()
        else:
            print("No hay historial de entrenamiento disponible.")
    
    def plot_evaluation_metrics(self, metrics, save_path=None):
        # Grafica la matriz de confusión y la curva ROC
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        fpr = metrics['roc_curve']['fpr']
        tpr = metrics['roc_curve']['tpr']
        axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["roc_auc"]:.3f})')
        axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1].set_title('ROC Curve')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].legend()
        axes[1].grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Métricas guardadas en: {save_path}")
        plt.show()
    
    def save_results(self, metrics, save_dir='models'):
        # Guarda las métricas y resultados en un archivo JSON
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_path = os.path.join(save_dir, f'metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Resultados guardados en {save_dir}:")
        print(f"- Métricas: {metrics_path}")
    
    def run_full_training(self, data_path=None):
        # Ejecuta todo el flujo: carga datos, entrena, evalúa y guarda resultados
        print("=== INICIANDO ENTRENAMIENTO COMPLETO ===")
        X_train, X_test, y_train, y_test, feature_names = self.load_and_prepare_data(data_path)
        self.create_model()
        self.train_model(X_train, y_train)
        metrics = self.evaluate_model(X_test, y_test)
        self.plot_evaluation_metrics(metrics, 'models/evaluation_metrics.png')
        self.plot_training_history('models/training_history.png')
        self.save_results(metrics)
        print("\n=== RESUMEN DE RESULTADOS ===")
        print(f"Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Precision: {metrics['test_precision']:.4f}")
        print(f"Recall: {metrics['test_recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        return metrics
