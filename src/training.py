"""
Módulo de entrenamiento del modelo MLP para clasificación de crédito implementado desde cero.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
import json
from datetime import datetime
from imblearn.over_sampling import SMOTE

from src.data_processing import CreditDataProcessor
from src.model import CreditMLP, create_model_from_config

class CreditModelTrainer:
    """
    Clase encargada de gestionar todo el proceso de entrenamiento y evaluación
    del modelo de clasificación de crédito, incluyendo carga y preparación de datos,
    creación del modelo, balanceo de clases con SMOTE, entrenamiento,
    evaluación con métricas completas y generación de gráficos y guardado de resultados.
    """
    
    def __init__(self, config=None):
        """
        Inicializa el entrenador con una configuración opcional.
        Si no se provee config, utiliza una configuración por defecto.
        
        Args:
            config (dict, opcional): Diccionario con configuración de modelo, datos y entrenamiento.
        """
        self.config = config or self._get_default_config()
        self.model = None
        self.processor = CreditDataProcessor()  # Instancia para preprocesamiento de datos
        self.feature_names = None
        self.input_dim = None
    
    def _get_default_config(self):
        """
        Devuelve configuración por defecto para el modelo y datos.
        
        Returns:
            dict: Configuración por defecto.
        """
        return {
            'model': {
                'hidden_layers': [15],      # Tamaño de capas ocultas
                'activation': 'relu',       # Función de activación
                'learning_rate': 0.001,     # Tasa de aprendizaje
                'max_iter': 200,            # Épocas de entrenamiento
                'random_state': 42          # Semilla para reproducibilidad
            },
            'training': {},
            'data': {
                'test_size': 0.2,           # Proporción de datos para prueba
                'random_state': 42          # Semilla para división de datos
            }
        }
    
    def load_and_prepare_data(self, data_path=None):
        """
        Carga los datos desde archivo o desde un origen por defecto y realiza
        la preparación y división en conjuntos de entrenamiento y prueba.
        
        Args:
            data_path (str, opcional): Ruta al archivo de datos. Si no se proporciona, se usa ruta por defecto.
        
        Returns:
            tuple: X_train, X_test, y_train, y_test, nombres de características
        """
        print("Cargando y preparando datos...")
        if data_path and os.path.exists(data_path):
            data = self.processor.load_data(data_path)
        else:
            data = self.processor.load_data()
        
        # Prepara y divide los datos, obteniendo además los nombres de características
        X_train, X_test, y_train, y_test, feature_names = self.processor.prepare_data(
            data,
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state']
        )
        self.feature_names = feature_names
        self.input_dim = len(feature_names)
        return X_train, X_test, y_train, y_test, feature_names
    
    def create_model(self):
        """
        Crea una instancia del modelo MLP usando la configuración actual y la dimensión de entrada.
        
        Returns:
            CreditMLP: Modelo creado.
        """
        print("Creando modelo...")
        model_config = self.config['model'].copy()
        model_config['input_dim'] = self.input_dim
        self.model = create_model_from_config(model_config)
        print("Modelo creado:")
        print(self.model.get_model_summary())
        return self.model
    
    def train_model(self, X_train, y_train):
        """
        Aplica balanceo de clases con SMOTE para corregir desbalance y entrena el modelo.
        
        Args:
            X_train (np.array): Datos de entrenamiento.
            y_train (np.array): Etiquetas de entrenamiento.
            
        Returns:
            CreditMLP: Modelo entrenado.
        """
        print("Aplicando SMOTE para balancear las clases...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        print(f"Clases después de SMOTE: {np.bincount(y_res)}")
        print("Iniciando entrenamiento...")
        
        # Actualizar dimensión de entrada después del remuestreo
        self.input_dim = X_res.shape[1]
        
        # Crear modelo si no existe aún
        if self.model is None:
            self.create_model()
        
        # Entrenar modelo con los datos balanceados
        self.model.fit(X_res, y_res)
        print("Entrenamiento completado!")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evalúa el modelo con métricas completas de clasificación, ROC, matriz de confusión y más.
        
        Args:
            X_test (np.array): Datos de prueba.
            y_test (np.array): Etiquetas reales para prueba.
            
        Returns:
            dict: Métricas de evaluación detalladas.
        """
        print("Evaluando modelo...")
        
        # Obtener probabilidades y predicciones binarias
        y_pred_proba = self.model.predict_proba(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Reporte detallado de clasificación (precision, recall, f1, etc)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Curva ROC y AUC
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Desglose de matriz de confusión
        tn, fp, fn, tp = cm.ravel()
        
        # Métricas derivadas
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Reunir todas las métricas en un diccionario
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
        Grafica el historial de pérdida de entrenamiento y validación,
        además de la importancia de características si está disponible.
        
        Args:
            save_path (str, opcional): Ruta para guardar la imagen del gráfico.
        """
        if hasattr(self.model, 'training_loss') and self.model.training_loss:
            plt.figure(figsize=(12, 5))
            
            # Gráfico de pérdida de entrenamiento y validación
            plt.subplot(1, 2, 1)
            plt.plot(self.model.training_loss, label='Training Loss', color='blue')
            if self.model.validation_loss:
                plt.plot(self.model.validation_loss, label='Validation Loss', color='red')
            plt.title('Historial de Entrenamiento')
            plt.xlabel('Época')
            plt.ylabel('Pérdida')
            plt.legend()
            plt.grid(True)
            
            # Gráfico de importancia de características (top 10)
            if self.feature_names:
                plt.subplot(1, 2, 2)
                importance = self.model.get_feature_importance(self.feature_names)
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                features, importances = zip(*sorted_importance[:10])
                
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
        """
        Grafica la matriz de confusión y la curva ROC con AUC.
        
        Args:
            metrics (dict): Diccionario con métricas devuelto por evaluate_model.
            save_path (str, opcional): Ruta para guardar la imagen.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Matriz de confusión
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Curva ROC
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
        """
        Guarda las métricas de evaluación en un archivo JSON con timestamp.
        
        Args:
            metrics (dict): Métricas de evaluación a guardar.
            save_dir (str): Directorio donde guardar los resultados.
        """
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metrics_path = os.path.join(save_dir, f'metrics_{timestamp}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Resultados guardados en {save_dir}:")
        print(f"- Métricas: {metrics_path}")
    
    def run_full_training(self, data_path=None):
        """
        Ejecuta todo el flujo de carga de datos, creación, entrenamiento,
        evaluación, visualización y guardado de resultados.
        
        Args:
            data_path (str, opcional): Ruta al archivo de datos para carga.
            
        Returns:
            dict: Métricas obtenidas tras la evaluación final.
        """
        print("=== INICIANDO ENTRENAMIENTO COMPLETO ===")
        
        # Carga y preparación de datos
        X_train, X_test, y_train, y_test, feature_names = self.load_and_prepare_data(data_path)
        
        # Creación del modelo
        self.create_model()
        
        # Entrenamiento con datos balanceados
        self.train_model(X_train, y_train)
        
        # Evaluación del modelo entrenado
        metrics = self.evaluate_model(X_test, y_test)
        
        # Visualización de métricas y entrenamiento
        self.plot_evaluation_metrics(metrics, 'models/evaluation_metrics.png')
        self.plot_training_history('models/training_history.png')
        
        # Guardado de resultados
        self.save_results(metrics)
        
        # Resumen por consola
        print("\n=== RESUMEN DE RESULTADOS ===")
        print(f"Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Precision: {metrics['test_precision']:.4f}")
        print(f"Recall: {metrics['test_recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
