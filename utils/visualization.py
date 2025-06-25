"""
Módulo de utilidades para visualización de datos y resultados del sistema de crédito.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class CreditVisualizer:
    """
    Clase para crear visualizaciones que ayudan a entender
    los datos, el desempeño del modelo y los resultados del sistema
    de aprobación de crédito.
    """
    
    def __init__(self, style='seaborn'):
        """
        Inicializa el visualizador con un estilo para matplotlib y define una paleta de colores.
        
        Args:
            style (str): Estilo visual para matplotlib (por defecto 'seaborn').
        """
        plt.style.use(style)
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def plot_data_distribution(self, data, save_path=None):
        """
        Genera histogramas para visualizar la distribución de las variables numéricas del dataset,
        excluyendo la variable objetivo 'default'.
        
        Args:
            data (pd.DataFrame): Dataset con los datos.
            save_path (str, opcional): Ruta para guardar la imagen resultante.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Selecciona las columnas numéricas excluyendo la variable target
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'default']
        
        # Grafica hasta 6 variables numéricas en histogramas
        for i, col in enumerate(numerical_cols[:6]):
            row = i // 3
            col_idx = i % 3
            
            axes[row, col_idx].hist(data[col], bins=30, alpha=0.7, 
                                  color=self.colors['primary'], edgecolor='black')
            axes[row, col_idx].set_title(f'Distribución de {col}')
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel('Frecuencia')
            axes[row, col_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, data, save_path=None):
        """
        Genera un mapa de calor de la matriz de correlación entre las variables del dataset,
        con máscara para la mitad superior para mejorar legibilidad.
        
        Args:
            data (pd.DataFrame): Dataset con los datos.
            save_path (str, opcional): Ruta para guardar la imagen resultante.
        """
        correlation_matrix = data.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, fmt='.2f')
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_target_distribution(self, data, save_path=None):
        """
        Visualiza la distribución del target 'default' con un gráfico de pastel,
        además muestra un boxplot de la primera variable numérica agrupada por el target.
        
        Args:
            data (pd.DataFrame): Dataset con los datos.
            save_path (str, opcional): Ruta para guardar la imagen resultante.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Conteo y gráfico circular del target
        target_counts = data['default'].value_counts()
        colors = [self.colors['success'], self.colors['danger']]
        labels = ['No Default', 'Default']
        
        axes[0].pie(target_counts.values, labels=labels, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
        axes[0].set_title('Distribución del Target')
        
        # Boxplot de la primera variable numérica separada por target
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'default']
        
        if len(numerical_cols) > 0:
            col = numerical_cols[0]
            data.boxplot(column=col, by='default', ax=axes[1])
            axes[1].set_title(f'{col} por Target')
            axes[1].set_xlabel('Default')
            axes[1].set_ylabel(col)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_training_history(self, history, save_path=None):
        """
        Grafica el historial de entrenamiento de un modelo Keras,
        mostrando accuracy, loss, precision y recall para entrenamiento y validación.
        
        Args:
            history: Objeto History retornado por Keras al entrenar un modelo.
            save_path (str, opcional): Ruta para guardar la imagen resultante.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Train', color=self.colors['primary'])
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation', color=self.colors['secondary'])
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Train', color=self.colors['primary'])
        axes[0, 1].plot(history.history['val_loss'], label='Validation', color=self.colors['secondary'])
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision (si disponible)
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Train', color=self.colors['primary'])
            axes[1, 0].plot(history.history['val_precision'], label='Validation', color=self.colors['secondary'])
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall (si disponible)
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Train', color=self.colors['primary'])
            axes[1, 1].plot(history.history['val_recall'], label='Validation', color=self.colors['secondary'])
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Grafica la matriz de confusión con etiquetas claras para clases positivas y negativas.
        
        Args:
            cm (np.array): Matriz de confusión (2x2).
            save_path (str, opcional): Ruta para guardar la imagen resultante.
        """
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, fpr, tpr, roc_auc, save_path=None):
        """
        Grafica la curva ROC con el área bajo la curva (AUC).
        
        Args:
            fpr (np.array): Tasa de falsos positivos.
            tpr (np.array): Tasa de verdaderos positivos.
            roc_auc (float): Área bajo la curva ROC.
            save_path (str, opcional): Ruta para guardar la imagen resultante.
        """
        plt.figure(figsize=(8, 6))
        
        plt.plot(fpr, tpr, color=self.colors['primary'], 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, importance_df, save_path=None):
        """
        Visualiza la importancia relativa de características en forma de barras horizontales
        y la distribución porcentual de las 5 características más importantes en un gráfico de pastel.
        
        Args:
            importance_df (pd.DataFrame): DataFrame con columnas ['feature', 'importance'] ordenado descendentemente.
            save_path (str, opcional): Ruta para guardar la imagen resultante.
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Barras horizontales top 10 características
        top_features = importance_df.head(10)
        
        bars = axes[0].barh(range(len(top_features)), top_features['importance'])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['feature'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title('Top 10 Feature Importance')
        axes[0].invert_yaxis()
        
        # Añadir etiquetas con valor numérico en barras
        for bar, value in zip(bars, top_features['importance']):
            axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                         f'{value:.4f}', ha='left', va='center')
        
        # Pie chart top 5 características
        top_5 = importance_df.head(5)
        axes[1].pie(top_5['importance'], labels=top_5['feature'], autopct='%1.1f%%')
        axes[1].set_title('Top 5 Features Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, data, model, processor, feature_names):
        """
        Crea un dashboard interactivo con Plotly que incluye la distribución del target,
        correlación con target, importancia de características y distribución de predicciones.
        
        Args:
            data (pd.DataFrame): Dataset original.
            model: Modelo entrenado que posee método predict.
            processor: Objeto para preprocesar datos antes de predecir.
            feature_names (list): Lista con nombres de las características usadas por el modelo.
        """
        # Crear una figura con subplots organizados en 2x2
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución del Target', 'Correlación con Target', 
                          'Importancia de Características', 'Predicciones por Probabilidad'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "histogram"}]]
        )
        
        # 1. Gráfico circular del target
        target_counts = data['default'].value_counts()
        fig.add_trace(
            go.Pie(labels=['No Default', 'Default'], values=target_counts.values,
                   name="Target Distribution"),
            row=1, col=1
        )
        
        # 2. Barras de correlación absoluta con target
        correlation_matrix = data.corr()
        target_corr = correlation_matrix['default'].abs().sort_values(ascending=False)
        target_corr = target_corr[target_corr.index != 'default']
        
        fig.add_trace(
            go.Bar(x=target_corr.index, y=target_corr.values,
                   name="Target Correlation"),
            row=1, col=2
        )
        
        # 3. Importancia simulada de características (en un caso real se usa análisis real)
        feature_importance = np.random.rand(len(feature_names))
        feature_importance = feature_importance / feature_importance.sum()
        
        fig.add_trace(
            go.Bar(x=feature_names, y=feature_importance,
                   name="Feature Importance"),
            row=2, col=1
        )
        
        # 4. Distribución de predicciones sobre una muestra de datos
        sample_data = data.sample(min(1000, len(data)))
        X_sample = processor.prepare_data(sample_data)[0]  # Solo características
        predictions = model.predict(X_sample)
        
        fig.add_trace(
            go.Histogram(x=predictions.flatten(), nbinsx=30,
                        name="Prediction Distribution"),
            row=2, col=2
        )
        
        # Configurar layout general
        fig.update_layout(
            title_text="Dashboard de Aprobación de Crédito",
            showlegend=False,
            height=800
        )
        
        fig.show()
    
    def plot_bias_analysis(self, bias_results, save_path=None):
        """
        Visualiza el análisis de sesgos para diferentes características,
        mostrando accuracy, precision y recall por valor categórico de cada característica.
        
        Args:
            bias_results (dict): Diccionario con resultados del análisis de sesgos,
                                 estructura: {feature: {value: {metric: score}}}
            save_path (str, opcional): Ruta para guardar la imagen resultante.
        """
        if not bias_results:
            print("No hay datos de análisis de sesgos disponibles")
            return
        
        n_features = len(bias_results)
        fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 6))
        
        if n_features == 1:
            axes = [axes]
        
        for i, (feature_name, metrics) in enumerate(bias_results.items()):
            values = list(metrics.keys())
            accuracies = [metrics[v]['accuracy'] for v in values]
            precisions = [metrics[v]['precision'] for v in values]
            recalls = [metrics[v]['recall'] for v in values]
            
            x = np.arange(len(values))
            width = 0.25
            
            axes[i].bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
            axes[i].bar(x, precisions, width, label='Precision', alpha=0.8)
            axes[i].bar(x + width, recalls, width, label='Recall', alpha=0.8)
            
            axes[i].set_xlabel(feature_name)
            axes[i].set_ylabel('Score')
            axes[i].set_title(f'Bias Analysis - {feature_name}')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(values)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

def main():
    """
    Función principal para demostrar las visualizaciones.
    """
    print("Módulo de visualización para el sistema de aprobación de crédito")
    print("Use las funciones de CreditVisualizer para crear gráficos")

if __name__ == "__main__":
    main()
