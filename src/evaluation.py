"""
Módulo de evaluación y análisis del modelo de clasificación de crédito.
Incluye análisis de interpretabilidad, sesgos y métricas avanzadas.
"""

# Importa librerías para métricas, visualización, interpretabilidad y warnings
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
    """
    
    def __init__(self, model, processor=None):
        """
        Inicializa el evaluador con el modelo entrenado y, opcionalmente, el procesador de datos.
        Prepara los atributos para almacenar los datos de prueba y los nombres de las características.
        """
        self.model = model
        self.processor = processor
        self.feature_names = None
        self.X_test = None
        self.y_test = None
    
    def set_test_data(self, X_test, y_test, feature_names=None):
        """
        Guarda los datos de prueba (X_test, y_test) y los nombres de las características
        para que el evaluador pueda usarlos en las métricas y análisis posteriores.
        """
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_test.shape[1])]
    
    def get_predictions(self, threshold=0.5):
        """
        Obtiene las probabilidades predichas por el modelo y las convierte en clases (0 o 1)
        usando un umbral (threshold). Devuelve ambas: probabilidades y clases predichas.
        """
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > threshold).astype(int)
        return y_pred_proba, y_pred
    
    def comprehensive_metrics(self, threshold=0.5):
        """
        Calcula un conjunto completo de métricas de evaluación (accuracy, precision, recall,
        specificity, f1, ROC-AUC, PR-AUC, matriz de confusión, etc.) usando los datos de prueba
        y el modelo. Devuelve un diccionario con todas las métricas.
        """
        y_pred_proba, y_pred = self.get_predictions(threshold)
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
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
        Genera varios gráficos para visualizar el desempeño del modelo: matriz de confusión,
        curva ROC, curva Precision-Recall, barras de métricas, distribución de probabilidades
        y comparación de métricas clave. Puede guardar los gráficos si se indica una ruta.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Matriz de confusión
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Curva ROC
        fpr = metrics['advanced']['false_positive_rate']
        tpr = metrics['advanced']['true_positive_rate']
        roc_auc = metrics['advanced']['roc_auc']
        
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Curva Precision-Recall
        pr_auc = metrics['advanced']['pr_auc']
        precision_curve = metrics['advanced']['precision_curve']
        recall_curve = metrics['advanced']['recall_curve']
        
        axes[0, 2].plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # 4. Métricas principales
        basic_metrics = metrics['basic']
        metric_names = list(basic_metrics.keys())[:6]  # Primeras 6 métricas
        metric_values = list(basic_metrics.values())[:6]
        
        bars = axes[1, 0].bar(metric_names, metric_values, 
                             color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'orange', 'purple'])
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar, value in zip(bars, metric_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 5. Distribución de probabilidades
        y_pred_proba = metrics['predictions']['probabilities']
        axes[1, 1].hist(y_pred_proba[self.y_test == 0], bins=30, alpha=0.7, 
                       label='No Default', color='green')
        axes[1, 1].hist(y_pred_proba[self.y_test == 1], bins=30, alpha=0.7, 
                       label='Default', color='red')
        axes[1, 1].set_title('Probability Distribution')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # 6. Comparación de métricas
        comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        comparison_values = [basic_metrics[m] for m in comparison_metrics]
        
        axes[1, 2].bar(comparison_metrics, comparison_values, 
                      color=['blue', 'green', 'orange', 'red'])
        axes[1, 2].set_title('Key Metrics Comparison')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_ylim(0, 1)
        
        for i, value in enumerate(comparison_values):
            axes[1, 2].text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluación comprehensiva guardada en: {save_path}")
        
        plt.show()
    
    def analyze_feature_importance(self, method='shap', n_samples=1000):
        """
        Analiza la importancia de las características usando el método especificado
        ('shap', 'permutation' o 'gradient'). Llama a la función interna correspondiente
        y devuelve los resultados del análisis.
        """
        print(f"Analizando importancia de características usando {method}...")
        if method == 'shap':
            return self._shap_analysis(n_samples)
        elif method == 'permutation':
            return self._permutation_importance_analysis()
        elif method == 'gradient':
            return self._gradient_importance_analysis()
        else:
            raise ValueError("Método no soportado. Use 'shap', 'permutation', o 'gradient'")
    
    def _shap_analysis(self, n_samples):
        """
        Realiza el análisis de importancia de características usando SHAP, que explica el impacto
        de cada variable en la predicción del modelo. Devuelve un DataFrame con la importancia
        de cada característica.
        """
        try:
            # Seleccionar muestra para análisis
            if n_samples < len(self.X_test):
                indices = np.random.choice(len(self.X_test), n_samples, replace=False)
                X_sample = self.X_test[indices]
            else:
                X_sample = self.X_test
            
            # Crear explainer
            explainer = shap.KernelExplainer(self.model.predict, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
            # Calcular importancia media
            feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            # Crear DataFrame con resultados
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            return {
                'method': 'shap',
                'shap_values': shap_values,
                'feature_importance': importance_df,
                'explainer': explainer
            }
        except Exception as e:
            print(f"Error en análisis SHAP: {e}")
            return None
    
    def _permutation_importance_analysis(self):
        """
        Calcula la importancia de las características usando el método de permutación de scikit-learn,
        que mide cuánto afecta la métrica de desempeño al permutar cada variable. Devuelve un DataFrame
        con la importancia de cada característica.
        """
        # Usar scikit-learn permutation importance
        result = permutation_importance(
            self.model, self.X_test, self.y_test,
            n_repeats=10, random_state=42, n_jobs=-1
        )
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std
        }).sort_values('importance', ascending=False)
        return {
            'method': 'permutation',
            'feature_importance': importance_df,
            'permutation_result': result
        }
    
    def plot_feature_importance(self, importance_analysis, save_path=None):
        """
        Genera gráficos (barras y pastel) para visualizar la importancia de las características
        según el análisis realizado. Puede guardar los gráficos si se indica una ruta.
        """
        if importance_analysis is None:
            print("No hay datos de importancia de características disponibles")
            return
        importance_df = importance_analysis['feature_importance']
        method = importance_analysis['method']
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        top_features = importance_df.head(10)
        bars = axes[0].barh(range(len(top_features)), top_features['importance'])
        axes[0].set_yticks(range(len(top_features)))
        axes[0].set_yticklabels(top_features['feature'])
        axes[0].set_xlabel('Importance')
        axes[0].set_title(f'Top 10 Feature Importance ({method.upper()})')
        axes[0].invert_yaxis()
        for i, (bar, value) in enumerate(zip(bars, top_features['importance'])):
            axes[0].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{value:.4f}', ha='left', va='center')
        top_5 = importance_df.head(5)
        axes[1].pie(top_5['importance'], labels=top_5['feature'], autopct='%1.1f%%')
        axes[1].set_title(f'Top 5 Features Distribution ({method.upper()})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Importancia de características guardada en: {save_path}")
        
        plt.show()
    
    def analyze_bias_and_fairness(self, sensitive_features=None):
        """
        Analiza el desempeño del modelo por grupos de características sensibles (por ejemplo, género, edad, etc.)
        para detectar posibles sesgos. Devuelve un diccionario con métricas por grupo.
        """
        print("Analizando sesgos y justicia del modelo...")
        if sensitive_features is None:
            print("No se proporcionaron características sensibles para análisis")
            return None
        bias_analysis = {}
        for feature_name, feature_values in sensitive_features.items():
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                feature_data = self.X_test[:, feature_idx]
                unique_values = np.unique(feature_data)
                group_metrics = {}
                for value in unique_values:
                    mask = feature_data == value
                    if np.sum(mask) > 0:
                        group_y_true = self.y_test[mask]
                        group_y_pred_proba = self.model.predict(self.X_test[mask])
                        group_y_pred = (group_y_pred_proba > 0.5).astype(int)
                        group_accuracy = np.mean(group_y_pred == group_y_true)
                        group_precision = np.mean(group_y_pred[group_y_true == 1] == 1) if np.sum(group_y_true == 1) > 0 else 0
                        group_recall = np.mean(group_y_true[group_y_pred == 1] == 1) if np.sum(group_y_pred == 1) > 0 else 0
                        group_metrics[value] = {
                            'accuracy': group_accuracy,
                            'precision': group_precision,
                            'recall': group_recall,
                            'sample_size': np.sum(mask)
                        }
                bias_analysis[feature_name] = group_metrics
        return bias_analysis
    
    def generate_explanation_report(self, sample_idx=0, save_path=None):
        """
        Genera un reporte explicativo para una muestra específica usando LIME, mostrando cómo influyen
        las características en la predicción de esa muestra. Puede guardar el reporte en un archivo JSON.
        """
        print(f"Generando explicación para muestra {sample_idx}...")
        X_sample = self.X_test[sample_idx:sample_idx+1]
        y_true = self.y_test[sample_idx]
        y_pred_proba = self.model.predict(X_sample)[0]
        y_pred = int(y_pred_proba > 0.5)
        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_test,
            feature_names=self.feature_names,
            class_names=['No Default', 'Default'],
            mode='classification'
        )
        explanation = explainer.explain_instance(
            X_sample[0], 
            self.model.predict,
            num_features=len(self.feature_names)
        )
        report = {
            'sample_index': sample_idx,
            'true_label': int(y_true),
            'predicted_label': y_pred,
            'predicted_probability': float(y_pred_proba),
            'explanation': explanation.as_list(),
            'feature_values': dict(zip(self.feature_names, X_sample[0]))
        }
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Reporte de explicación guardado en: {save_path}")
        return report

