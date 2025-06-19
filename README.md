# Sistema de Aprobación de Crédito con Red Neuronal

## Descripción del Proyecto

Este proyecto implementa un sistema de machine learning para la aprobación de créditos y predicción de incumplimiento de pago (credit default) utilizando una red neuronal artificial (MLP - Multi-Layer Perceptron) de una sola capa oculta, implementada con scikit-learn.

### Dominio
- **Finanzas / Riesgo Crediticio / FinTech**
- Evaluación automatizada de solicitudes de préstamo
- Predicción de probabilidad de incumplimiento

### Problema a Resolver
Un banco o empresa FinTech debe decidir si aprueba o no la solicitud de préstamo de un cliente. El sistema evalúa el perfil del solicitante y predice la probabilidad de que incumpla con el pago del crédito.

### Clasificación Binaria
- **"Buen" riesgo crediticio**: Probablemente pagará
- **"Mal" riesgo crediticio**: Probablemente no pagará

## Estructura del Proyecto

```
machineLearning/
├── data/                   # Datasets y datos de ejemplo
├── models/                 # Modelos entrenados
├── notebooks/              # Jupyter notebooks para análisis
├── src/                    # Código fuente
│   ├── data_processing.py  # Procesamiento de datos
│   ├── model.py           # Definición del modelo MLP (scikit-learn)
│   ├── training.py        # Entrenamiento del modelo
│   └── evaluation.py      # Evaluación y métricas
├── utils/                  # Utilidades y funciones auxiliares
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

## Características del Sistema

### Aspectos Técnicos
- **Algoritmo**: Multi-Layer Perceptron (MLP) de una sola capa oculta
- **Tipo de Problema**: Clasificación Binaria
- **Framework**: scikit-learn
- **Preprocesamiento**: Scikit-learn

### Aspectos Éticos y de Negocio
- **Interpretabilidad**: Análisis de importancia de características
- **Sesgos**: Detección y mitigación de sesgos en datos
- **Transparencia**: Explicabilidad de decisiones automatizadas
- **Cumplimiento**: Adherencia a regulaciones financieras

## Instalación y Uso

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecutar el notebook principal**:
   ```bash
   jupyter notebook notebooks/credit_approval_analysis.ipynb
   ```

3. **Entrenar el modelo**:
   ```bash
   python src/training.py
   ```

## Métricas de Evaluación

- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión en predicción de malos pagadores
- **Recall**: Sensibilidad en detección de malos pagadores
- **F1-Score**: Media armónica de precisión y recall
- **ROC-AUC**: Área bajo la curva ROC
- **Confusion Matrix**: Matriz de confusión

## Consideraciones Éticas

- **Sesgos en Datos**: Análisis de representatividad demográfica
- **Transparencia**: Explicabilidad de decisiones
- **Justicia**: Evaluación de impacto en diferentes grupos
- **Privacidad**: Protección de datos personales

## Aplicaciones en la Industria

- **Scoring Crediticio**: Evaluación automática de riesgo
- **Aprobación de Préstamos**: Decisión automatizada
- **Gestión de Riesgo**: Monitoreo continuo de cartera
- **Compliance**: Cumplimiento regulatorio 