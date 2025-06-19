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


## Instalación y Uso

1. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ejecucion del modelo**
   ```bash
   python main.py
   ```

