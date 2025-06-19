#!/usr/bin/env python3
"""
Script principal para el sistema de aprobación de crédito.
Ahora solo orquesta el pipeline usando los módulos separados de entrenamiento y evaluación.
"""

import argparse
from entrenamiento import ejecutar_entrenamiento
from evaluacion import ejecutar_evaluacion

def main():
    parser = argparse.ArgumentParser(
        description='Sistema de Aprobación de Crédito con Red Neuronal (Orquestador)'
    )
    parser.add_argument('--samples', type=int, default=10000, help='Número de muestras a generar (default: 10000)')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de entrenamiento (default: 50)')
    parser.add_argument('--quick', action='store_true', help='Ejecutar en modo rápido (menos muestras y épocas)')
    parser.add_argument('--kaggle', action='store_true', help='Usar el dataset real de Kaggle (uciml/german-credit)')
    args = parser.parse_args()

    if args.quick:
        args.samples = 2000
        args.epochs = 20
        print("Ejecutando en modo rápido...")

    print("\n=== PROCEDIMIENTO DEL PIPELINE ===")
    print("1. Procesamiento y generación/carga de datos")
    print("2. Entrenamiento del modelo")
    print("3. Evaluación y visualización de resultados\n")

    # Entrenamiento
    modelo, procesador, X_test, y_test, feature_names = ejecutar_entrenamiento(
        n_samples=args.samples, epochs=args.epochs, use_kaggle_data=args.kaggle
    )

    print("\nResumen de la arquitectura del modelo:")
    modelo.get_model_summary()

    # Evaluación
    ejecutar_evaluacion(modelo, procesador, X_test, y_test, feature_names)

if __name__ == "__main__":
    main() 