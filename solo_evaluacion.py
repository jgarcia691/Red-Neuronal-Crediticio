import pickle
from evaluacion import ejecutar_evaluacion
import matplotlib.pyplot as plt
import numpy as np

# Cargar objetos guardados
with open('modelo_entrenado.pkl', 'rb') as f:
    modelo = pickle.load(f)
with open('procesador.pkl', 'rb') as f:
    procesador = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print("=== EVALUACIÓN DEL MODELO ===")
ejecutar_evaluacion(modelo, procesador, X_test, y_test, feature_names)

# Graficar cantidad de créditos aprobados y rechazados
y_pred = modelo.predict(X_test)
if hasattr(y_pred, 'numpy'):
    y_pred = y_pred.numpy()
# Si es probabilidad, convertir a clases
if y_pred.ndim > 1 or (y_pred.max() > 1 or y_pred.min() < 0):
    y_pred = (y_pred > 0.5).astype(int)

aprobados = np.sum(y_pred == 1)
rechazados = np.sum(y_pred == 0)

plt.bar(['Aprobados', 'Rechazados'], [aprobados, rechazados], color=['green', 'red'])
plt.title('Cantidad de créditos aprobados y rechazados')
plt.ylabel('Cantidad')
plt.savefig('aprobados_vs_rechazados.png')
plt.show()
