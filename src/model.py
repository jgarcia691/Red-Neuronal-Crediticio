"""
Módulo que define la arquitectura de la red neuronal MLP para clasificación de crédito implementada desde cero.
"""

# Importa numpy para operaciones numéricas y matplotlib para visualización
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """
    Una neurona individual con activación y gradientes.
    """
    def __init__(self, input_size, activation='relu'):
        """
        Inicializa los pesos y el bias de la neurona según la función de activación.
        - input_size: número de entradas a la neurona.
        - activation: función de activación ('relu', 'sigmoid', 'tanh').
        """
        # Inicializa los pesos y el bias de la neurona según la función de activación
        if activation == 'relu':
            self.weights = np.random.randn(input_size) * np.sqrt(2.0 / input_size)  # He
        else:
            self.weights = np.random.randn(input_size) * np.sqrt(1.0 / input_size)  # Xavier
        self.bias = 0.0
        self.activation = activation
        self.input = None
        self.output = None
        self.delta = None
    
    def forward(self, inputs):
        """
        Calcula la salida de la neurona aplicando pesos, bias y función de activación.
        - inputs: vector de entrada.
        Devuelve la salida activada de la neurona.
        """
        # Calcula la salida de la neurona aplicando pesos, bias y función de activación
        self.input = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.output = self._activate(self.z)
        return self.output
    
    def _activate(self, z):
        """
        Aplica la función de activación seleccionada sobre el valor z.
        - z: valor antes de la activación.
        Devuelve el valor activado.
        """
        # Aplica la función de activación seleccionada
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            return z
    
    def _activate_derivative(self, z):
        """
        Calcula la derivada de la función de activación para z.
        - z: valor antes de la activación.
        Devuelve la derivada para usar en backpropagation.
        """
        # Calcula la derivada de la función de activación
        if self.activation == 'relu':
            return np.where(z > 0, 1, 0)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z)**2
        else:
            return 1

class Layer:
    """
    Una capa de neuronas.
    """
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Crea una lista de neuronas para la capa.
        - input_size: número de entradas a la capa.
        - output_size: número de neuronas en la capa.
        - activation: función de activación de las neuronas.
        """
        # Crea una lista de neuronas para la capa
        self.neurons = [Neuron(input_size, activation) for _ in range(output_size)]
        self.input = None
        self.output = None
    
    def forward(self, inputs):
        """
        Propaga la entrada a través de todas las neuronas de la capa.
        - inputs: vector de entrada.
        Devuelve un vector con la salida de cada neurona.
        """
        # Propaga la entrada a través de todas las neuronas de la capa
        self.input = inputs
        out = np.array([neuron.forward(inputs) for neuron in self.neurons])
        if out.size == 1:
            return out.item()
        return out
    
    def backward(self, delta_next, learning_rate):
        """
        Realiza el paso de backpropagation para la capa.
        - delta_next: gradiente de la siguiente capa.
        - learning_rate: tasa de aprendizaje para actualizar los pesos.
        Devuelve el gradiente para la capa anterior.
        """
        # Realiza el paso de backpropagation para la capa
        if not isinstance(delta_next, np.ndarray):
            delta_next = np.array([delta_next])
        if delta_next.size == 1 and len(self.neurons) > 1:
            delta_next = np.full(len(self.neurons), delta_next[0])
        elif delta_next.size != len(self.neurons):
            delta_next = np.resize(delta_next, len(self.neurons))
        delta_current = np.zeros(len(self.neurons))
        for i, neuron in enumerate(self.neurons):
            neuron.delta = delta_next[i] * neuron._activate_derivative(neuron.z)
            neuron.weights -= learning_rate * neuron.delta * neuron.input
            neuron.bias -= learning_rate * neuron.delta
            delta_current[i] = neuron.delta
        return delta_current

class CreditMLP:
    """
    MLP implementado desde cero para clasificación de riesgo crediticio.
    """
    def __init__(self, input_dim, hidden_layer_sizes=(64,), activation='relu', learning_rate=0.001, max_iter=200, random_state=42):
        """
        Inicializa la arquitectura del MLP y sus hiperparámetros.
        - input_dim: número de características de entrada.
        - hidden_layer_sizes: tupla con el tamaño de cada capa oculta.
        - activation: función de activación de las capas ocultas.
        - learning_rate: tasa de aprendizaje.
        - max_iter: número de épocas de entrenamiento.
        - random_state: semilla para reproducibilidad.
        """
        # Inicializa la arquitectura del MLP y sus hiperparámetros
        np.random.seed(random_state)
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        self.layers = []
        # Construye las capas ocultas
        prev_size = input_dim
        for hidden_size in hidden_layer_sizes:
            self.layers.append(Layer(prev_size, hidden_size, activation))
            prev_size = hidden_size
        # Añade la capa de salida (sigmoid para clasificación binaria)
        self.layers.append(Layer(prev_size, 1, 'sigmoid'))
        # Historial de pérdidas
        self.training_loss = []
        self.validation_loss = []
    
    def forward(self, X):
        """
        Propaga la entrada X a través de todas las capas del MLP.
        - X: vector de entrada.
        Devuelve la salida final del modelo (probabilidad).
        """
        # Propaga la entrada a través de todas las capas
        current_input = np.asarray(X).flatten()
        for layer in self.layers:
            current_input = layer.forward(current_input)
            if not isinstance(current_input, np.ndarray) and layer != self.layers[-1]:
                current_input = np.array([current_input])
        return current_input
    
    def backward(self, X, y, y_pred):
        """
        Realiza backpropagation para actualizar los pesos de todas las capas.
        - X: vector de entrada.
        - y: valor real (target).
        - y_pred: predicción del modelo.
        """
        # Realiza backpropagation para actualizar los pesos
        m = X.shape[0]
        delta = (y_pred - y) / m
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate)
    
    def compute_loss(self, y_true, y_pred):
        """
        Calcula la pérdida de entropía cruzada binaria entre y_true y y_pred.
        - y_true: valor real.
        - y_pred: valor predicho.
        Devuelve el valor de la pérdida.
        """
        # Calcula la pérdida de entropía cruzada binaria
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y, validation_split=0.2):
        """
        Entrena el modelo usando backpropagation y guarda el historial de pérdidas.
        - X: matriz de características de entrenamiento.
        - y: vector de etiquetas.
        - validation_split: proporción de datos para validación.
        """
        # Entrena el modelo usando backpropagation y guarda el historial de pérdidas
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        print(f"Entrenando con {len(X_train)} muestras, validando con {len(X_val)} muestras")
        for epoch in range(self.max_iter):
            total_loss = 0
            for i in range(len(X_train)):
                if hasattr(X_train, 'iloc'):
                    xi = np.asarray(X_train.iloc[i]).flatten()
                else:
                    xi = np.asarray(X_train[i]).flatten()
                if hasattr(y_train, 'iloc'):
                    yi = float(np.asarray(y_train.iloc[i]).squeeze())
                else:
                    yi = float(np.asarray(y_train[i]).squeeze())
                y_pred = self.forward(xi)
                self.backward(xi, yi, y_pred)
                total_loss += self.compute_loss(yi, y_pred)
            avg_loss = total_loss / len(X_train)
            self.training_loss.append(avg_loss)
            val_loss = 0
            for i in range(len(X_val)):
                if hasattr(X_val, 'iloc'):
                    xi = np.asarray(X_val.iloc[i]).flatten()
                else:
                    xi = np.asarray(X_val[i]).flatten()
                if hasattr(y_val, 'iloc'):
                    yi = float(np.asarray(y_val.iloc[i]).squeeze())
                else:
                    yi = float(np.asarray(y_val[i]).squeeze())
                y_pred_val = self.forward(xi)
                val_loss += self.compute_loss(yi, y_pred_val)
            val_loss /= len(X_val)
            self.validation_loss.append(val_loss)
            if (epoch + 1) % 20 == 0:
                print(f"Época {epoch + 1}/{self.max_iter} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    def predict(self, X):
        """
        Predice la clase (0 o 1) para cada muestra de X.
        - X: matriz de características.
        Devuelve un array de predicciones (0 o 1).
        """
        # Predice la clase (0 o 1) para cada muestra
        predictions = []
        for i in range(len(X)):
            if hasattr(X, 'iloc'):
                xi = np.asarray(X.iloc[i]).flatten()
            else:
                xi = np.asarray(X[i]).flatten()
            pred = self.forward(xi)
            predictions.append(1 if pred > 0.5 else 0)
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Predice la probabilidad de la clase positiva para cada muestra de X.
        - X: matriz de características.
        Devuelve un array de probabilidades.
        """
        # Predice la probabilidad de la clase positiva para cada muestra
        predictions = []
        for i in range(len(X)):
            if hasattr(X, 'iloc'):
                xi = np.asarray(X.iloc[i]).flatten()
            else:
                xi = np.asarray(X[i]).flatten()
            pred = self.forward(xi)
            predictions.append(pred)
        return np.array(predictions)
    
    def get_model_summary(self):
        """
        Devuelve un resumen de la arquitectura y número de parámetros del modelo.
        Imprime la estructura de capas y el total de parámetros.
        """
        # Devuelve un resumen de la arquitectura y número de parámetros
        total_params = 0
        prev_size = self.input_dim
        for i, layer in enumerate(self.layers):
            output_size = len(layer.neurons)
            params = (prev_size + 1) * output_size  # +1 por el bias
            total_params += params
            print(f"Capa {i+1}: {prev_size} → {output_size} neuronas ({params} parámetros)")
            prev_size = output_size
        print(f"Total de parámetros: {total_params}")
        return f"MLP Manual - {len(self.layers)} capas, {total_params} parámetros"
    
    def get_feature_importance(self, feature_names=None):
        """
        Calcula la importancia de las características usando los pesos de la primera capa.
        - feature_names: lista de nombres de las características (opcional).
        Devuelve un diccionario {nombre: importancia}.
        """
        # Calcula la importancia de las características usando los pesos de la primera capa
        if not self.layers:
            raise ValueError("El modelo debe estar entrenado primero")
        first_layer = self.layers[0]
        importances = np.zeros(self.input_dim)
        for neuron in first_layer.neurons:
            importances += np.abs(neuron.weights)
        importances /= len(first_layer.neurons)
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        return dict(zip(feature_names, importances))
    
    def plot_training_history(self):
        """
        Grafica el historial de pérdidas de entrenamiento y validación.
        """
        # Grafica el historial de pérdidas de entrenamiento y validación
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_loss, label='Training Loss', color='blue')
        plt.plot(self.validation_loss, label='Validation Loss', color='red')
        plt.title('Historial de Entrenamiento')
        plt.xlabel('Época')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)
        plt.show()

    def score(self, X, y):
        """
        Calcula el accuracy para compatibilidad con scikit-learn.
        - X: matriz de características.
        - y: etiquetas verdaderas.
        Devuelve el accuracy (proporción de aciertos).
        """
        # Calcula el accuracy para compatibilidad con scikit-learn
        y_pred = self.predict(X)
        return (y_pred == y).mean()

# Función auxiliar para crear un modelo a partir de una configuración

def create_model_from_config(config):
    """
    Crea un modelo basado en una configuración.
    - config: diccionario con los parámetros del modelo.
    Devuelve una instancia de CreditMLP configurada.
    """
    input_dim = config['input_dim']
    hidden_layer_sizes = tuple(config.get('hidden_layers', [64]))
    activation = config.get('activation', 'relu')
    learning_rate = config.get('learning_rate', 0.001)
    max_iter = config.get('max_iter', 200)
    random_state = config.get('random_state', 42)
    return CreditMLP(
        input_dim=input_dim,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        learning_rate=learning_rate,
        max_iter=max_iter,
        random_state=random_state
    )