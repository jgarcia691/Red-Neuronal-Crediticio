"""
Módulo que define la arquitectura de la red neuronal MLP para clasificación de crédito implementada desde cero.
"""

import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """
    Una neurona individual con activación y gradientes.
    """
    def __init__(self, input_size, activation='relu'):
        """
        Inicializa una neurona.
        Args:
            input_size (int): Número de entradas
            activation (str): Función de activación ('relu', 'sigmoid', 'tanh')
        """
        # Inicialización He para ReLU, Xavier para otras
        if activation == 'relu':
            self.weights = np.random.randn(input_size) * np.sqrt(2.0 / input_size)
        else:
            self.weights = np.random.randn(input_size) * np.sqrt(1.0 / input_size)
        
        self.bias = 0.0
        self.activation = activation
        self.input = None
        self.output = None
        self.delta = None
        
    def forward(self, inputs):
        """
        Propagación hacia adelante.
        """
        self.input = inputs
        self.z = np.dot(self.weights, inputs) + self.bias
        self.output = self._activate(self.z)
        return self.output
    
    def _activate(self, z):
        """
        Aplica la función de activación.
        """
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
        Derivada de la función de activación.
        """
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
        Inicializa una capa.
        Args:
            input_size (int): Número de entradas
            output_size (int): Número de neuronas en la capa
            activation (str): Función de activación
        """
        self.neurons = [Neuron(input_size, activation) for _ in range(output_size)]
        self.input = None
        self.output = None
        
    def forward(self, inputs):
        """
        Propagación hacia adelante de toda la capa.
        """
        self.input = inputs
        out = np.array([neuron.forward(inputs) for neuron in self.neurons])
        # Si la capa tiene una sola neurona, devolver escalar
        if out.size == 1:
            return out.item()
        return out
    
    def backward(self, delta_next, learning_rate):
        """
        Backpropagation para esta capa.
        """
        # Si delta_next es escalar, convertirlo a array
        if not isinstance(delta_next, np.ndarray):
            delta_next = np.array([delta_next])
        # Ajustar delta_next para que tenga el tamaño correcto
        if delta_next.size == 1 and len(self.neurons) > 1:
            delta_next = np.full(len(self.neurons), delta_next[0])
        elif delta_next.size != len(self.neurons):
            # Si el tamaño sigue sin coincidir, hacer broadcast seguro
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
        Inicializa la arquitectura del MLP.
        Args:
            input_dim (int): Dimensión de entrada
            hidden_layer_sizes (tuple): Tamaño de las capas ocultas
            activation (str): Función de activación
            learning_rate (float): Tasa de aprendizaje
            max_iter (int): Número máximo de iteraciones
            random_state (int): Semilla
        """
        np.random.seed(random_state)
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Construir la arquitectura
        self.layers = []
        
        # Capas ocultas
        prev_size = input_dim
        for hidden_size in hidden_layer_sizes:
            self.layers.append(Layer(prev_size, hidden_size, activation))
            prev_size = hidden_size
        
        # Capa de salida (sigmoid para clasificación binaria)
        self.layers.append(Layer(prev_size, 1, 'sigmoid'))
        
        # Historial de entrenamiento
        self.training_loss = []
        self.validation_loss = []
        
    def forward(self, X):
        """
        Propagación hacia adelante.
        """
        current_input = np.asarray(X).flatten()
        for layer in self.layers:
            current_input = layer.forward(current_input)
            # Si la salida es un escalar, conviértelo a array para la siguiente capa (excepto la última)
            if not isinstance(current_input, np.ndarray) and layer != self.layers[-1]:
                current_input = np.array([current_input])
        return current_input
    
    def backward(self, X, y, y_pred):
        """
        Backpropagation.
        """
        # Calcular error de salida
        m = X.shape[0]
        delta = (y_pred - y) / m
        
        # Backpropagation a través de las capas
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate)
    
    def compute_loss(self, y_true, y_pred):
        """
        Calcula la pérdida (binary cross-entropy).
        """
        epsilon = 1e-15  # Para evitar log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y, validation_split=0.2):
        """
        Entrena el modelo usando backpropagation.
        """
        # Dividir datos de validación
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        print(f"Entrenando con {len(X_train)} muestras, validando con {len(X_val)} muestras")
        for epoch in range(self.max_iter):
            # Entrenamiento
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
                # Forward pass
                y_pred = self.forward(xi)
                # Backward pass
                self.backward(xi, yi, y_pred)
                total_loss += self.compute_loss(yi, y_pred)
            # Calcular pérdida promedio
            avg_loss = total_loss / len(X_train)
            self.training_loss.append(avg_loss)
            # Validación
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
            # Mostrar progreso cada 20 épocas
            if (epoch + 1) % 20 == 0:
                print(f"Época {epoch + 1}/{self.max_iter} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    def predict(self, X):
        """
        Predice la clase (0 o 1).
        """
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
        Predice la probabilidad de la clase positiva.
        """
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
        Devuelve un resumen del modelo.
        """
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
        Devuelve la importancia de las características basada en los pesos de la primera capa.
        """
        if not self.layers:
            raise ValueError("El modelo debe estar entrenado primero")
        
        # Usar los pesos de la primera capa
        first_layer = self.layers[0]
        importances = np.zeros(self.input_dim)
        
        for neuron in first_layer.neurons:
            importances += np.abs(neuron.weights)
        
        importances /= len(first_layer.neurons)  # Promedio
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        return dict(zip(feature_names, importances))
    
    def plot_training_history(self):
        """
        Grafica el historial de entrenamiento.
        """
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
        Calcula el accuracy score para compatibilidad con scikit-learn.
        
        Args:
            X: Features
            y: Labels verdaderos
            
        Returns:
            float: Accuracy score
        """
        y_pred = self.predict(X)
        return (y_pred == y).mean()

def create_model_from_config(config):
    """
    Crea un modelo basado en una configuración.
    Args:
        config (dict): Configuración del modelo
    Returns:
        CreditMLP: Instancia del modelo
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

def main():
    """
    Demostración de la creación y entrenamiento del modelo.
    """
    from src.data_processing import CreditDataProcessor
    processor = CreditDataProcessor()
    data = processor.generate_sample_data(n_samples=500)
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(data)
    
    model = CreditMLP(input_dim=X_train.shape[1], max_iter=50)
    print("Resumen del modelo:")
    model.get_model_summary()
    
    print("\nEntrenando modelo...")
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"Accuracy en test: {accuracy:.4f}")
    
    # Mostrar importancia de características
    importance = model.get_feature_importance(feature_names)
    print("Importancia de características:")
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {imp:.4f}")

if __name__ == "__main__":
    main() 