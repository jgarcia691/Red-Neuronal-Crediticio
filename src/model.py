"""
Este módulo crea una red neuronal que ayuda a decidir si una persona es buena o mala para pedir un crédito. 
Esta red está hecha paso a paso, sin usar trucos complicados.

Imagina que la red es como un conjunto de pequeñas "cajitas" llamadas neuronas, que se conectan entre sí para aprender y tomar decisiones.
"""

import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """
    Esta clase representa una neurona, que es como una pequeña cajita que recibe información, la procesa y da un resultado.
    """
    def __init__(self, input_size, activation='relu'):
        """
        Aquí preparamos la neurona para trabajar. 

        - input_size: Cuántos datos entran a la neurona (como el número de preguntas que recibe).
        - activation: Cómo la neurona decide qué hacer con la información que recibe (es como si la neurona tuviera una regla para pensar).
        """
        # Creamos los "pesos", que son números que dicen cuánto importa cada dato.
        # También ponemos un "bias", que es un número extra para ayudar a decidir.
        if activation == 'relu':
            self.weights = np.random.randn(input_size) * np.sqrt(2.0 / input_size)
        else:
            self.weights = np.random.randn(input_size) * np.sqrt(1.0 / input_size)

        self.bias = 0.0
        self.activation = activation
        self.input = None  # Guardamos lo que entra para usarlo después
        self.output = None  # Guardamos lo que sale después de pensar
        self.delta = None  # Esto es para aprender de los errores

    def forward(self, inputs):
        """
        Aquí la neurona recibe datos, los combina usando sus pesos y bias, y decide qué sacar.

        Recuerda que es como si la neurona pusiera números en una balanza para decidir si "enciende" o no.
        """
        self.input = inputs
        self.z = np.dot(self.weights, inputs) + self.bias  # Suma ponderada
        self.output = self._activate(self.z)  # Aplica la regla para decidir
        return self.output

    def _activate(self, z):
        """
        Esta función es la regla de la neurona para decidir qué salida dar, dependiendo de cómo le llegó la información.
        """
        if self.activation == 'relu':
            return np.maximum(0, z)  # Si el número es negativo, saca 0; si no, saca el mismo número
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))  # Saca un número entre 0 y 1, como una probabilidad
        elif self.activation == 'tanh':
            return np.tanh(z)  # Saca un número entre -1 y 1
        else:
            return z  # No cambia nada (función identidad)

    def _activate_derivative(self, z):
        """
        Esta es la forma en que la neurona se prepara para aprender cuando se equivoca, viendo qué tanto cambió su salida.
        """
        if self.activation == 'relu':
            return np.where(z > 0, 1, 0)  # Si z > 0, dice 1, sino 0
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-z))
            return s * (1 - s)  # Fórmula especial para sigmoid
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2  # Fórmula especial para tanh
        else:
            return 1

class Layer:
    """
    Una capa es un grupo de neuronas que trabajan juntas.

    Cada capa recibe información, la procesa con todas sus neuronas, y manda la respuesta a la siguiente capa.
    """
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Creamos una capa con muchas neuronas.

        - input_size: cuántos datos entran.
        - output_size: cuántas neuronas hay en la capa.
        """
        self.neurons = [Neuron(input_size, activation) for _ in range(output_size)]
        self.input = None  # Guardamos lo que entra a la capa
        self.output = None  # Guardamos lo que sale de la capa

    def forward(self, inputs):
        """
        Cada neurona de la capa procesa la información, y la capa junta todas esas respuestas.
        """
        self.input = inputs
        out = np.array([neuron.forward(inputs) for neuron in self.neurons])
        if out.size == 1:
            return out.item()  # Si es solo un número, lo devuelve así
        return out

    def backward(self, delta_next, learning_rate):
        """
        Aquí la capa aprende. 

        Recibe cuánto se equivocó la capa siguiente (delta_next) y usa eso para ajustar sus pesos y bias.
        """
        if not isinstance(delta_next, np.ndarray):
            delta_next = np.array([delta_next])
        if delta_next.size == 1 and len(self.neurons) > 1:
            delta_next = np.full(len(self.neurons), delta_next[0])
        elif delta_next.size != len(self.neurons):
            delta_next = np.resize(delta_next, len(self.neurons))

        delta_current = np.zeros(len(self.neurons))
        for i, neuron in enumerate(self.neurons):
            neuron.delta = delta_next[i] * neuron._activate_derivative(neuron.z)
            neuron.weights -= learning_rate * neuron.delta * neuron.input  # Cambia los pesos para aprender
            neuron.bias -= learning_rate * neuron.delta  # Cambia el bias
            delta_current[i] = neuron.delta
        return delta_current

class CreditMLP:
    """
    Esta es la red neuronal completa que usamos para decidir si alguien es buen o mal candidato para un crédito.

    Tiene varias capas: las ocultas que aprenden cosas complejas, y la capa final que decide "sí" o "no".
    """
    def __init__(self, input_dim, hidden_layer_sizes=(64,), activation='relu',
                 learning_rate=0.001, max_iter=200, random_state=42):
        """
        Aquí armamos la red con las capas que queremos y configuramos cómo va a aprender.

        - input_dim: cuántos datos tiene cada persona (por ejemplo, edad, salario, etc).
        - hidden_layer_sizes: cuántas neuronas hay en cada capa oculta.
        - learning_rate: qué tan rápido aprende.
        - max_iter: cuántas veces revisa todos los datos para aprender.
        """
        np.random.seed(random_state)
        self.input_dim = input_dim
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.random_state = random_state

        self.layers = []

        # Creamos todas las capas ocultas
        prev_size = input_dim
        for hidden_size in hidden_layer_sizes:
            self.layers.append(Layer(prev_size, hidden_size, activation))
            prev_size = hidden_size

        # Capa final que da una probabilidad (entre 0 y 1)
        self.layers.append(Layer(prev_size, 1, 'sigmoid'))

        self.training_loss = []  # Guarda cómo va mejorando
        self.validation_loss = []  # Guarda qué tan bien va con datos que no vio

    def forward(self, X):
        """
        Pasamos los datos desde la primera capa hasta la última para obtener una predicción.
        """
        current_input = np.asarray(X).flatten()
        for layer in self.layers:
            current_input = layer.forward(current_input)
            if not isinstance(current_input, np.ndarray) and layer != self.layers[-1]:
                current_input = np.array([current_input])
        return current_input

    def backward(self, X, y, y_pred):
        """
        Aquí la red aprende viendo cuánto se equivocó y ajustando sus pesos.
        """
        m = X.shape[0] if hasattr(X, 'shape') else 1
        delta = (y_pred - y) / m
        for layer in reversed(self.layers):
            delta = layer.backward(delta, self.learning_rate)

    def compute_loss(self, y_true, y_pred):
        """
        Calcula qué tan mal lo hizo la red usando una fórmula especial para clasificación.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y, validation_split=0.2):
        """
        Enseña a la red a partir de ejemplos.

        Divide los datos en dos grupos: uno para aprender y otro para ver qué tan bien aprendió.
        """
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        print(f"Entrenando con {len(X_train)} ejemplos, validando con {len(X_val)} ejemplos")
        for epoch in range(self.max_iter):
            total_loss = 0
            for i in range(len(X_train)):
                xi = np.asarray(X_train.iloc[i] if hasattr(X_train, 'iloc') else X_train[i]).flatten()
                yi = float(y_train.iloc[i] if hasattr(y_train, 'iloc') else y_train[i])
                y_pred = self.forward(xi)
                self.backward(xi, yi, y_pred)
                total_loss += self.compute_loss(yi, y_pred)

            avg_loss = total_loss / len(X_train)
            self.training_loss.append(avg_loss)

            val_loss = 0
            for i in range(len(X_val)):
                xi = np.asarray(X_val.iloc[i] if hasattr(X_val, 'iloc') else X_val[i]).flatten()
                yi = float(y_val.iloc[i] if hasattr(y_val, 'iloc') else y_val[i])
                y_pred_val = self.forward(xi)
                val_loss += self.compute_loss(yi, y_pred_val)

            val_loss /= len(X_val)
            self.validation_loss.append(val_loss)

            if (epoch + 1) % 20 == 0:
                print(f"Época {epoch + 1}/{self.max_iter} - Pérdida entrenamiento: {avg_loss:.4f} - Pérdida validación: {val_loss:.4f}")

    def predict(self, X):
        """
        Después de aprender, la red puede decir si alguien es buen candidato (1) o no (0).
        """
        predictions = []
        for i in range(len(X)):
            xi = np.asarray(X.iloc[i] if hasattr(X, 'iloc') else X[i]).flatten()
            pred = self.forward(xi)
            predictions.append(1 if pred > 0.5 else 0)
        return np.array(predictions)

    def predict_proba(self, X):
        """
        También puede dar la probabilidad de que alguien sea buen candidato (un número entre 0 y 1).
        """
        predictions = []
        for i in range(len(X)):
            xi = np.asarray(X.iloc[i] if hasattr(X, 'iloc') else X[i]).flatten()
            pred = self.forward(xi)
            predictions.append(pred)
        return np.array(predictions)

    def get_model_summary(self):
        """
        Dice cuántas capas y conexiones tiene la red, para entender qué tan grande es.
        """
        total_params = 0
        prev_size = self.input_dim
        for i, layer in enumerate(self.layers):
            output_size = len(layer.neurons)
            params = (prev_size + 1) * output_size
            total_params += params
            print(f"Capa {i + 1}: {prev_size} → {output_size} neuronas ({params} conexiones)")
            prev_size = output_size
        print(f"Total de conexiones: {total_params}")
        return f"MLP Manual - {len(self.layers)} capas, {total_params} conexiones"

    def get_feature_importance(self, feature_names=None):
        """
        Nos dice qué datos son más importantes para la red al decidir.

        Mira cuánto pesan los datos en la primera capa para entender esto.
        """
        if not self.layers:
            raise ValueError("El modelo debe estar entrenado primero")

        first_layer = self.layers[0]
        importances = np.zeros(self.input_dim)
        for neuron in first_layer.neurons:
            importances += np.abs(neuron.weights)
        importances /= len(first_layer.neurons)

        if feature_names is None:
            feature_names = [f'Dato_{i}' for i in range(len(importances))]

        return dict(zip(feature_names, importances))

    def plot_training_history(self):
        """
        Muestra un dibujo con cómo la red fue aprendiendo y mejorando con el tiempo.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_loss, label='Pérdida entrenamiento', color='blue')
        plt.plot(self.validation_loss, label='Pérdida validación', color='red')
        plt.title('Cómo mejoró la red con el entrenamiento')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.grid(True)
        plt.show()

    def score(self, X, y):
        """
        Calcula qué tan bien adivina la red, diciéndonos qué porcentaje de respuestas son correctas.
        """
        y_pred = self.predict(X)
        return (y_pred == y).mean()

# Función extra para crear el modelo con una configuración
def create_model_from_config(config):
    """
    Crea la red usando una lista de instrucciones (configuración).
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
    Aquí mostramos cómo usar la red neuronal para entrenarla y probarla con datos falsos.
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
    print(f"Precisión en test: {accuracy:.4f}")

    importance = model.get_feature_importance(feature_names)
    print("Datos más importantes (Top 5):")
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {feature}: {imp:.4f}")

if __name__ == "__main__":
    main()
