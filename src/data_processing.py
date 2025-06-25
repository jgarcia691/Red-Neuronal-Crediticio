# Este archivo contiene funciones para preparar datos de personas que piden crédito (préstamos).
# Con estos datos se entrenará un modelo para predecir si alguien pagará o no.

import pandas as pd  # Librería para trabajar con tablas
import numpy as np  # Librería para trabajar con números
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento y prueba
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Para escalar números y convertir texto a números
from sklearn.impute import SimpleImputer  # Para llenar datos que faltan
import warnings  # Para evitar mensajes de advertencia molestos
import kagglehub  # Para descargar datasets desde Kaggle
import os  # Para trabajar con carpetas y archivos
warnings.filterwarnings('ignore')  # Apagar advertencias

# Esta clase tiene todo lo necesario para preparar los datos de crédito
class CreditDataProcessor:
    def __init__(self):
        # Objeto para escalar características numéricas
        self.scaler = StandardScaler()
        # Diccionario para guardar codificadores de texto a números
        self.label_encoders = {}
        # Objeto para rellenar valores que faltan usando la mediana
        self.imputer = SimpleImputer(strategy='median')
        # Aquí se guardarán los nombres de las columnas numéricas
        self.feature_names = None

    # Función que crea datos falsos (sintéticos) para hacer pruebas
    def generate_sample_data(self, n_samples=10000):
        np.random.seed(42)  # Semilla para que los datos sean reproducibles

        # Crear columnas de datos numéricos (edad, ingreso, puntaje, etc.)
        age = np.random.normal(35, 10, n_samples).astype(int)
        age = np.clip(age, 18, 80)  # Limitar valores extremos

        income = np.random.lognormal(10.5, 0.5, n_samples)
        income = np.clip(income, 15000, 200000)

        credit_score = np.random.normal(650, 100, n_samples).astype(int)
        credit_score = np.clip(credit_score, 300, 850)

        debt_to_income = np.random.beta(2, 5, n_samples) * 100
        debt_to_income = np.clip(debt_to_income, 0, 100)

        employment_length = np.random.exponential(5, n_samples)
        employment_length = np.clip(employment_length, 0, 30)

        loan_amount = np.random.lognormal(10, 0.8, n_samples)
        loan_amount = np.clip(loan_amount, 1000, 500000)

        loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples)

        # Crear columnas de texto (categorías)
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                     n_samples, p=[0.3, 0.4, 0.2, 0.1])

        employment_type = np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 
                                           n_samples, p=[0.6, 0.2, 0.15, 0.05])

        home_ownership = np.random.choice(['Rent', 'Own', 'Mortgage'], 
                                          n_samples, p=[0.4, 0.2, 0.4])

        purpose = np.random.choice(['Home', 'Car', 'Education', 'Business', 'Personal'], 
                                   n_samples, p=[0.3, 0.2, 0.15, 0.2, 0.15])

        # Crear una tabla (DataFrame) con todos los datos
        data = pd.DataFrame({
            'age': age,
            'income': income,
            'credit_score': credit_score,
            'debt_to_income': debt_to_income,
            'employment_length': employment_length,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'education': education,
            'employment_type': employment_type,
            'home_ownership': home_ownership,
            'purpose': purpose
        })

        # Calcular probabilidad de que no pague (default)
        default_prob = (
            1 / (1 + np.exp(-(-2 + 
                0.02 * (credit_score - 650) / 100 +
                0.01 * (income - 50000) / 50000 +
                0.03 * debt_to_income / 100 +
                -0.05 * employment_length / 10 +
                0.01 * (age - 35) / 20 +
                np.random.normal(0, 0.1, n_samples))))
        )

        # Crear columna 'default': 1 si no paga, 0 si sí paga
        data['default'] = np.random.binomial(1, default_prob, n_samples)

        return data

    # Función que carga datos desde archivo o genera datos de ejemplo
    def load_data(self, file_path=None):
        if file_path:
            try:
                data = pd.read_csv(file_path)  # Intentar cargar archivo
                print(f"Datos cargados desde: {file_path}")
            except FileNotFoundError:
                print(f"Archivo no encontrado: {file_path}")
                print("Generando datos de ejemplo...")
                data = self.generate_sample_data()
        else:
            print("Generando datos de ejemplo...")
            data = self.generate_sample_data()

        # Mostrar tamaño y distribución del target
        print(f"Dataset shape: {data.shape}")
        print(f"Distribución de target:\n{data['default'].value_counts(normalize=True)}")

        return data

    # Limpia datos: borra duplicados, muestra faltantes
    def clean_data(self, data):
        print("Limpiando datos...")
        df = data.copy()

        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Valores faltantes encontrados:\n{missing_values[missing_values > 0]}")

        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            print(f"Eliminados {initial_rows - len(df)} registros duplicados")

        print(f"Tipos de datos:\n{df.dtypes}")

        return df

    # Convierte columnas de texto en números usando LabelEncoder
    def encode_categorical_features(self, data):
        print("Codificando variables categóricas...")
        df = data.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            if col != 'default':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Codificada columna: {col}")

        return df

    # Escala los números para que estén en la misma escala
    def scale_numerical_features(self, data, fit=True):
        print("Escalando características numéricas...")
        df = data.copy()
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        numerical_columns = [col for col in numerical_columns if col != 'default']

        if fit:
            df[numerical_columns] = self.scaler.fit_transform(df[numerical_columns])
        else:
            df[numerical_columns] = self.scaler.transform(df[numerical_columns])

        self.feature_names = numerical_columns
        return df

    # Descarga el dataset 'german-credit' desde Kaggle
    def download_german_credit_dataset(self):
        print("Descargando el dataset de Kaggle (uciml/german-credit)...")
        path = kagglehub.dataset_download("uciml/german-credit")
        print("Ruta de los archivos del dataset:", path)

        for file in os.listdir(path):
            if file.endswith('.csv'):
                return os.path.join(path, file)

        raise FileNotFoundError("No se encontró un archivo CSV en el dataset descargado.")

    # Agrega una columna 'default' basada en reglas simples
    def add_default_column(self, df):
        conditions = [
            df['Age'] > 25,
            df['Job'] >= 2,
            df['Housing'] == 'own',
            df['Saving accounts'] == 'rich',
            df['Checking account'].isin(['moderate', 'rich']),
            (df['Duration'] > 6) & (df['Duration'] < 24)
        ]
        num_conditions = sum(conditions)
        df['default'] = (num_conditions >= 3).astype(int)
        return df

    # Carga el dataset real desde la carpeta o desde Kaggle
    def load_german_credit_data(self):
        local_path = 'data/german_credit_data.csv'
        if os.path.exists(local_path):
            data = pd.read_csv(local_path)
            print(f"Datos cargados desde: {local_path}")
        else:
            csv_path = self.download_german_credit_dataset()
            data = pd.read_csv(csv_path)
            print(f"Datos cargados desde Kaggle: {csv_path}")

        data = self.add_default_column(data)
        return data

    # Prepara los datos completamente para entrenar un modelo
    def prepare_data(self, data, test_size=0.4, random_state=42):
        print("Preparando datos para entrenamiento...")
        df = self.clean_data(data)
        X = df.drop('default', axis=1)  # Características
        y = df['default']  # Target

        X = self.encode_categorical_features(X)
        X = self.scale_numerical_features(X, fit=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Conjunto de entrenamiento: {X_train.shape}")
        print(f"Conjunto de prueba: {X_test.shape}")
        print(f"Características utilizadas: {list(X.columns)}")

        return X_train, X_test, y_train, y_test, list(X.columns)

    # Analiza qué columnas tienen más relación con el resultado
    def get_feature_importance_analysis(self, data):
        print("Analizando importancia de características...")
        df = data.copy()
        correlations = df.corr()['default'].abs().sort_values(ascending=False)

        default_stats = df[df['default'] == 1].describe()
        non_default_stats = df[df['default'] == 0].describe()

        numerical_features = df.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col != 'default']

        feature_analysis = {
            'correlations': correlations,
            'default_stats': default_stats,
            'non_default_stats': non_default_stats,
            'numerical_features': numerical_features
        }

        return feature_analysis

# Esta función se ejecuta si corres este archivo directamente
def main():
    processor = CreditDataProcessor()

    # Generar datos falsos
    data = processor.generate_sample_data(n_samples=5000)

    # Guardar los datos en un archivo CSV
    data.to_csv('data/credit_data_sample.csv', index=False)
    print("Datos de ejemplo guardados en 'data/credit_data_sample.csv'")

    # Preparar los datos para el modelo
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(data)

    # Guardar datos ya procesados
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    print("Datos procesados guardados en el directorio 'data/'")

    # Mostrar análisis de importancia de características
    analysis = processor.get_feature_importance_analysis(data)
    print("\nCorrelaciones con el target:")
    print(analysis['correlations'])

# Llama a la función main si este archivo es ejecutado directamente
if __name__ == "__main__":
    main()