"""
Módulo de procesamiento de datos para el sistema de aprobación de crédito.
Incluye funciones para cargar, limpiar, transformar y preparar los datos.
"""

# Importa librerías para manejo de datos, preprocesamiento y utilidades
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
import kagglehub
import os
warnings.filterwarnings('ignore')

class CreditDataProcessor:
    """
    Clase para procesar datos de crédito y prepararlos para el entrenamiento del modelo.
    """
    
    def __init__(self):
        # Inicializa los objetos de escalado, codificación y completado de valores
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
    
    def generate_sample_data(self, n_samples=10000):
        """
        Genera datos de ejemplo para demostración del sistema.
        
        Args:
            n_samples (int): Número de muestras a generar
            
        Returns:
            pd.DataFrame: Dataset con características de crédito
        """
        np.random.seed(42)
        
        # Características demográficas
        age = np.random.normal(35, 10, n_samples).astype(int)
        age = np.clip(age, 18, 80)
        
        income = np.random.lognormal(10.5, 0.5, n_samples)
        income = np.clip(income, 15000, 200000)
        
        # Características financieras
        credit_score = np.random.normal(650, 100, n_samples).astype(int)
        credit_score = np.clip(credit_score, 300, 850)
        
        debt_to_income = np.random.beta(2, 5, n_samples) * 100
        debt_to_income = np.clip(debt_to_income, 0, 100)
        
        employment_length = np.random.exponential(5, n_samples)
        employment_length = np.clip(employment_length, 0, 30)
        
        loan_amount = np.random.lognormal(10, 0.8, n_samples)
        loan_amount = np.clip(loan_amount, 1000, 500000)
        
        loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples)
        
        # Variables categóricas
        education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                   n_samples, p=[0.3, 0.4, 0.2, 0.1])
        
        employment_type = np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 
                                         n_samples, p=[0.6, 0.2, 0.15, 0.05])
        
        home_ownership = np.random.choice(['Rent', 'Own', 'Mortgage'], 
                                        n_samples, p=[0.4, 0.2, 0.4])
        
        purpose = np.random.choice(['Home', 'Car', 'Education', 'Business', 'Personal'], 
                                 n_samples, p=[0.3, 0.2, 0.15, 0.2, 0.15])
        
        # Crear DataFrame
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
        
        # Generar target (default) basado en características
        # Probabilidad de default basada en múltiples factores
        default_prob = (
            (1 / (1 + np.exp(-(-2 + 
                              0.02 * (credit_score - 650) / 100 +
                              0.01 * (income - 50000) / 50000 +
                              0.03 * debt_to_income / 100 +
                              -0.05 * employment_length / 10 +
                              0.01 * (age - 35) / 20 +
                              np.random.normal(0, 0.1, n_samples)))))
        )
        
        data['default'] = np.random.binomial(1, default_prob, n_samples)
        
        return data
    
    def load_data(self, file_path=None):
        """
        Carga datos desde un archivo o genera datos de ejemplo.
        
        Args:
            file_path (str): Ruta al archivo de datos (opcional)
            
        Returns:
            pd.DataFrame: Dataset cargado
        """
        if file_path:
            try:
                data = pd.read_csv(file_path)
                print(f"Datos cargados desde: {file_path}")
            except FileNotFoundError:
                print(f"Archivo no encontrado: {file_path}")
                print("Generando datos de ejemplo...")
                data = self.generate_sample_data()
        else:
            print("Generando datos de ejemplo...")
            data = self.generate_sample_data()
        
        print(f"Dataset shape: {data.shape}")
        print(f"Distribución de target:\n{data['default'].value_counts(normalize=True)}")
        
        return data
    
    def clean_data(self, data):
        """
        Limpia y prepara los datos.
        
        Args:
            data (pd.DataFrame): Dataset original
            
        Returns:
            pd.DataFrame: Dataset limpio
        """
        print("Limpiando datos...")
        
        # Crear copia para no modificar el original
        df = data.copy()
        
        # Verificar valores faltantes
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"Valores faltantes encontrados:\n{missing_values[missing_values > 0]}")
        
        # Eliminar duplicados
        initial_rows = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_rows:
            print(f"Eliminados {initial_rows - len(df)} registros duplicados")
        
        # Verificar tipos de datos
        print(f"Tipos de datos:\n{df.dtypes}")
        
        return df
    
    def encode_categorical_features(self, data):
        """
        Codifica variables categóricas usando Label Encoding.
        
        Args:
            data (pd.DataFrame): Dataset con variables categóricas
            
        Returns:
            pd.DataFrame: Dataset con variables codificadas
        """
        print("Codificando variables categóricas...")
        
        df = data.copy()
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col != 'default':  # No codificar el target
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                print(f"Codificada columna: {col}")
        
        return df
    
    def scale_numerical_features(self, data, fit=True):
        """
        Escala las características numéricas usando StandardScaler.
        
        Args:
            data (pd.DataFrame): Dataset con características numéricas
            fit (bool): Si es True, ajusta el scaler; si es False, solo transforma
            
        Returns:
            pd.DataFrame: Dataset con características escaladas
        """
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
    
    def download_german_credit_dataset(self):
        """
        Descarga el dataset 'uciml/german-credit' desde Kaggle usando kagglehub.
        Devuelve la ruta al archivo CSV descargado.
        """
        print("Descargando el dataset de Kaggle (uciml/german-credit)...")
        path = kagglehub.dataset_download("uciml/german-credit")
        print("Ruta de los archivos del dataset:", path)
        # Buscar el archivo CSV principal
        for file in os.listdir(path):
            if file.endswith('.csv'):
                return os.path.join(path, file)
        raise FileNotFoundError("No se encontró un archivo CSV en el dataset descargado.")
    
    def add_default_column(self, df):
        """
        Agrega la columna 'default' al DataFrame según reglas de negocio.
        Si cumple al menos 3 condiciones, se considera 'default'=1 (malo), si no, 'default'=0 (bueno).
        """
        conditions = [
            df['Age'] > 25,
            df['Job'] >= 2,
            df['Housing'] == 'own',
            df['Saving accounts'] == 'rich',
            df['Checking account'].isin(['moderate', 'rich']),
            (df['Duration'] > 6) & (df['Duration'] < 24)
        ]
        # Sumar cuántas condiciones cumple cada fila
        num_conditions = sum(conditions)
        df['default'] = (num_conditions >= 3).astype(int)
        return df
    
    def load_german_credit_data(self):
        """
        Descarga y carga el dataset de crédito alemán desde Kaggle o data/.
        Devuelve un DataFrame de pandas con columna 'default'.
        """
        # Intentar cargar desde data/ si existe
        local_path = 'data/german_credit_data.csv'
        if os.path.exists(local_path):
            data = pd.read_csv(local_path)
            print(f"Datos cargados desde: {local_path}")
        else:
            csv_path = self.download_german_credit_dataset()
            data = pd.read_csv(csv_path)
            print(f"Datos cargados desde Kaggle: {csv_path}")
        # Agregar columna default según reglas
        data = self.add_default_column(data)
        return data
    
    def prepare_data(self, data, test_size=0.4, random_state=42):
        """
        Prepara completamente los datos para entrenamiento.
        Args:
            data (pd.DataFrame): Dataset original
            test_size (float): Proporción del conjunto de prueba (por defecto 0.4)
            random_state (int): Semilla para reproducibilidad
        Returns:
            tuple: (X_train, X_test, y_train, y_test, feature_names)
        """
        print("Preparando datos para entrenamiento...")
        # Limpiar datos
        df = self.clean_data(data)
        # Separar features y target
        X = df.drop('default', axis=1)
        y = df['default']
        # Codificar variables categóricas
        X = self.encode_categorical_features(X)
        # Escalar características numéricas
        X = self.scale_numerical_features(X, fit=True)
        # Dividir en train y test (ahora 60/40)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Conjunto de entrenamiento: {X_train.shape}")
        print(f"Conjunto de prueba: {X_test.shape}")
        print(f"Características utilizadas: {list(X.columns)}")
        return X_train, X_test, y_train, y_test, list(X.columns)
    
    def get_feature_importance_analysis(self, data):
        """
        Analiza la importancia de las características para interpretabilidad.
        
        Args:
            data (pd.DataFrame): Dataset original
            
        Returns:
            dict: Análisis de características
        """
        print("Analizando importancia de características...")
        
        df = data.copy()
        
        # Correlación con el target
        correlations = df.corr()['default'].abs().sort_values(ascending=False)
        
        # Estadísticas por grupo (default vs no default)
        default_stats = df[df['default'] == 1].describe()
        non_default_stats = df[df['default'] == 0].describe()
        
        # Análisis de características numéricas
        numerical_features = df.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col != 'default']
        
        feature_analysis = {
            'correlations': correlations,
            'default_stats': default_stats,
            'non_default_stats': non_default_stats,
            'numerical_features': numerical_features
        }
        
        return feature_analysis

# Ejemplo de uso y prueba manual del procesador de datos

def main():
    """
    Función principal para demostrar el procesamiento de datos.
    """
    # Crear procesador
    processor = CreditDataProcessor()
    
    # Generar datos de ejemplo
    data = processor.generate_sample_data(n_samples=5000)
    
    # Guardar datos de ejemplo
    data.to_csv('data/credit_data_sample.csv', index=False)
    print("Datos de ejemplo guardados en 'data/credit_data_sample.csv'")
    
    # Preparar datos para entrenamiento
    X_train, X_test, y_train, y_test, feature_names = processor.prepare_data(data)
    
    # Guardar datos procesados
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    print("Datos procesados guardados en el directorio 'data/'")
    
    # Análisis de características
    analysis = processor.get_feature_importance_analysis(data)
    print("\nCorrelaciones con el target:")
    print(analysis['correlations'])

if __name__ == "__main__":
    main()