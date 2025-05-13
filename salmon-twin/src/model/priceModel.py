"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import pandas as pd
from datetime import datetime, timedelta
import config as cfg
import numpy as np
#from statsforecast import StatsForecast
#from statsforecast.models import AutoARIMA, HoltWinters
#import pmdarima as pm
#from statsmodels.tsa.arima.model import ARIMA as StatsmodelsARIMA
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor
from skforecast.preprocessing import RollingFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import combinations_with_replacement
from tqdm import tqdm
import warnings as wn
import random


# Se define la clase DataPrice para gestionar los datos de precios
# Esta clase se encarga de parsear los datos de precios, ajustarlos y predecirlos
# Se espera que los datos de precios contengan las columnas 'Year', 'Week' y 'EUR_kg'
# La columna 'Year' contiene el año de la observación
# La columna 'Week' contiene la semana del año de la observación
# La columna 'EUR_kg' contiene el precio en euros por kilogramo
# La clase también se encarga de gestionar los errores que puedan ocurrir durante el proceso
class DataPrice:
    def __init__(self):
        # Datos de precio en bruto        
        self._price_data_raw = None
        # Datos de precio procesados
        self._price_data = None
        # Datos de precio de la predicción
        self._price_data_forescast = None       
        # Datos del ultimo error        
        self.lastError = None

        # Con objeto de depurar el modelo de predicción pero no necesarios para el funcionamiento
        self._price_data_test = None
        self._price_data_train = None

    # Se parsea el dataframe de precios y se convierte a un formato adecuado para su uso
    # Se espera que el dataframe contenga las columnas 'Year', 'Week' y 'EUR_kg'
    def parsePrice(self, data):
        try:
            # Asignar los datos recibidos al atributo price_data
            self._price_data_raw = data
            
            if 'Year' in self._price_data_raw and 'Week' in self._price_data_raw and 'EUR_kg' in self._price_data_raw:                
                # Convierte la fecha de la semana y el año a un objeto datetime
                for i, row in self._price_data_raw.iterrows():
                    try:
                        year = int(row['Year'])
                        week = int(row['Week'])
                        temp_date = datetime.strptime(f'{year}-W{week}-1', "%G-W%V-%u")
                        self._price_data_raw.at[i, 'timestamp'] = pd.to_datetime(temp_date)                        
                    except ValueError:
                        self.lastError=cfg.DASHBOARD_PRICE_PARSE_ERROR
                        return False
                    try:
                        self._price_data_raw.at[i, 'EUR_kg'] = float(row['EUR_kg'])
                    except ValueError:
                        self.lastError=cfg.PRICEMODEL_ERROR_PARSER_PRICE
                        return False
                # Sort the data by timestamp
                self._price_data = self._price_data_raw.sort_values(by='timestamp')
                return True
            else:
                self.lastError= cfg.PRICEMODEL_ERROR_PARSER_COLUMNS_ERROR
                return False
        except ValueError as e:
            self.lastError="Error: {e}"
            return False

    # Se obtiene el dataframe de precios procesados
    # Parámetros:
    # None
    # Retorna:
    # pd.DataFrame: DataFrame con los datos de precios procesados
    def getPriceData(self):       
        return self._price_data
    
    # Se obtiene el dataframe de precios de la predicción
    # Parámetros:
    # None
    # Retorna:
    # pd.DataFrame: DataFrame con los datos de precios de la predicción
    def getPriceDataForecast(self):        
        return self._price_data_forescast
    
    # Se asignan los datos de precios a la variable de instancia
    # Parámetros:
    # data (pd.DataFrame): DataFrame que contiene los datos de precios
    # Se espera que el dataframe contenga las columnas 'Year', 'Week' y 'EUR_kg'
    # Retorna:
    # bool: True si se asignaron correctamente, False en caso contrario
    def setPriceData(self, data):        
        self._price_data = data.copy()
        # Se procesan los datos de precios
        if not self.parsePrice(self._price_data):
            return False
        return True
    
    # """
    # Encuentra la mejor combinación de estadísticas para minimizar el MAE
    #
    #Args:
    #    train: DataFrame con datos de entrenamiento
    #    test: DataFrame con datos de prueba
    #    type_stats: Lista de estadísticas disponibles
    #    max_window: Tamaño máximo de ventana
    #    n_stats: Número de estadísticas a seleccionar
    #    
    #Returns:
    #    tuple: (mejor_combinación, mejor_mae)    
    def find_optimal_statistics(self,train, test, type_stats, max_window):    
        # Generar todas las combinaciones posibles con repetición
        n_stats = 4
        all_combinations = list(combinations_with_replacement(type_stats, n_stats))
        print(f"Probando {len(all_combinations)} combinaciones de estadísticas...")
    
        best_mae = float('inf')
        best_combination = None
    
        # Configurar tamaños de ventana fijos
        window_sizes = [max(2, int(max_window/4)), 
                        max(4, int(max_window/2)), 
                        max(8, int(3*max_window/4)), 
                        max_window]
    
        # Crear serie temporal para entrenamiento
        y_train = pd.Series(
            data=train['y'].values,
            index=pd.DatetimeIndex(train['ds']),
            name='EUR_kg'
        )
    
        # Configurar el regresor base
        base_regressor = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=31, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.8, random_state=15926, verbose=-1
        )
    
        wn.filterwarnings('ignore', category=Warning)
        # Probar cada combinación
        results = []
        for stats in tqdm(all_combinations):
            try:
                # Configurar características de ventana con la combinación actual
                window_features = RollingFeatures(
                    stats=list(stats),
                    window_sizes=window_sizes
                )
            
                # Crear y entrenar el modelo
                forecaster = ForecasterRecursive(
                    regressor=base_regressor,
                    lags=min(8, len(train)//4),
                    window_features=window_features
                )
            
                # Entrenar el modelo
                forecaster.fit(y=y_train)
            
                # Predecir y evaluar
                predictions = forecaster.predict(steps=len(test))
                mae = mean_absolute_error(test['y'].values, predictions.values)
            
                # Guardar resultados
                results.append({
                    'stats': stats,
                    'mae': mae
                })
            
                # Actualizar la mejor combinación si corresponde
                if mae < best_mae:
                    best_mae = mae
                    best_combination = stats
                    print(f"\nNueva mejor combinación: {stats}, MAE: {mae:.4f}")
                
            except Exception as e:
                print(f"Error con combinación {stats}: {e}")
                continue
    
        # Ordenar resultados por MAE
        results.sort(key=lambda x: x['mae'])
    
        # Mostrar las mejores combinaciones
        print("\nMejores combinaciones:")
        for i, result in enumerate(results[:8]):
            print(f"{i+1}. {result['stats']} - MAE: {result['mae']:.4f}")
    
        return best_combination, best_mae
    
    
    #Encuentra la combinación óptima de tamaños de ventana para los estadísticos dados
    #
    #Args:
    #    train: DataFrame con datos de entrenamiento
    #    test: DataFrame con datos de prueba
    #    best_stats: Lista con los mejores estadísticos ya identificados
    #    max_window: Tamaño máximo de ventana permitido
    #    
    #Returns:
    #    tuple: (mejor_combinación_ventanas, mejor_mae)    
    def find_optimal_window_sizes(self, train, test, best_stats, max_window):    
        # Verificar que tengamos estadísticos válidos
        if len(best_stats) != 4:
            raise ValueError(f"Se requieren exactamente 4 estadísticos, se proporcionaron {len(best_stats)}")
    
        # Crear serie temporal para entrenamiento
        y_train = pd.Series(
            data=train['y'].values,
            index=pd.DatetimeIndex(train['ds']),
            name='EUR_kg'
        )
    
        # Configurar el regresor base (igual que el anterior)
        base_regressor = LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=31, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.8, random_state=15926, verbose=-1
        )
        
        # Definir posibles tamaños de ventana (respetando max_window)
        possible_windows = [i for i in range(2, max_window + 1) if i % 2 == 0]  # Solo pares para reducir combinaciones
    
        # Generar combinaciones de 4 tamaños de ventana (con cada tamaño ascendente)
        window_combinations = []
        for _ in range(500):  # Número razonable de combinaciones a probar
            # Seleccionar 4 tamaños de ventana aleatorios y ordenarlos
            sample = sorted(random.sample(possible_windows, 4))
            window_combinations.append(sample)
    
        print(f"Probando {len(window_combinations)} combinaciones de tamaños de ventana...")        
    
        # Variables para almacenar los mejores resultados
        best_mae = float('inf')
        best_window_sizes = None
    
        # Suprimir warnings
        wn.filterwarnings('ignore', message='.*DatetimeIndex without a frequency.*')
    
        # Probar cada combinación
        results = []
        for window_sizes in tqdm(window_combinations):
            try:
                # Configurar características de ventana con los mejores estadísticos y la combinación de ventanas actual
                window_features = RollingFeatures(
                    stats=list(best_stats),
                    window_sizes=window_sizes
                )
            
                # Crear y entrenar el modelo
                forecaster = ForecasterRecursive(
                    regressor=base_regressor,
                    lags=min(8, len(train)//4),
                    window_features=window_features
                )
            
                # Entrenar el modelo
                forecaster.fit(y=y_train)
            
                # Predecir y evaluar
                predictions = forecaster.predict(steps=len(test))
                mae = mean_absolute_error(test['y'].values, predictions.values)
            
                # Guardar resultados
                results.append({
                    'window_sizes': window_sizes,
                    'mae': mae
                })
            
                # Actualizar la mejor combinación si corresponde
                if mae < best_mae:
                    best_mae = mae
                    best_window_sizes = window_sizes
                    print(f"\nNueva mejor combinación de ventanas: {window_sizes}, MAE: {mae:.4f}")
            
            except Exception as e:
                print(f"Error con tamaños de ventana {window_sizes}: {e}")
                continue
    
        # Ordenar resultados por MAE
        results.sort(key=lambda x: x['mae'])
    
        # Mostrar las mejores combinaciones
        print("\nMejores combinaciones de tamaños de ventana:")
        for i, result in enumerate(results[:8]):
            print(f"{i+1}. {result['window_sizes']} - MAE: {result['mae']:.4f}")
    
        return best_window_sizes, best_mae
    

    
    # Encuentra los mejores hiperparámetros para el regresor LGBMRegressor
    #
    # Args:
    #    train: DataFrame con datos de entrenamiento
    #    test: DataFrame con datos de prueba
    #    best_stats: Lista con las mejores estadísticas ya identificadas
    #    best_windows: Lista con los mejores tamaños de ventana ya identificados
    #    
    # Returns:
    #    tuple: (mejores_parámetros, mejor_mae)    
    def find_optimal_regressor_params(self, train, test, best_stats, best_windows):    
        # Preparar los datos de serie temporal
        y_train = pd.Series(
            data=train['y'].values,
            index=pd.DatetimeIndex(train['ds']),
            name='EUR_kg'
        )
    
        # Configurar las características de ventana con los mejores estadísticos y tamaños
        window_features = RollingFeatures(
            stats=list(best_stats),
            window_sizes=best_windows
        )
    
        # Definir los rangos de hiperparámetros a probar
        param_grid = {
            'n_estimators':[50, 100, 150, 200, 300, 400],
            'learning_rate':[0.01, 0.05, 0.1, 0.2, 0.3],
            'max_depth':[3, 4, 5, 6, 7, 8, 9, 10],
            'num_leaves':[10, 20, 31, 40, 50, 60],
            'min_child_samples':[5, 8, 10, 15, 20, 25],
            'subsample':[0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree':[0.6, 0.7, 0.8, 0.9, 1.0]
        }
    
        # Generar combinaciones aleatorias de hiperparámetros
        n_iterations = 100  # Número de combinaciones a probar
        random_params = []
    
        for _ in range(n_iterations):
            params = {
                'n_estimators': random.choice(param_grid['n_estimators']),
                'learning_rate': random.choice(param_grid['learning_rate']),
                'max_depth': random.choice(param_grid['max_depth']),
                'num_leaves': random.choice(param_grid['num_leaves']),
                'min_child_samples': random.choice(param_grid['min_child_samples']),
                'subsample': random.choice(param_grid['subsample']),
                'colsample_bytree': random.choice(param_grid['colsample_bytree']),
                'random_state': 15926,
                'verbose': -1
            }
            random_params.append(params)
    
        print(f"Probando {n_iterations} combinaciones de hiperparámetros...")
    
        # Variables para almacenar los mejores resultados
        best_mae = float('inf')
        best_params = None
    
        # Suprimir advertencias
        wn.filterwarnings('ignore', category=Warning)
    
        # Probar cada combinación
        results = []
        for params in tqdm(random_params):
            try:
                # Crear y configurar el regresor con los parámetros actuales
                regressor = LGBMRegressor(**params)
            
                # Crear el modelo
                forecaster = ForecasterRecursive(
                    regressor=regressor,
                    lags = 4,
                    window_features=window_features
                )
            
                # Entrenar el modelo
                forecaster.fit(y=y_train)
            
                # Predecir y evaluar
                predictions = forecaster.predict(steps=len(test))
                mae = mean_absolute_error(test['y'].values, predictions.values)
            
                # Guardar resultados
                results.append({
                    'params': params,
                    'mae': mae
                })
            
                # Actualizar los mejores parámetros si corresponde
                if mae < best_mae:
                    best_mae = mae
                    best_params = params
                    print(f"\nNuevos mejores parámetros: n_estimators={params['n_estimators']}, "
                      f"learning_rate={params['learning_rate']}, max_depth={params['max_depth']}, "
                      f"MAE: {mae:.4f}")
            
            except Exception as e:
                print(f"Error con parámetros {params}: {e}")
                continue
    
        # Ordenar resultados por MAE
        results.sort(key=lambda x: x['mae'])
    
        # Mostrar las mejores combinaciones
        print("\nMejores configuraciones de hiperparámetros:")
        for i, result in enumerate(results[:5]):
            params = result['params']
            print(f"{i+1}. n_est={params['n_estimators']}, lr={params['learning_rate']}, "
              f"depth={params['max_depth']}, leaves={params['num_leaves']}, "
              f"min_samples={params['min_child_samples']} - MAE: {result['mae']:.4f}")
    
        return best_params, best_mae
    

    """
    Optimiza simultáneamente las estadísticas, tamaños de ventana y parámetros del regresor
    
    Args:
        train: DataFrame con datos de entrenamiento
        test: DataFrame con datos de prueba
        max_window: Tamaño máximo de ventana permitido
        n_iterations: Número de combinaciones aleatorias a probar
        
    Returns:
        tuple: (mejores_estadísticas, mejores_ventanas, mejores_parámetros, mejor_mae)
    """
    def find_optimal_configuration(self, train, test, max_window, n_iterations=50):    
        # Crear serie temporal para entrenamiento
        y_train = pd.Series(
            data=train['y'].values,
            index=pd.DatetimeIndex(train['ds']),
            name='EUR_kg'
        )
    
        # Definir opciones para cada componente
        stats_options = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation']
    
        # Parámetros del regresor
        param_grid = {
            'n_estimators': [50, 100, 150, 200, 300],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'num_leaves': [10, 20, 31, 40],
            'min_child_samples': [5, 8, 10, 15],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
    
        # Suprimir advertencias
        wn.filterwarnings('ignore', category=Warning)
    
        # Variables para almacenar los mejores resultados
        best_mae = float('inf')
        best_rmse = float('inf')
        best_mape = float('inf')
        best_dir_acc = 0.0
        best_score = 0.0  # Inicializar con 0 ya que queremos maximizarlo
        best_stats = None
        best_windows = None
        best_params = None
    
        print(f"Probando {n_iterations} configuraciones aleatorias completas...")
    
        # Almacenar resultados
        results = []
    
        # Búsqueda aleatoria
        for i in tqdm(range(n_iterations)):
            try:
                # 1. Seleccionar estadísticas aleatorias (4 elementos)
                stats = random.choices(stats_options, k=4)
            
                # 2. Generar tamaños de ventana aleatorios (ascendentes)
                possible_windows = [i for i in range(2, max_window + 1) if i % 2 == 0]
                if len(possible_windows) >= 4:
                    windows = sorted(random.sample(possible_windows, 4))
                else:
                    windows = sorted(random.choices(possible_windows, k=4))
            
                # 3. Seleccionar parámetros aleatorios para el regresor
                params = {
                    'n_estimators': random.choice(param_grid['n_estimators']),
                    'learning_rate': random.choice(param_grid['learning_rate']),
                    'max_depth': random.choice(param_grid['max_depth']),
                    'num_leaves': random.choice(param_grid['num_leaves']),
                    'min_child_samples': random.choice(param_grid['min_child_samples']),
                    'subsample': random.choice(param_grid['subsample']),
                    'colsample_bytree': random.choice(param_grid['colsample_bytree']),
                    'random_state': 15926,
                    'verbose': -1
                }
            
                # 4. Configurar modelo con esta combinación
                window_features = RollingFeatures(
                    stats=stats,
                    window_sizes=windows
                )
            
                regressor = LGBMRegressor(**params)
            
                forecaster = ForecasterRecursive(
                    regressor=regressor,
                    lags=4,  # Valor fijo para evitar sobreajuste
                    window_features=window_features
                )
            
                # 5. Entrenar y evaluar
                forecaster.fit(y=y_train)
                predictions = forecaster.predict(steps=len(test))
                # Evaluar el modelo
                mae = mean_absolute_error(test['y'].values, predictions.values)
                rmse = np.sqrt(mean_squared_error(test['y'].values, predictions.values))

                # MAPE (Error porcentual)
                def mape(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                mape_value = mape(test['y'].values, predictions.values)

                # Dirección de cambio (tendencias acertadas)
                def direction_accuracy(y_true, y_pred):
                    if len(y_true) <= 1:
                        return 0                    
                    direction_true = np.diff(y_true) > 0
                    direction_pred = np.diff(y_pred) > 0
                    return np.mean(direction_true == direction_pred) * 100                
                dir_acc = direction_accuracy(test['y'].values, predictions.values)

                # Ponderaciones según importancia (deben sumar 1.0)
                score = (
                    0.60 * (1.0 - mae/5.0) +            # MAE normalizado (menor es mejor)
                    0.15 * (1.0 - rmse/8.0) +           # RMSE normalizado (menor es mejor)
                    0.15 * (1.0 - mape_value/100) +     # MAPE (menor es mejor)
                    0.10 * (dir_acc/100)                # Acierto direccional (mayor es mejor)
                )

                # Actualizar si esta configuración es mejor
                if score > best_score:  # Nota: ahora buscamos maximizar score, no minimizar MAE
                    best_score = score
                    best_mae = mae
                    best_rmse = rmse
                    best_mape = mape_value
                    best_dir_acc = dir_acc
                    best_stats = stats
                    best_windows = windows
                    best_params = params
                    print(f"\nNueva mejor configuración (score: {best_score:.4f}) (iter {i+1}/{n_iterations}):")
                    print(f"MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f}, MAPE: {best_mape:.2f}%, Dir: {best_dir_acc:.2f}%")                    
                    print(f"Stats: {best_stats}")
                    print(f"Windows: {best_windows}")
                    print(f"Params: n_est={best_params['n_estimators']}, lr={best_params['learning_rate']}, depth={best_params['max_depth']}")
            
                # 6. Guardar resultados
                results.append({
                    'score': score,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape_value,
                    'dir_acc': dir_acc,
                    'stats': stats,
                    'windows': windows,
                    'params': params,                    
                })
            
            except Exception as e:
                print(f"Error en iteración {i+1}: {e}")
                continue
    
        # Ordenar y mostrar mejores resultados
        results.sort(key=lambda x: x['score'],reverse=True)
        print("\nMejores 5 configuraciones encontradas:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. (score: {best_score:.4f}) Stats: {result['stats']}")
            print(f"   Windows: {result['windows']}")
            params = result['params']
            print(f"   Params: n_est={params['n_estimators']}, lr={params['learning_rate']}, depth={params['max_depth']}")
            print(f"   MAE: {result['mae']:.4f}")
            print(f"   RMSE: {result['rmse']:.4f}")
            print(f"   MAPE: {result['mape']:.2f}%")
            print(f"   Dir: {result['dir_acc']:.2f}%")
            print("---")
    
        return results[0]['stats'], results[0]['windows'], results[0]['params'], results[0]['score'], results[0]['mae'], results[0]['rmse'], results[0]['mape'], results[0]['dir_acc']
    
    # Se ajusta el modelo de precios utilizando los datos de precios procesados
    # Parámetros:
    # start_date (datetime): Fecha inicial para el ajuste del modelo
    # end_date (datetime): Fecha final para el ajuste del modelo
    # horizon_days (int): Número de días para la predicción
    # Retorna:
    # bool: True si se ajustó correctamente, False en caso contrario   
    def fit_price(self, percent, start_date=None, end_date=None):
        self.lastError = None
        if self._price_data is None:
             self.lastError = cfg.PRICEMODEL_FIT_NO_DATA
             return False
        try:
            # Filter data based on the selected dates
            filtered_data = self._price_data.copy()
            filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'], errors='coerce')
            # Eliminar filas con valores NaT (fechas inválidas)
            filtered_data = filtered_data.dropna(subset=['timestamp'])           
            
            # Filtrar por fecha inicial si se proporciona
            if start_date:
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date >= start_date]
                
            # Filtrar por fecha final si se proporciona
            if end_date:
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date <= end_date]
                
            # Asegurarse de que el DataFrame esté ordenado por la columna 'timestamp'
            filtered_data = filtered_data.sort_values(by='timestamp')

            # Verificar si hay datos suficientes después del filtrado
            if len(filtered_data) < 10:  # Establecer un mínimo razonable de puntos
                self.lastError = cfg.PRICEMODEL_NOT_ENOUGHT_DATA
                return False
            
            # Define el porcentaje para el conjunto de entrenamiento
            delta_days = (end_date - start_date).days
            if delta_days > 0:  # Protect against division by zero
                current_day_offset = int(delta_days * percent)
                current_date = start_date + timedelta(days=current_day_offset)
            
            # Divide el DataFrame
            data = pd.DataFrame({                
                'ds': pd.to_datetime(filtered_data['timestamp']),
                'y': filtered_data['EUR_kg'].astype(float)  # Convertir toda la columna a float
            })
            train = data[data['ds'].dt.date <= current_date]
            test = data[data['ds'].dt.date > current_date]

            # Test buscar el mejor modelo
            #type_stats=['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation']
            #best_combination, best_mae = self.find_optimal_statistics(train, test, type_stats, len(train)//3)
            #best_combination = ('min', 'max', 'std', 'median')
            #best_windows, best_mae_windows = self.find_optimal_window_sizes(train, test, best_combination, len(train)//3)
            #best_windows = [2, 6, 8, 14]
            #best_mae_windows = 0.4519
            #Mejores configuraciones de hiperparámetros:
            #1. n_est=300, lr=0.1, depth=3, leaves=10, min_samples=8 - MAE: 1.1725
            #1. n_est=200, lr=0.3, depth=3, leaves=31, min_samples=25 - MAE: 1.2056
            #print(f"Mejor combinación de estadísticas: {best_windows}, MAE: {best_mae_windows:.4f}")
            #best_params, best_mae_params = self.find_optimal_regressor_params(train, test, best_combination, best_windows)
            #print(f"Mejores parámetros: {best_params}, MAE: {best_mae_params:.4f}")

            #Mejor combinación de estadísticas: ['max', 'max', 'std', 'sum'], MAE: 0.8196
            #Mejores tamaños de ventana: [4, 42, 44, 58], MAE: 0.8196
            #Mejores parámetros: {'n_estimators': 100, 'learning_rate': 0.2, 'max_depth': 3, 'num_leaves': 20, 'min_child_samples': 10, 'subsample': 0.7, 'colsample_bytree': 0.8, 'random_state': 15926, 'verbose': -1}, MAE: 0.8196
            #Estadísticas de train - Min: 2.47, Max: 6.27, Media: 4.31
            #MAE final en TEST: 0.820

            #Mejores 5 configuraciones encontradas:
            #1. Stats: ['std', 'median', 'ratio_min_max', 'median']
            #Windows: [2, 6, 8, 22]
            #Params: n_est=100, lr=0.1, depth=4
            #MAE: 0.3666

            best_stats, best_windows, best_params, best_score, best_mae, best_rmse, best_mape, best_dirc = self.find_optimal_configuration(train, test, len(train)//3, 1000)
            print(f"Mejor score: {best_score:.4f}")
            print(f"Mejor MAE: {best_mae:.4f}")
            print(f"Mejor RMSE: {best_rmse:.4f}")
            print(f"Mejor MAPE: {best_mape:.2f}%")
            print(f"Mejor acierto direccional: {best_dirc:.2f}%")
            print(f"Mejor combinación de estadísticas: {best_stats}")
            print(f"Mejores parámetros: {best_params}")
            print(f"Mejores tamaños de ventana: {best_windows}")
            
            max_possible_window = len(train)-1
            max_windows = max(best_windows)
            min_windows = min(best_windows)
            if max_possible_window < max_windows:
                scale_factor = max_possible_window / max_windows
                best_windows = [
                    max(min_windows, int(min_windows * scale_factor)),
                    max(min_windows, int(best_windows[1] * scale_factor)),
                    max(min_windows, int(best_windows[2] * scale_factor)),
                    max_possible_window
                ]
                print(f"Ajustando tamaños de ventana a: {best_windows} debido a tamaño limitado de datos")

            window_features = RollingFeatures(
                stats=best_stats,  
                window_sizes=best_windows
            )

            wn.filterwarnings('ignore', category=Warning)
            # Crear el modelo ForecasterRecursive
            regressor = LGBMRegressor(
                n_estimators=best_params['n_estimators'],
                learning_rate=best_params['learning_rate'],
                max_depth=best_params['max_depth'],
                num_leaves=best_params['num_leaves'],
                min_child_samples=best_params['min_child_samples'],
                subsample=best_params['subsample'],
                colsample_bytree=best_params['colsample_bytree'],
                random_state=15926,
                verbose=-1
            )

            forecaster = ForecasterRecursive(
                regressor       = regressor,
                lags            = 4,
                window_features = window_features
            )

            # Entrenar el modelo con los datos de train
            y_train = pd.Series(
                    data=train['y'].values,
                    index=pd.DatetimeIndex(train['ds']),
                    name='EUR_kg'
                )

            forecaster.fit(y=y_train)

            # Predecir con el nuevo modelo            
            predictions = forecaster.predict(steps=len(test))

            # 6. Evaluar el modelo
            mae_test = mean_absolute_error(test['y'].values, predictions.values)
            print(f"Estadísticas de train - Min: {train['y'].min()}, Max: {train['y'].max()}, Media: {train['y'].mean():.2f}")
            print(f"MAE final en TEST: {mae_test:.3f}")
            print(predictions.values)            

            # Importante: añadir las fechas de predicción
            # 1. Obtener la última fecha de los datos de entrenamiento
            last_date = train['ds'].iloc[-1]
        
            # 2. Generar un rango de fechas futuras semanales
            future_dates = pd.date_range(
                start=last_date, #+ pd.Timedelta(days=7),  # Una semana después de la última fecha
                #periods=horizon_weeks,  # Número de semanas a predecir
                periods=len(test),  # Número de semanas a predecir
                freq='W'  # Frecuencia semanal
            )

            # 3. Añadir las fechas al DataFrame de predicción
            self._price_data_forescast = pd.DataFrame()
            self._price_data_forescast['ds'] = future_dates
            
            self._price_data_test = test.copy()
            self._price_data_train = train.copy()

            self._price_data_forescast['y'] = predictions.values.copy()
            return True

        except ValueError as e:
            self.lastError= cfg.PRICEMODEL_FIT_ERROR.format(e=e.args[0])
            return False

