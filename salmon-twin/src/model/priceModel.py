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

    """
        Prepara los datos de entrenamiento y prueba para la optimización de parámetros
        
        Args:
            percent: Porcentaje para dividir train/test (0.0 - 1.0)
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            tuple: (train_data, test_data) o None si hay error
        """
    def prepare_data_for_optimization(self, percent, start_date, end_date):
    
        try:
            if self._price_data is None:
                self.lastError = cfg.PRICEMODEL_FIT_NO_DATA
                return None
        
            # Información inicial para diagnóstico
            total_records = len(self._price_data)
        
            # Filtrar datos basado en las fechas seleccionadas
            filtered_data = self._price_data.copy()
            filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'], errors='coerce')
            filtered_data = filtered_data.dropna(subset=['timestamp'])
        
            records_after_date_parsing = len(filtered_data)
            invalid_dates = total_records - records_after_date_parsing
        
            if start_date:
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date >= start_date]
            if end_date:
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date <= end_date]
            
            filtered_data = filtered_data.sort_values(by='timestamp')
        
            records_after_date_filter = len(filtered_data)
            records_filtered_out = records_after_date_parsing - records_after_date_filter
        
            # Diagnóstico detallado de datos insuficientes
            if len(filtered_data) < 10:
                error_details = []
                error_details.append(f"Datos insuficientes para optimización (mínimo: 10, encontrados: {len(filtered_data)})")
                error_details.append(f"Registros originales: {total_records}")
            
                if invalid_dates > 0:
                    error_details.append(f"Registros con fechas inválidas eliminados: {invalid_dates}")
            
                if records_filtered_out > 0:
                    error_details.append(f"Registros fuera del rango de fechas ({start_date} - {end_date}): {records_filtered_out}")
            
                if len(filtered_data) > 0:
                    first_date = filtered_data['timestamp'].min().strftime('%Y-%m-%d')
                    last_date = filtered_data['timestamp'].max().strftime('%Y-%m-%d')
                    error_details.append(f"Rango de datos disponibles: {first_date} a {last_date}")
                else:
                    error_details.append("No hay datos válidos en el rango especificado")
            
                # Sugerencias para el usuario
                error_details.append("\nSugerencias:")
                if records_filtered_out > records_after_date_filter:
                    error_details.append("- Amplíe el rango de fechas de la balsa")
                if invalid_dates > 0:
                    error_details.append("- Revise el formato de fechas en los datos de precios")
                if total_records < 10:
                    error_details.append("- Cargue más datos históricos de precios")
            
                self.lastError = "\n".join(error_details)
                return None
        
            # Calcular fecha de corte
            delta_days = (end_date - start_date).days
            if delta_days > 0:
                current_day_offset = int(delta_days * percent)
                current_date = start_date + timedelta(days=current_day_offset)
            else:
                current_date = start_date
        
            # Preparar datos en formato requerido
            data = pd.DataFrame({
                'ds': pd.to_datetime(filtered_data['timestamp']),
                'y': filtered_data['EUR_kg'].astype(float)
            })
        
            train = data[data['ds'].dt.date <= current_date]
            test = data[data['ds'].dt.date > current_date]
        
            # Verificación de datos de entrenamiento insuficientes con diagnóstico detallado
            if len(train) < 5:
                error_details = []
                error_details.append(f"Datos de entrenamiento insuficientes (mínimo: 5, encontrados: {len(train)})")
                error_details.append(f"Datos totales disponibles: {len(data)}")
                error_details.append(f"Datos de prueba: {len(test)}")
                error_details.append(f"Fecha de corte actual: {current_date.strftime('%Y-%m-%d')}")
                error_details.append(f"Porcentaje de división: {percent*100:.1f}%")
            
                if len(data) > 0:
                    first_date = data['ds'].min().strftime('%Y-%m-%d')
                    last_date = data['ds'].max().strftime('%Y-%m-%d')
                    error_details.append(f"Rango de datos: {first_date} a {last_date}")
            
                error_details.append("\nSugerencias:")
                if len(data) >= 5:
                    error_details.append("- Ajuste el slider de fecha actual para incluir más datos en entrenamiento")
                    new_percent = max(0.1, 5.0 / len(data))
                    error_details.append(f"- Use al menos {new_percent*100:.1f}% del rango de fechas para entrenamiento")
                else:
                    error_details.append("- Cargue más datos históricos de precios")
                    error_details.append("- Amplíe el rango de fechas de la balsa")
            
                self.lastError = "\n".join(error_details)
                return None
        
            # Guardar para uso posterior
            self._last_train_data = train.copy()
            self._last_test_data = test.copy()
        
            return train, test
        
        except Exception as e:
            self.lastError = f"Error preparando datos: {str(e)}"
            return None

    """        
    Ejecuta la optimización de parámetros del modelo de precios
        Args:
            train_data: DataFrame con datos de entrenamiento
            test_data: DataFrame con datos de prueba
            n_iterations: Número de iteraciones para la optimización
            fixed_stats: Estadísticas fijas a usar (opcional)
            fixed_windows: Ventanas fijas a usar (opcional)
            fixed_params: Parámetros fijos del modelo (opcional)
            lags: Número de lags a usar (opcional)
            progress_callback: Función de callback para actualizar el progreso (opcional)
        Returns:
            dict: Resultados de la optimización o None si hay error
    """    
    def run_parameter_optimization(self, train_data, test_data, n_iterations=100,
                               fixed_stats=None, fixed_windows=None, fixed_params=None, lags=None,
                               progress_callback=None):
       
        try:
            results = self.find_optimal_configuration(
                train=train_data,
                test=test_data,
                max_window=len(train_data)//2,
                n_iterations=n_iterations,
                fixed_stats=fixed_stats,
                fixed_windows=fixed_windows,
                fixed_params=fixed_params,
                lags=lags,
                progress_callback=progress_callback
            )
            # Guardar resultados para uso posterior
            self._optimal_results = results
            return results
            
        except Exception as e:
            self.lastError = f"Error en optimización: {str(e)}"
            return None
        
    """
        Entrena el modelo final con los mejores parámetros encontrados
        
        Args:
            best_results: Resultados de la optimización (opcional, usa los últimos si no se especifica)
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
    def train_final_model(self, best_results=None):
        
        try:
            if self._last_train_data is None or self._last_test_data is None:
                self.lastError = "No hay datos preparados para entrenar"
                return False
            
            # Usar los mejores resultados disponibles
            if best_results is None:
                if self._optimal_results is None or len(self._optimal_results) == 0:
                    self.lastError = "No hay resultados de optimización disponibles"
                    return False
                best_results = self._optimal_results[0]
            
            train = self._last_train_data
            test = self._last_test_data
            
            # Extraer mejores parámetros
            best_stats = best_results['stats']
            best_windows = best_results['windows']
            best_params = best_results['params']
            
            # Ajustar ventanas si es necesario
            max_possible_window = len(train) - 1
            if max_possible_window < max(best_windows):
                scale_factor = max_possible_window / max(best_windows)
                best_windows = [max(1, int(w * scale_factor)) for w in best_windows]
                if len(set(best_windows)) < len(best_windows):
                    best_windows = [1, 2, 3, 4]
            
            # Configurar modelo
            window_features = RollingFeatures(
                stats=best_stats,
                window_sizes=best_windows
            )
            
            regressor = LGBMRegressor(**best_params)
            
            forecaster = ForecasterRecursive(
                regressor=regressor,
                lags=min(53, len(train)-1),
                window_features=window_features
            )
            
            # Entrenar
            y_train = pd.Series(
                data=train['y'].values,
                index=pd.DatetimeIndex(train['ds']),
                name='EUR_kg'
            )
            
            forecaster.fit(y=y_train)
            predictions = forecaster.predict(steps=len(test))
            
            # Generar fechas futuras
            last_date = train['ds'].iloc[-1]
            future_dates = pd.date_range(
                start=last_date,
                periods=len(test),
                freq='W'
            )
            
            # Guardar resultados
            self._price_data_forescast = pd.DataFrame({
                'ds': future_dates[:len(predictions)],
                'y': predictions.values
            })
            
            return True
            
        except Exception as e:
            self.lastError = f"Error entrenando modelo final: {str(e)}"
            return False

    def get_train_test_data(self):
        """Retorna los últimos datos de entrenamiento y prueba preparados"""
        return self._last_train_data, self._last_test_data
    
    def get_optimization_results(self):
        """Retorna los últimos resultados de optimización"""
        return self._optimal_results

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
    def find_optimal_configuration(self, train, test, max_window, n_iterations=50,
                                   fixed_stats=None, fixed_windows=None,fixed_params=None,lags=None,
                                   progress_callback=None):    
        # Crear serie temporal para entrenamiento
        y_train = pd.Series(
            data=train['y'].values,
            index=pd.DatetimeIndex(train['ds']),
            name='EUR_kg'
        )
    
        # Definir opciones para cada componente
        if fixed_stats is None:
            stats_options = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation']
        else:
            stats_options = fixed_stats

        if fixed_windows is None:
            possible_windows = [i for i in range(1, max_window-1)]
        else:
            possible_windows = fixed_windows
    
        # Parámetros del regresor
        if fixed_params is None:
            param_grid = {
                'n_estimators': [50, 100, 150, 200, 300],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'num_leaves': [10, 20, 31, 40],
                'min_child_samples': [5, 8, 10, 15],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        else:
            param_grid = fixed_params

        if not lags is None:
            # Asegurarse de que lags no exceda el tamaño de train
            lags=min(lags,len(train)-1)
    
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
        best = False
        stop = False
        for i in tqdm(range(n_iterations)):
            try:
                # 1. Seleccionar estadísticas aleatorias (4 elementos)
                stats = random.choices(stats_options, k=4)
            
                # 2. Generar tamaños de ventana aleatorios (ascendentes)                
                if len(possible_windows) >= 4:
                    windows = sorted(random.sample(possible_windows, 4))
                else:
                    windows = sorted(random.choices(possible_windows, k=4))

                max_possible_window = len(train)-1
                max_windows = max(windows)
                min_windows = min(windows)
                if max_possible_window < max_windows:
                    scale_factor = max_possible_window / max_windows
                    windows = [
                        max(min_windows, int(min_windows * scale_factor)),
                        max(min_windows, int(windows[1] * scale_factor)),
                        max(min_windows, int(windows[2] * scale_factor)),
                        max_possible_window
                    ]
                    # Si hay repetidos dejar como 1,2,3,4
                    if len(set(windows)) < len(windows):  # Detecta valores duplicados
                        windows = [1, 2, 3, 4] 
                    print(f"\nAjustando tamaños de ventana a {windows} debido a tamaño limitado de datos")
            
                # 3. Seleccionar parámetros aleatorios para el regresor
                if fixed_params is None:
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
                else:
                    params = fixed_params
            
                # 4. Configurar modelo con esta combinación
                window_features = RollingFeatures(
                    stats=stats,
                    window_sizes=windows
                )
            
                regressor = LGBMRegressor(**params)
            
                forecaster = ForecasterRecursive(
                    regressor=regressor,                    
                    lags=lags,
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
                    print(f"\nNueva mejor configuración (score: {best_score:.6f}) (iter {i+1}/{n_iterations}):")
                    print(f"MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f}, MAPE: {best_mape:.2f}%, Dir: {best_dir_acc:.2f}%")                    
                    print(f"Stats: {best_stats}")
                    print(f"Windows: {best_windows}")
                    if fixed_params is None:
                        print(f"Params: n_est={best_params['n_estimators']}, lr={best_params['learning_rate']}, depth={best_params['max_depth']}")
                    else:
                        print(f"Params: {fixed_params}")
            
                    # 6. Guardar resultados
                    results.append({
                        'score':    score,
                        'mae':      mae,
                        'rmse':     rmse,
                        'mape':     mape_value,
                        'dir_acc':  dir_acc,
                        'stats':    stats,
                        'windows':  windows,
                        'params':   params,                    
                    })
                    best_result = results[-1].copy()
                    best = True

                if not progress_callback is None:                    
                    if best:
                        best = False
                        progress_callback(i, best_result)
                    else:
                        progress_callback(i, None)         
            
            except Exception as e:
                print(f"Error en iteración {i+1}: {e}")
                continue
    
        # Ordenar y mostrar mejores resultados
        results.sort(key=lambda x: x['score'],reverse=True)
        print("\nMejores 5 configuraciones encontradas:")
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. (score: {result['score']:.4f}) Stats: {result['stats']}")
            print(f"   Windows: {result['windows']}")

            if fixed_params is None:
                params = result['params']
                print(f"   Params: n_est={params['n_estimators']}, lr={params['learning_rate']}, depth={params['max_depth']}")
            else:
                print(f"   Params: {fixed_params}")

            print(f"   MAE: {result['mae']:.4f}")
            print(f"   RMSE: {result['rmse']:.4f}")
            print(f"   MAPE: {result['mape']:.2f}%")
            print(f"   Dir: {result['dir_acc']:.2f}%")
            print("---")
    
        return results
    
    # Se ajusta el modelo de precios utilizando los datos de precios procesados
    # Parámetros:
    # start_date (datetime): Fecha inicial para el ajuste del modelo
    # end_date (datetime): Fecha final para el ajuste del modelo
    # horizon_days (int): Número de días para la predicción
    # Retorna:
    # bool: True si se ajustó correctamente, False en caso contrario   
    def fit_price(self, percent, start_date=None, end_date=None, adjust=False):
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

            # Inicializar variables de control
            results = None
            fixed_windows = None
            fixed_stats = None
            fixed_lags = None
            fixed_params = None

            # Variables fijadas para la búsqueda
            if not adjust:
                # Si no se ajusta, se fijan las ventanas y estadísticas por defecto
                fixed_windows = [4, 12, 26, 53]
                fixed_stats = ["mean", "mean","mean","mean"]
                fixed_lags = 53
                fixed_params = {'random_state': 15926,'verbose': -1}            
            else:
                results = self.find_optimal_configuration(train, test, len(train)//1, 1000,fixed_stats, fixed_windows, fixed_params,fixed_lags)

            if results is None or len(results) == 0:
               best_score = 0.0
               best_mae = 0.0
               best_rmse = 0.0
               best_mape = 0.0
               best_dirc = 0.0
            else:
                best_score = results[0]['score']
                best_mae = results[0]['mae']
                best_rmse = results[0]['rmse']
                best_mape = results[0]['mape']
                best_dirc = results[0]['dir_acc']

            if fixed_stats is None and len(results) > 0:
                best_stats = results[0]['stats']
            elif fixed_stats is None and len(results) == 0:
                best_stats = ["mean", "mean","mean","mean"]
            else:
                best_stats = fixed_stats

            if fixed_windows is None and len(results) > 0:
                best_windows = results[0]['windows']
            elif fixed_windows is None and len(results) == 0:
                best_windows = [4, 12, 26, 53]
            else:
                best_windows = fixed_windows

            if fixed_params is None and len(results) > 0:
                best_params = results[0]['params']
            elif fixed_params is None and len(results) == 0:
                best_params = {'random_state': 15926,'verbose': -1}
            else:
                best_params = fixed_params

            print(f"Mejor score: {best_score:.4f}")
            print(f"Mejor MAE: {best_mae:.4f}")
            print(f"Mejor RMSE: {best_rmse:.4f}")
            print(f"Mejor MAPE: {best_mape:.2f}%")
            print(f"Mejor acierto direccional: {best_dirc:.2f}%")
            print(f"Mejor combinación de estadísticas: {best_stats}")
            print(f"Mejores parámetros: {best_params}")
            print(f"Mejores tamaños de ventana: {best_windows}")
            
            max_possible_window = len(train)-1
            if max_possible_window < 5:
                # Si el tamaño máximo de ventana es menor a 5, ajustar las ventanas
                best_windows = [1, 2, 3]
                best_stats = ["mean", "mean","mean"]
                print(f"Ajustando tamaños de ventana a: {best_windows} debido a tamaño limitado de datos")
            else:
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
                    # Si hay repetidos dejar como 1,2,3,4
                    if len(set(best_windows)) < len(best_windows):  # Detecta valores duplicados
                        best_windows = [1, 2, 3, 4]
                        print(f"Ajustando tamaños de ventana a: {best_windows} debido a tamaño limitado de datos")

            window_features = RollingFeatures(
                stats=best_stats,  
                window_sizes=best_windows
            )

            wn.filterwarnings('ignore', category=Warning)
            # Crear el modelo ForecasterRecursive
            if fixed_params is None and len(results) > 0:
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
            else:
               params = best_params
               regressor = LGBMRegressor(**params)   
            
            forecaster = ForecasterRecursive(
                regressor       = regressor,
                lags            = min(fixed_lags,len(train)-1),  # Asegurarse de que lags no exceda el tamaño de train
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

