"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import pandas as pd
from datetime import datetime, timedelta
import config as cfg
import numpy as np
from skforecast.recursive import ForecasterRecursive
from lightgbm import LGBMRegressor
from skforecast.preprocessing import RollingFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import warnings as wn
import random
import os
import json

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
    def prepare_data_for_optimization(self, percent, start_date, end_date, prev_start_date):
    
        try:
            if self._price_data is None:
                self.lastError = cfg.PRICEMODEL_FIT_NO_DATA
                return None
        
            # Información inicial para diagnóstico
            total_records = len(self._price_data)
            print(f"Total registros originales: {total_records}")
            print(f"fechas: {start_date} - {end_date} Fecha anterior:{prev_start_date}")
        
            # Filtrar datos basado en las fechas seleccionadas
            filtered_data = self._price_data.copy()
            filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'], errors='coerce')
            filtered_data = filtered_data.dropna(subset=['timestamp'])
            # Contar registros antes de la conversión de fecha
            records_after_date_parsing = len(filtered_data)
            invalid_dates = total_records - records_after_date_parsing
            # Filtrar por fecha previa si se proporciona
            if prev_start_date:                
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date >= prev_start_date]
            elif start_date:
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date >= start_date]
            # Filtrar por fecha final si se proporciona
            if end_date:
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date <= end_date]
            # Asegurarse de que el DataFrame esté ordenado por la columna 'timestamp'
            filtered_data = filtered_data.sort_values(by='timestamp')
            # Contar registros después del filtrado por fecha
            records_after_date_filter = len(filtered_data)
            records_filtered_out = records_after_date_parsing - records_after_date_filter
            print(f"Registros después del filtrado por fecha: {records_after_date_filter}")
            print(f"Registros filtrados por fecha inválida: {invalid_dates}")
        
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
            print(f"Datos de entrenamiento: {len(train)}, Datos de prueba: {len(test)}")

        
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
    def run_parameter_optimization(self, train_data, test_data, n_iterations,
                               fixed_stats=None, fixed_windows=None, fixed_params=None, lags=None,
                               progress_callback=None):
       
        try:
            results = self.find_optimal_configuration(
                train=train_data,
                test=test_data,
                max_window=len(train_data) - 1,
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
                freq='MS'
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

    # Se aplica un suavizado mensual a los datos de precios    
    def smoothPriceMonthly(self):        
        if self._price_data is None:
            self.lastError = cfg.PRICEMODEL_ERROR_NO_PRICE_DATA
            return False
        
        try:
            # Convertir la columna 'timestamp' a datetime si no lo está
            self._price_data['timestamp'] = pd.to_datetime(self._price_data['timestamp'], errors='coerce')
            # Agrupar por mes y calcular la media
            monthly_data = self._price_data.resample('MS', on='timestamp')[['EUR_kg']].mean().reset_index()            
            # Asignar el resultado al atributo _price_data
            self._price_data = monthly_data
            return True
        except Exception as e:
            self.lastError = cfg.PRICEMODEL_ERROR_SMOOTH_PRICE.format(e=e.args[0])
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
    
    # Se establece el dataframe de precios procesados
    # Parámetros:
    # data: DataFrame con los datos de precios procesados
    # Retorna:
    # None
    def setPriceData(self, data):        
        self._price_data = data.copy()
        
    
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
    def find_optimal_configuration(self, train, test, max_window, n_iterations,
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
            stats_options = ['mean', 'std', 'min', 'max', 'sum', 'median', 'ratio_min_max', 'coef_variation', 'ewm']
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

        # Suprimir advertencias
        wn.filterwarnings('ignore', category=Warning)
    
        # Variables para almacenar los mejores resultados        
        best_score = 0.0  # Inicializar con 0 ya que queremos maximizarlo
        
        print(f"Probando {n_iterations} configuraciones aleatorias completas...")
    
        # Almacenar resultados
        results = []
    
        # Búsqueda aleatoria
        best = False        
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

                # 5. Elegir lags a usar
                if not lags is None:
                    # Asegurarse de que lags no exceda el tamaño de train
                    lags_to_use=min(lags,len(train)-1)
                else:
                    num_lags_to_generate = random.randint(1, min(4, len(train)-1))
                    # Limitar el rango de los lags a un valor razonable (ej. hasta max_window o max_possible_lag)
                    max_lag_value_pool = min(max_window + 1, len(train) - 1 + 1)
                    if max_lag_value_pool < num_lags_to_generate: # Si no hay suficientes lags disponibles
                        lags_to_use = list(range(1, max_lag_value_pool)) # Usar todos los lags posibles
                    else:
                        lags_to_use = sorted(random.sample(range(1, max_lag_value_pool), k=num_lags_to_generate))
            
                # Regresor con los parámetros seleccionados
                regressor = LGBMRegressor(**params)
                # Crear el modelo ForecasterRecursive con las características de regresor, lags y ventana
                #lags_to_use = [3,6,12,24]
                forecaster = ForecasterRecursive(
                    regressor=regressor,                    
                    lags=lags_to_use,
                    window_features=window_features
                )
            
                # 6. Entrenar el modelo con los datos de train y predecir una longitud igual a test
                forecaster.fit(y=y_train)
                predictions = forecaster.predict(steps=len(test))

                # 7. Evaluar el modelo
                # MAE (Error absoluto medio)                
                mae = mean_absolute_error(test['y'].values, predictions.values)
                # RMSE (Raíz del error cuadrático medio)
                rmse = np.sqrt(mean_squared_error(test['y'].values, predictions.values))
                # MAPE (Error porcentual)
                def mape(y_true, y_pred):
                    return np.mean(np.abs((y_true - y_pred) / (y_true+ 1e-8))) * 100 # Pequeño epsilon 1e-8 para evitar división por cero
                mape_value = mape(test['y'].values, predictions.values)

                # Dirección de cambio (tendencias acertadas)
                def direction_accuracy(y_true, y_pred):
                    if len(y_true) <= 1:
                        return 0
                    min_len = min(len(y_true), len(y_pred))
                    if min_len <= 1: # Si no hay al menos 2 puntos para calcular diff
                        return 0                  
                    direction_true = np.diff(y_true[:min_len]) > 0
                    direction_pred = np.diff(y_pred[:min_len]) > 0

                    if len(direction_true) == 0 or len(direction_pred) == 0:
                        return 0
                    # Asegurarse de que ambos arrays tengan la misma longitud
                    return np.mean(direction_true == direction_pred) * 100                
                dir_acc = direction_accuracy(test['y'].values, predictions.values)

                # Ponderaciones según importancia (deben sumar 1.0)
                score = (
                    0.30 * (1.0 - mae/5.0) +            # MAE normalizado (menor es mejor)
                    0.30 * (1.0 - rmse/8.0) +           # RMSE normalizado (menor es mejor)
                    0.20 * (1.0 - mape_value/100) +     # MAPE (menor es mejor)
                    0.20 * (dir_acc/100)                # Acierto direccional (mayor es mejor)
                )

                # Actualizar si esta configuración es mejor
                # Nota: buscamos maximizar score, no minimizar MAE
                if score > best_score:  
                    best_score = score

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
                        'lags':     lags_to_use,               
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
                print(f"\nError en iteración {i+1}: {e}")
                return None
    
        # 8. Ordenar y mostrar mejores resultados
        results.sort(key=lambda x: x['score'],reverse=True)
        # Depuración de resultados
        print(f"train ini:{train['ds'].dt.date.min()} fin:{train['ds'].dt.date.max()}")
        print(f"test ini:{test['ds'].dt.date.min()} fin:{test['ds'].dt.date.max()}")
        # Resumen de resultados        
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
    
    # Ajusta el modelo de precios con los datos proporcionados
    # Parámetros:
    # percent: Porcentaje de datos a usar para el entrenamiento (0 - 1000)
    # start_date: Fecha de inicio del rango de datos a usar (opcional)
    # end_date: Fecha de fin del rango de datos a usar (opcional)
    # adjust: Si es True, ajusta las ventanas y estadísticas del modelo (opcional)
    # estimator: Estimador a usar (opcional, no se usa en este caso)
    # Retorna:
    # bool: True si el ajuste fue exitoso, False en caso contrario    
    def fit_price(self, percent, start_date=None, end_date=None, adjust=False, estimator=None, prev_start_date=None):

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
            
            if prev_start_date:
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date >= prev_start_date]                
            elif start_date:
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
            delta_days = (filtered_data['timestamp'].dt.date.max() - start_date).days
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
            # 30.4375 es el promedio de días en un mes (365.25 / 12)
            delta_days_forescast = (test['ds'].dt.date.max() - test['ds'].dt.date.min()).days
            if delta_days_forescast > 0:
                delta_months_forecast = max(1, int(1 + round(delta_days_forescast / 30.4375)))
            else:
                delta_months_forecast = 0

            # Variables fijadas para la búsqueda
            if not adjust:
                # Si no se ajusta, se fijan las ventanas y estadísticas por defecto                
                windows = [min(4, len(train)-4), min(12, len(train)-3), min(26, len(train)-2), min(53, len(train)-1)]
                stats = ["mean", "mean","mean","mean"]
                lags_to_use = min(53, len(train)-1)  # Asegurarse de que lags no exceda el tamaño de train
                params = {'random_state': 15926,'verbose': -1}            
            else:
                windows = estimator['windows']
                stats = estimator['stats']
                lags_to_use = estimator['lags']
                params = {
                        'n_estimators': estimator['params']['n_estimators'],
                        'learning_rate': estimator['params']['learning_rate'],
                        'max_depth': estimator['params']['max_depth'],
                        'num_leaves': estimator['params']['num_leaves'],
                        'min_child_samples': estimator['params']['min_child_samples'],
                        'subsample': estimator['params']['subsample'],
                        'colsample_bytree': estimator['params']['colsample_bytree'],
                        'random_state': estimator['params']['random_state'],
                        'verbose': estimator['params']['verbose']
                    }
                
            # Suprimir advertencias
            wn.filterwarnings('ignore', category=Warning)
            
            # Window features
            window_features = RollingFeatures(
                    stats=stats,
                    window_sizes=windows
                )
            # Regresor con los parámetros seleccionados
            regressor = LGBMRegressor(**params)
            # Crear el modelo ForecasterRecursive con las características de regresor, lags y ventana
            forecaster = ForecasterRecursive(
                regressor=regressor,                    
                lags=lags_to_use,
                window_features=window_features
            )
            
            # Entrenar el modelo con los datos de train y predecir una longitud igual a test
            y_train = pd.Series(
                data=train['y'].values,
                index=pd.DatetimeIndex(train['ds']),
                name='EUR_kg'
            )
            forecaster.fit(y_train)
            # Generar predicciones para el número de semanas restantes
            predictions = forecaster.predict(steps=delta_months_forecast)
            
            self._price_data_forescast = pd.DataFrame({
                'ds': pd.date_range(start=current_date, periods=delta_months_forecast, freq='MS'),
                'y': predictions.values
            })

            # Mostrar parametros de entrenamiento
            print("--- Parameters used for training ---")
            print(f"Stats: {stats}")
            print(f"Windows:{windows}")
            print(f"Lags: {lags_to_use}")
            print(f"Params: {params}")      
            print("---")
            
            return True

        except Exception as e:
            self.lastError= cfg.PRICEMODEL_FIT_ERROR.format(e=str(e))
            return False

    # Se guardan los mejores estimadores en un archivo JSON    
    def save_top_estimators(self, new_estimators):
        self.lastError = None
        # Cargar los existentes si el archivo existe
        json_path = cfg.PRICEMODEL_TOP_ESTIMATORS_FILE

        # Si el archivo no existe, créalo vacío
        if not os.path.exists(json_path):
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump([], f, indent=4)
            except Exception as e:
                self.lastError = cfg.PRICEMODEL_ERROR_CREATE_FILE.format(path=json_path, e=e.args[0])
                return False

        # Cargar los existentes
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except Exception as e:
                self.lastError = cfg.PRICEMODEL_ERROR_LOAD_FILE.format(path=json_path, e=e.args[0])
                return False
            
        # Combinar y ordenar por score descendente
        combined = existing + new_estimators
        combined.sort(key=lambda x: x['score'], reverse=True)

        # Mantener solo los 5 mejores
        top5 = combined[:5]

        # Guardar en el archivo
        with open(json_path, "w", encoding="utf-8") as f:
            try:
                json.dump(top5, f, indent=4)
                return True
            except Exception as e:
                self.lastError = cfg.PRICEMODEL_ERROR_SAVE_FILE.format(path=json_path, e=e.args[0])
                return False

    # Se obtienen los mejores estimadores guardados en el archivo JSON
    # Retorna una lista vacía si no existen o hay error de lectura       
    def get_saved_top_estimators(self):
        self.lastError = None
        # Verificar si el archivo existe      
        json_path = cfg.PRICEMODEL_TOP_ESTIMATORS_FILE
        if not os.path.exists(json_path):
            return []
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                estimators = json.load(f)
                if isinstance(estimators, list):
                    return estimators
                else:
                    return []
        except Exception as e:
            self.lastError = cfg.PRICEMODEL_ERROR_LOAD_FILE.format(path=json_path, e=e.args[0])
            return []
            

