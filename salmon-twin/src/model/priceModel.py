"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import pandas as pd
from datetime import datetime
import config as cfg
import os
from statsforecast import StatsForecast
from statsforecast.models import ARIMA

class DataPrice:
    def __init__(self):        
        self._price_data_raw = None
        self._price_data = None
        self._price_data_forescast = None
        self._price_data_train = None
        self._price_data_test = None
        self.lastError = None

    """
    Carga los datos iniciales desde el archivo de configuración
    si existe.    
    Retorna:
    bool: True si se cargaron los datos, False en caso contrario
    """
    def load_initial_data(self):
        if os.path.exists(cfg.PRICEMODEL_CONFIG_FILE):
            return self.load_from_json(cfg.PRICEMODEL_CONFIG_FILE)
        return False

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
            lastError="Error: {e}"
            return False

    """
    Retorna los datos de precios procesados
    Retorna:
    pd.DataFrame: DataFrame con los datos de precios procesados
    """
    def getPriceData(self):       
        return self._price_data
    
    """
        Retorna los datos de precios procesados para el conjunto de prueba
        Retorna:
        pd.DataFrame: DataFrame con los datos de precios procesados para el conjunto de prueba
        """
    def getPriceDataTest(self):        
        return self._price_data_test
    
    """
        Retorna los datos de precios procesados
        Retorna:
        pd.DataFrame: DataFrame con los datos de precios procesados
        """
    def getPriceDataForecast(self):        
        return self._price_data_forescast

    """
    Guarda los datos de precios en un archivo JSON    
    Parámetros:
    filepath (str): Ruta del archivo donde se guardará el JSON    
    Retorna:
    bool: True si se guardó correctamente, False en caso contrario
    """    
    def save_to_json(self, filepath):    
        try:
            if self._price_data is None or self._price_data.empty:
                self.lastError = cfg.PRICEMODEL_PRICE_EMPTY_DATA_SAVE_ERROR
                return False
            
            # Crear una copia con solo las columnas necesarias
            price_data_copy = self._price_data[['timestamp', 'EUR_kg']].copy()
        
            # Convertir la columna timestamp a formato ISO
            if 'timestamp' in price_data_copy.columns:
                price_data_copy['timestamp'] = price_data_copy['timestamp'].apply(
                    lambda x: x.isoformat() if pd.notnull(x) else None
                )
        
            # Convertir a formato JSON y guardar
            price_data_copy.to_json(filepath, orient='records', date_format='iso')
            return True
        
        except Exception as e:
            self.lastError = cfg.PRICEMODEL_PRICE_JSON_SAVE_ERROR.format(e)
            return False
    
    """
    Carga los datos de precios desde un archivo JSON    
    Parámetros:
    filepath (str): Ruta del archivo JSON a cargar    
    Retorna:
    bool: True si se cargó correctamente, False en caso contrario
    """
    def load_from_json(self, filepath):    
        try:
            # Cargar los datos del archivo JSON
            loaded_data = pd.read_json(filepath, orient='records')
        
            # Convertir la columna timestamp a datetime
            if 'timestamp' in loaded_data.columns:
                loaded_data['timestamp'] = pd.to_datetime(loaded_data['timestamp'])
            
            # Asignar los datos cargados
            self._price_data_raw = None
            self._price_data = loaded_data.sort_values(by='timestamp')
        
            return True
        except Exception as e:
            self.lastError = cfg.PRICEMODEL_PRICE_JSON_LOAD_ERROR.format(e)
            return False
        
    def fit_price(self):
        self.lastError = None
        if self._price_data is None:
             self.lastError = cfg.PRICEMODEL_FIT_NO_DATA
             return False
        try:
            # Filter data based on the selected dates
            #start_date = '2006-01-01'
            #end_date = '2012-12-31'     
            #filtered_data = self.price_data_raw[(self.price_data_raw['timestamp'] >= start_date) & (self.price_data_raw['timestamp'] <= end_date)]
            filtered_data = self._price_data.copy()
            # Asegúrate de que el DataFrame esté ordenado por la columna 'Fecha'
            filtered_data = filtered_data.sort_values(by='timestamp')

            # Crear un DataFrame directamente con los datos necesarios
            data = pd.DataFrame({
                'unique_id': ['EUR_kg_forecast'] * len(filtered_data),
                'ds': pd.to_datetime(filtered_data['timestamp']),
                'y': filtered_data['EUR_kg'].astype(float)  # Convertir toda la columna a float
            })           

            # #############################################################################            
            # Define el porcentaje para el conjunto de entrenamiento
            train_size = int(len(data) * 0.92)
            # Divide el DataFrame
            train = data.iloc[:train_size]
            test = data.iloc[train_size:]
            
            sf = StatsForecast(
                models=[ARIMA(order=(3, 0, 0), season_length=52, seasonal_order=(1, 1, 0))],
                freq='W',
                )

            sf.fit(train)

            horizonte = len(test)
            self._price_data_forescast = sf.predict(h=horizonte)

            # Importante: añadir las fechas de predicción
            # 1. Obtener la última fecha de los datos de entrenamiento
            last_date = train['ds'].iloc[-1]
        
            # 2. Generar un rango de fechas futuras semanales
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=7),  # Una semana después de la última fecha
                periods=horizonte,
                freq='W'  # Frecuencia semanal
            )

            # 3. Añadir las fechas al DataFrame de predicción
            self._price_data_forescast['ds'] = future_dates
            self._price_data_forescast['y'] = self._price_data_forescast['ARIMA'].astype(float)  # Convertir a float si es necesario
            self._price_data_test = test.copy()
            return True

        except ValueError as e:
            self.lastError= cfg.PRICEMODEL_FIT_ERROR.format(e)
            return False