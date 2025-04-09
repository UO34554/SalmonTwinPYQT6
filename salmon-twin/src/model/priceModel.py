"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import pandas as pd
from datetime import datetime
import config as cfg
import os

class DataPrice:
    def __init__(self):        
        self._price_data_raw = None
        self._price_data = None
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