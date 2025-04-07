"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import pandas as pd
from datetime import datetime
import config as cfg

class DataPrice:
    def __init__(self):        
        self._price_data_raw = None
        self._price_data = None
        self.lastError = None

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
                        lastError=cfg.DASHBOARD_PRICE_PARSE_ERROR
                        return False
                    try:
                        self._price_data_raw.at[i, 'EUR_kg'] = float(row['EUR_kg'])
                    except ValueError:
                        lastError=cfg.PRICEMODEL_ERROR_PARSER_PRICE
                        return False
                # Sort the data by timestamp
                self._price_data = self._price_data_raw.sort_values(by='timestamp')
                return True
            else:
                lastError= cfg.PRICEMODEL_ERROR_PARSER_COLUMNS_ERROR
                return False
        except ValueError as e:
            lastError="Error: {e}"
            return False