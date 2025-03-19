"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import config as cfg
import pandas as pd

class DataTemperature:
    def __init__(self):
        # Nombre de las columnas de los datos de temperatura
        self.temp_column_names = cfg.TEMP_COLUMN_NAMES
        # Se crea un diccionario con el índice de la región y el nombre de la región sueca
        self.index_region = cfg.INDEX_SEA_REGIONS
        # Se crea un diccionario con el nombre del mes en sueco y el número del mes
        self.temp_month_names = cfg.SWEDISH_MONTH_NAMES
        # Se crea un DataFrame para cada región con las columnas 'ds' y 'y' que son requeridas por la libreria de Prophet
        self.data_regions = {}
        for region in self.index_region:
            self.data_regions[region] = pd.DataFrame(columns=['ds', 'y'])
        # Se crea un DataFrame para almacenar los datos de temperatura para la predicción futura
        self.data_regions_forecast = {}
        for region in self.index_region:
            self.data_regions_forecast[region] = pd.DataFrame(columns=['ds', 'yhat', 'yhat_lower', 'yhat_upper'])
        # Se almacena el último error
        self.lastError = None
        
    # Se almacenan los datos de temperatura en un DataFrame para cada región
    def parseTemperature(self, data):
        try:
            self.temp_data_raw = data
            # Verificar si las columnas requeridas están presentes en los datos            
            for name in self.temp_column_names:
                if (not name in self.temp_data_raw):
                    raise ValueError(cfg.PARSER_ERROR_COLUMN_NAME_NOT_FOUND.format(columnName=name))
            for i, row in self.temp_data_raw.iterrows():
                for region in self.index_region:
                    self.data_regions[region].at[i, 'ds'] = pd.to_datetime(str(row['Año']) + '-' + self.temp_month_names[row['Mes']], format='%Y-%m')
                    self.data_regions[region].at[i, 'y'] = row[self.index_region[region]]
            return True
        except ValueError as e:
            self.lastError=("Error:" + e.__str__())
            return False
        except Exception as e:
            self.lastError=("Error:" + e.__str__())
            return False
        
    def getTemperatureData(self, region):
        try:
            # Devolver el indice de la cadena region
            for i in self.index_region.keys():
                if region == self.index_region[i]:
                    return self.data_regions[i]
            return None
        except KeyError:
            self.lastError=cfg.REGION_NOT_FOUND.format(region)
            return None