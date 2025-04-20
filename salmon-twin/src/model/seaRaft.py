"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from datetime import datetime
import pandas as pd
import config as cfg

# Clase que representa una balsa marina
class seaRaft:        
    # Constructor de la clase seaRaft    
    # Se inicializan los atributos de la clase seaRaft
    # id: int, nombre: str, regionMarina: str, fechaInicio: datetime, fechaFin: datetime, temperatura: pd.DataFrame, precio: pd.DataFrame
    # temperatureForecast: pd.DataFrame, priceForecast: pd.DataFrame
    def __init__(self, id=None, name=None, seaRegion=None, startDate=None, endDate=None, 
                 temperature=None, temperatureForecast=None, price=None, priceForecast=None):
        self._id = id
        self._name = name
        self._seaRegion = seaRegion
        self._startDate = startDate
        self._endDate = endDate        
        self._temperature = temperature
        self._temperatureForecast = temperatureForecast
        self._price = price
        self._priceForecast = priceForecast        

    # --- Setters ---
    def setId(self, id:int):
        self._id = int(id)

    def setName(self, name:str):
        self._name = str(name)

    def setSeaRegion(self, seaRegion:str):
        self._seaRegion = str(seaRegion)

    def setStartDate(self, startDate:datetime):
        # Convertir la fecha a las 00:00:00
        self._startDate = datetime.combine(startDate, datetime.min.time())

    def setEndDate(self, endDate:datetime):
        # Convertir la fecha a las 00:00:00
        self._endDate = datetime.combine(endDate, datetime.min.time())
        
    def setTemperature(self, temperature:pd.DataFrame):
        self._temperature = pd.DataFrame(temperature)

    def setTemperatureForecast(self, temp_forecast:pd.DataFrame):
        self._temperatureForecast = pd.DataFrame(temp_forecast)

    def setPrice(self, price:pd.DataFrame):
        self._price = pd.DataFrame(price)

    def setPriceForecast(self, price_forecast:pd.DataFrame):
        self._priceForecast = pd.DataFrame(price_forecast)

    # --- Getters ---
    def getId(self)->int:
        return int(self._id)

    def getName(self)->str:
        return str(self._name)

    def getSeaRegion(self)->str:
        return str(self._seaRegion)
    
    def getStartDate(self)->datetime:
        return self._startDate.date()
    
    def getEndDate(self)->datetime:
        return self._endDate.date()
    
    def getTemperature(self)->pd.DataFrame:
        return pd.DataFrame(self._temperature)
    
    def getTemperatureForecast(self)->pd.DataFrame:
        return pd.DataFrame(self._temperatureForecast)
    
    def getPrice(self)->pd.DataFrame:
        return pd.DataFrame(self._price)
    
    def getPriceForecast(self)->pd.DataFrame:
        return pd.DataFrame(self._priceForecast)
    
    # Convierte los datos de la balsa a un diccionario para serialización
    def to_dict(self):
        if self._temperature is not None:
            # Asegurarse de que la columna 'ds' sea de tipo datetime
            self._temperature['ds'] = pd.to_datetime(self._temperature['ds'], errors='coerce')

            # Aplicar .isoformat() directamente a cada valor de la columna 'ds'
            self._temperature['ds'] = self._temperature['ds'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

            # Convertir a un formato serializable
            temperature_data = self._temperature.to_dict(orient='records')
        else:
            temperature_data = None

        if self._temperatureForecast is not None:
            # Asegurarse de que la columna 'ds' sea de tipo datetime
            self._temperatureForecast['ds'] = pd.to_datetime(self._temperatureForecast['ds'], errors='coerce')

            # Aplicar .isoformat() directamente a cada valor de la columna 'ds'
            self._temperatureForecast['ds'] = self._temperatureForecast['ds'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

            # Convertir a un formato serializable
            temperature_forecast_data = self._temperatureForecast.to_dict(orient='records')
        else:
            temperature_forecast_data = None

        if self._price is not None:
            # Asegurarse de que la columna 'ds' sea de tipo datetime
            self._price['timestamp'] = pd.to_datetime(self._price['timestamp'], errors='coerce')

            # Aplicar .isoformat() directamente a cada valor de la columna 'ds'
            self._price['timestamp'] = self._price['timestamp'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

            # Convertir a un formato serializable
            price_data = self._price.to_dict(orient='records')
        else:
            price_data = None

        
        if self._priceForecast is not None:
            # Asegurarse de que la columna 'ds' sea de tipo datetime
            self._priceForecast['ds'] = pd.to_datetime(self._priceForecast['ds'], errors='coerce')

            # Aplicar .isoformat() directamente a cada valor de la columna 'ds'
            self._priceForecast['ds'] = self._priceForecast['ds'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

            # Convertir a un formato serializable
            price_forecast_data = self._priceForecast.to_dict(orient='records')
        else:
            price_forecast_data = None


        return {
            'id': self._id,
            'name': self._name,
            'seaRegion': self._seaRegion,
            'startDate': self._startDate.isoformat(),
            'endDate': self._endDate.isoformat(),
            'temperature': temperature_data,
            'temperatureForecast': temperature_forecast_data,
            'price': price_data,
            'priceForecast': price_forecast_data
        }
    
    # Crea una instancia de seaRaft a partir de un diccionario
    @staticmethod
    def from_dict(data):
        lastError = None
        try:
            # Convertir las fechas de formato ISO si existen
            start_date = None
            end_date = None

            if 'startDate' in data and data['startDate']:
                try:
                    start_date = datetime.fromisoformat(data['startDate'])
                except ValueError as e:
                    lastError = cfg.RAFTS_ERROR_PARSER_START_DATE.format(e)
                    return None, lastError

            if 'endDate' in data and data['endDate']:
                try:
                    end_date = datetime.fromisoformat(data['endDate'])
                except ValueError as e:
                    lastError = cfg.RAFTS_ERROR_PARSER_END_DATE.format(e)
                    return None, lastError

            # Reconstruir temperature como un DataFrame si existe
            temperature = None
            if 'temperature' in data and data['temperature']:
                try:                    
                    temperature = pd.DataFrame(data['temperature'])
                except Exception as e:
                    lastError = cfg.RAFTS_ERROR_PARSER_TEMPERATURE.format(e)
                    return None, lastError
                
            # Reconstruir temperatureForecast como un DataFrame si existe
            temperatureForecast = None
            if 'temperatureForecast' in data and data['temperatureForecast']:
                try:                    
                    temperatureForecast = pd.DataFrame(data['temperatureForecast'])
                except Exception as e:
                    lastError = cfg.RAFTS_ERROR_PARSER_TEMPERATURE_FORECAST.format(e)
                    return None, lastError
                
            # Reconstruir price como un DataFrame si existe
            price = None
            if 'price' in data and data['price']:
                try:                    
                    price = pd.DataFrame(data['price'])
                except Exception as e:
                    lastError = cfg.RAFTS_ERROR_PARSER_PRICE.format(e)
                    return None, lastError
                
            # Reconstruir priceForecast como un DataFrame si existe
            priceForecast = None
            if 'priceForecast' in data and data['priceForecast']:
                try:                    
                    priceForecast = pd.DataFrame(data['priceForecast'])
                except Exception as e:
                    lastError = cfg.RAFTS_ERROR_PARSER_PRICE_FORECAST.format(e)
                    return None, lastError

            # Crear el objeto seaRaft
            return seaRaft(
                id=data.get('id'),
                name=data.get('name'),
                seaRegion=data.get('seaRegion'),
                startDate=start_date,
                endDate=end_date,
                temperature=temperature,
                temperatureForecast=temperatureForecast,
                price=price,
                priceForecast=priceForecast
            ), lastError
        except Exception as e:
            lastError = cfg.RAFTS_ERROR_FROM_DICT_TO_RAFT.format(e)
        return None, lastError
