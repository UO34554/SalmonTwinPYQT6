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
    def __init__(self, id=None, name=None, seaRegion=None, startDate=None, endDate=None, perCentage=None, 
                 temperature=None, temperatureForecast=None, price=None, priceForecast=None,growth=None, growthForecast=None,
                 numberFishes=None):
        self._id = id
        self._name = name
        self._seaRegion = seaRegion
        self._startDate = startDate
        self._perCentage = perCentage
        self._endDate = endDate        
        self._temperature = temperature
        self._temperatureForecast = temperatureForecast
        self._price = price
        self._priceForecast = priceForecast
        self._growth = growth
        self._growthForecast = growthForecast
        self._numberFishes = numberFishes        

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

    def setPerCentage(self, perCentage:int):
        self._perCentage = int(perCentage)

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

    def setGrowth(self, growth:pd.DataFrame):
        self._growth = pd.DataFrame(growth)

    def setGrowthForecast(self, growth_forecast:pd.DataFrame):
        self._growthForecast = pd.DataFrame(growth_forecast)

    def setNumberFishes(self, numberFishes:int):
        self._numberFishes = int(numberFishes)

    # --- Getters ---
    def getId(self)->int:
        return int(self._id)

    def getName(self)->str:
        return str(self._name)

    def getSeaRegion(self)->str:
        return str(self._seaRegion)
    
    def getStartDate(self)->datetime:
        return self._startDate.date()
    
    def getPerCentage(self)->int:
        # Si el porcentaje es None, devolver 25
        if self._perCentage is None:
            return 25
        return int(self._perCentage)
    
    def getEndDate(self)->datetime:
        return self._endDate.date()
    
    def getTemperature(self)->pd.DataFrame:
        return pd.DataFrame(self._temperature)
    
    # Devuelve la fecha actual calculada a partir de la fecha de inicio y el porcentaje
    def getCurrentDate(self)->datetime:        
        return self._startDate + (self._endDate - self._startDate) * (self._perCentage / 100.0)
    
    # Devuelve la temperatura interpolada para una fecha dada
    # Se usa el porcentage para calcular la fecha actual
    def geCurrentDateTemperature(self)->float:    
        if self._temperature is None or self._temperature.empty:
            return None
    
        # Asegurar que trabajamos con fechas en formato datetime
        target_date = pd.to_datetime(self.getCurrentDate())
        sorted_temp = self.getTemperature()       
        sorted_temp['ds'] = pd.to_datetime(self.getTemperature()['ds'])
        # Encontrar la fecha anterior y posterior más cercanas
        prev_date_idx = sorted_temp[sorted_temp['ds'] <= target_date]['ds'].idxmax() if not sorted_temp[sorted_temp['ds'] <= target_date].empty else None
        next_date_idx = sorted_temp[sorted_temp['ds'] > target_date]['ds'].idxmin() if not sorted_temp[sorted_temp['ds'] > target_date].empty else None
    
        # Caso 1: Fecha exacta encontrada
        if prev_date_idx is not None and sorted_temp.loc[prev_date_idx, 'ds'] == target_date:
            return sorted_temp.loc[prev_date_idx, 'y']
    
        # Caso 2: Fecha está antes de la primera medición
        if prev_date_idx is None and next_date_idx is not None:
            return sorted_temp.loc[next_date_idx, 'y']
    
        # Caso 3: Fecha está después de la última medición
        if prev_date_idx is not None and next_date_idx is None:
            return sorted_temp.loc[prev_date_idx, 'y']
    
        # Caso 4: Fecha está entre dos mediciones - realizar interpolación lineal
        if prev_date_idx is not None and next_date_idx is not None:
            prev_date = sorted_temp.loc[prev_date_idx, 'ds']
            next_date = sorted_temp.loc[next_date_idx, 'ds']
            prev_temp = sorted_temp.loc[prev_date_idx, 'y']
            next_temp = sorted_temp.loc[next_date_idx, 'y']
        
        # Calcular la proporción del tiempo transcurrido
        total_days = (next_date - prev_date).total_seconds() / (60*60*24)
        days_passed = (target_date - prev_date).total_seconds() / (60*60*24)
        proportion = days_passed / total_days if total_days > 0 else 0
        
        # Interpolar linealmente
        interpolated_temp = prev_temp + proportion * (next_temp - prev_temp)
        return interpolated_temp
    
    def getTemperatureForecast(self)->pd.DataFrame:
        return pd.DataFrame(self._temperatureForecast)
    
    def getPrice(self)->pd.DataFrame:
        return pd.DataFrame(self._price)
    
    def getPriceForecast(self)->pd.DataFrame:
        return pd.DataFrame(self._priceForecast)
    
    def getGrowth(self)->pd.DataFrame:
        return pd.DataFrame(self._growth)
    
    def getGrowthForecast(self)->pd.DataFrame:
        return pd.DataFrame(self._growthForecast)
    
    def getNumberFishes(self)->int:
        if self._numberFishes is None:
            return 0
        else:
            return self._numberFishes
    
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
        
        if self._growth is not None:
            # Asegurarse de que la columna 'ds' sea de tipo datetime
            self._growth['ds'] = pd.to_datetime(self._growth['ds'], errors='coerce')

            # Aplicar .isoformat() directamente a cada valor de la columna 'ds'
            self._growth['ds'] = self._growth['ds'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

            # Convertir a un formato serializable
            growth_data = self._growth.to_dict(orient='records')
        else:
            growth_data = None

        if self._growthForecast is not None:
            # Asegurarse de que la columna 'ds' sea de tipo datetime
            self._growthForecast['ds'] = pd.to_datetime(self._growthForecast['ds'], errors='coerce')

            # Aplicar .isoformat() directamente a cada valor de la columna 'ds'
            self._growthForecast['ds'] = self._growthForecast['ds'].apply(lambda x: x.isoformat() if pd.notnull(x) else None)

            # Convertir a un formato serializable
            growth_forecast_data = self._growthForecast.to_dict(orient='records')
        else:
            growth_forecast_data = None

        if self._numberFishes is not None:
            numberFishes = int(self._numberFishes)
        else:
            numberFishes = 0

        return {
            'id': self._id,
            'name': self._name,
            'seaRegion': self._seaRegion,
            'startDate': self._startDate.isoformat(),
            'perCentage': self._perCentage,
            'endDate': self._endDate.isoformat(),
            'temperature': temperature_data,
            'temperatureForecast': temperature_forecast_data,
            'price': price_data,
            'priceForecast': price_forecast_data,
            'growth': growth_data,
            'growthForecast': growth_forecast_data,
            'numberFishes': numberFishes
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
                
            growth = None
            if 'growth' in data and data['growth']:
                try:                    
                    growth = pd.DataFrame(data['growth'])
                except Exception as e:
                    lastError = cfg.RAFTS_ERROR_PARSER_GROWTH.format(e)
                    return None, lastError
                
            growthForecast = None
            if 'growthForecast' in data and data['growthForecast']:
                try:                    
                    growthForecast = pd.DataFrame(data['growthForecast'])
                except Exception as e:
                    lastError = cfg.RAFTS_ERROR_PARSER_GROWTH_FORECAST.format(e)
                    return None, lastError

            # Crear el objeto seaRaft
            return seaRaft(
                id=data.get('id'),
                name=data.get('name'),
                seaRegion=data.get('seaRegion'),
                startDate=start_date,
                perCentage=data.get('perCentage'),
                endDate=end_date,
                temperature=temperature,
                temperatureForecast=temperatureForecast,
                price=price,
                priceForecast=priceForecast,
                growth=growth,
                growthForecast=growthForecast,
                numberFishes=data.get('numberFishes')
            ), lastError
        except Exception as e:
            lastError = cfg.RAFTS_ERROR_FROM_DICT_TO_RAFT.format(e)
        return None, lastError
