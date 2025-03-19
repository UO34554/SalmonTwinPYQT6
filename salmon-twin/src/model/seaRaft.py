"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from datetime import datetime
import pandas as pd

# Clase que representa una balsa marina
class seaRaft:        
    
    def __init__(self, id=None, name=None, seaRegion=None, startDate=None, endDate=None, temperature=None):
        self._id = id
        self._name = name
        self._seaRegion = seaRegion
        self._startDate = startDate
        self._endDate = endDate        
        self._temperature = temperature

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

        return {
            'id': self._id,
            'name': self._name,
            'seaRegion': self._seaRegion,
            'startDate': self._startDate.isoformat(),
            'endDate': self._endDate.isoformat(),
            'temperature': temperature_data
        }
    
    # Convierte los datos de la balsa a un diccionario para serialización
    @staticmethod
    def from_dict(data):
        lastError = None
        try:
            # Convertir las fechas de ISO formato si existen
            start_date = None
            end_date = None

            if 'startDate' in data and data['startDate']:
                try:
                    start_date = datetime.fromisoformat(data['startDate'])
                except ValueError as e:
                    lastError = f"Error al convertir fecha de inicio: {e}"
                    return None, lastError

            if 'endDate' in data and data['endDate']:
                try:
                    end_date = datetime.fromisoformat(data['endDate'])
                except ValueError as e:
                    lastError = f"Error al convertir fecha de fin: {e}"
                    return None, lastError

            return seaRaft(
                id=data.get('id'),
                name=data.get('name'),
                seaRegion=data.get('seaRegion'),
                startDate=start_date,
                endDate=end_date,
                temperature=None  # La temperatura se cargará por separado
            ), lastError
        except Exception as e:
            lastError = f"Error al crear balsa desde diccionario: {e}"
            return None, lastError