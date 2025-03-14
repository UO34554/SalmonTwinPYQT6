"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from datetime import datetime
import pandas as pd

# Clase que representa una balsa marina
class seaRaft:        
    
    def __init__(self, id=None, name=None, seaRegion=None, startDate=None, endDate=None, temperature=None):
        self.id = id
        self.name = name
        self.seaRegion = seaRegion
        self.startDate = startDate
        self.endDate = endDate        
        self.temperature = temperature

    # --- Setters ---
    def setId(self, id:int):
        self.id = int(id)

    def setName(self, name:str):
        self.name = str(name)

    def setSeaRegion(self, seaRegion:str):
        self.seaRegion = str(seaRegion)

    def setStartDate(self, startDate:datetime):
        # Convertir la fecha a las 00:00:00
        self.startDate = datetime.combine(startDate, datetime.min.time())

    def setEndDate(self, endDate:datetime):
        # Convertir la fecha a las 00:00:00
        self.endDate = datetime.combine(endDate, datetime.min.time())
        
    def setTemperature(self, temperature:pd.DataFrame):
        self.temperature = pd.DataFrame(temperature)

    # --- Getters ---
    def getId(self)->int:
        return int(self.id)

    def getName(self)->str:
        return str(self.name)

    def getSeaRegion(self)->str:
        return str(self.seaRegion)
    
    def getStartDate(self)->datetime:
        return self.startDate.date()
    
    def getEndDate(self)->datetime:
        return self.endDate.date()
    
    def getTemperature(self)->pd.DataFrame:
        return pd.DataFrame(self.temperature)
    
    # Convierte los datos de la balsa a un diccionario para serialización
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'seaRegion': self.seaRegion,
            'startDate': self.startDate.isoformat(),
            'endDate': self.endDate.isoformat()
            # No incluimos temperature aquí ya que es un DataFrame y no es serializable directamente
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