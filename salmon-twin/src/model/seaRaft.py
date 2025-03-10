"""
@author: Pedro López Treitiño
"""
import datetime
import pandas as pd

class seaRaft:        
    
    def __init__(self, id=None, name=None, seaRegion=None, startDate=None, endDate=None, temperature=None):
        self.id = id
        self.name = name
        self.seaRegion = seaRegion
        self.startDate = startDate
        self.endDate = endDate        
        self.temperature = temperature

    def getId(self)->int:
        return self.id

    def getName(self)->str:
        return self.name

    def getSeaRegion(self)->str:
        return self.seaRegion
    
    def getStartDate(self)->datetime:
        return self.startDate
    
    def getEndDate(self)->datetime:
        return self.endDate
    
    def getTemperature(self)->pd.DataFrame:
        return self.temperature
    
    # Convierte los datos de la balsa a un diccionario para serialización
    def to_dict(self):        
        return {
            'id': self.id,
            'name': self.name,
            'seaRegion': self.seaRegion,
            'startDate': self.startDate.isoformat() if self.startDate else None,
            'endDate': self.endDate.isoformat() if self.endDate else None,
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
                    start_date = datetime.date.fromisoformat(data['startDate'])
                except ValueError as e:
                    lastError = f"Error al convertir fecha de inicio: {e}"
                    return None, lastError

            if 'endDate' in data and data['endDate']:
                try:
                    end_date = datetime.date.fromisoformat(data['endDate'])
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