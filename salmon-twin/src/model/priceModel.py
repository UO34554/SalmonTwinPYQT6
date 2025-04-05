"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import pandas as pd
from datetime import datetime

class DataPrice:
    def __init__(self):
        # Inicializar el atributo price_data como DataFrame vacío
        self.price_data = None

    # Se almacenan los datos de los precios en un DataFrame
    def parsePrice(self, data):
        try:
            # Asignar los datos recibidos al atributo price_data
            self.price_data = data
            
            if 'Year' in self.price_data and 'Week' in self.price_data and 'EUR_kg' in self.price_data:
                # Convert the timestamp to a date
                for i, row in self.price_data.iterrows():
                    try:
                        year = int(row['Year'])
                        week = int(row['Week'])
                        temp_date = datetime.strptime(f'{year}-W{week}-1', "%G-W%V-%u")
                        self.price_data.at[i, 'timestamp'] = pd.to_datetime(temp_date)
                        #print("i: ",i, "timestamp: ",temp_date)                        
                    except ValueError:
                        raise ValueError("It was not possible to convert the timestamp to a date.")
                    try:
                        self.price_data.at[i, 'EUR_kg'] = float(row['EUR_kg'])
                    except ValueError:
                        raise ValueError("It was not possible to convert the price to a float.")
                # Sort the data by timestamp
                self.price_data = self.price_data.sort_values(by='timestamp')
            else:
                raise ValueError("The required columns 'Year', 'Week', 'Month' and 'EUR_kg' are not present in the data.")
        except ValueError as e:
            print(f"Error: {e}")