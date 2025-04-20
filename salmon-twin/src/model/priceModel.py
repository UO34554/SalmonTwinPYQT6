"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import pandas as pd
from datetime import datetime, timedelta
import config as cfg
import os
from statsforecast import StatsForecast
from statsforecast.models import ARIMA

class DataPrice:
    def __init__(self):
        # Datos de precio en bruto        
        self._price_data_raw = None
        # Datos de precio procesados
        self._price_data = None
        # Datos de precio de la predicción
        self._price_data_forescast = None       
        # Datos del ultimo error
        self.lastError = None

        # Con objeto de depurar el modelo de predicción pero no necesarios para el funcionamiento
        self._price_data_test = None
        self._price_data_train = None

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
            self.lastError="Error: {e}"
            return False

    """
    Retorna los datos de precios procesados
    Retorna:
    pd.DataFrame: DataFrame con los datos de precios procesados
    """
    def getPriceData(self):       
        return self._price_data
    
    """
        Retorna los datos de precios procesados
        Retorna:
        pd.DataFrame: DataFrame con los datos de precios procesados
        """
    def getPriceDataForecast(self):        
        return self._price_data_forescast
    
    # Se ajusta el modelo de precios utilizando los datos de precios procesados
    # Parámetros:
    # start_date (datetime): Fecha inicial para el ajuste del modelo
    # end_date (datetime): Fecha final para el ajuste del modelo
    # horizon_days (int): Número de días para la predicción
    # Retorna:
    # bool: True si se ajustó correctamente, False en caso contrario   
    def fit_price(self, start_date=None, end_date=None, horizon_days=365):
        self.lastError = None
        if self._price_data is None:
             self.lastError = cfg.PRICEMODEL_FIT_NO_DATA
             return False
        try:
            # Filter data based on the selected dates
            filtered_data = self._price_data.copy()
            
            # Filtrar por fecha inicial si se proporciona
            if start_date:
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date >= start_date]
                
            # Filtrar por fecha final si se proporciona
            if end_date:
                filtered_data = filtered_data[filtered_data['timestamp'].dt.date <= end_date]
                
            # Asegurarse de que el DataFrame esté ordenado por la columna 'timestamp'
            filtered_data = filtered_data.sort_values(by='timestamp')

            # Verificar si hay datos suficientes después del filtrado
            if len(filtered_data) < 10:  # Establecer un mínimo razonable de puntos
                self.lastError = "No hay suficientes datos para el rango de fechas seleccionado"
                return False

            # Crear un nuevo DataFrame con las columnas requeridas por StatsForecast
            # Se crea un nuevo DataFrame con las columnas 'unique_id', 'ds' y 'y'
            # 'unique_id' es un identificador único para la serie temporal
            # 'ds' es la fecha y hora de la observación
            # 'y' es el valor de la observación (en este caso, el precio en EUR/kg)
            data = pd.DataFrame({
                'unique_id': ['EUR_kg_forecast'] * len(filtered_data),
                'ds': pd.to_datetime(filtered_data['timestamp']),
                'y': filtered_data['EUR_kg'].astype(float)  # Convertir toda la columna a float
            })
            
            # Define el porcentaje para el conjunto de entrenamiento
            train_size = int(len(data) * 0.92)
            # Divide el DataFrame
            train = data.iloc[:train_size]
            test = data.iloc[train_size:]
            
            # **************** Ajusta el modelo ARIMA
            # Se crea un objeto StatsForecast con el modelo ARIMA
            # Se especifica la frecuencia de los datos (semanal en este caso)
            # Se utiliza el modelo ARIMA con orden (3, 0, 0) y estacionalidad semanal (52 semanas)
            # Se ajusta el modelo a los datos de entrenamiento
            # Se predice el horizonte especificado (número de días)
            # Se crea un objeto StatsForecast con el modelo ARIMA
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
            self._price_data_train = train.copy()
            return True

        except ValueError as e:
            self.lastError= cfg.PRICEMODEL_FIT_ERROR.format(e)
            return False

