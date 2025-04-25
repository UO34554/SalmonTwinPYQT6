"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import pandas as pd
from datetime import datetime, timedelta
import config as cfg
import os
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, HoltWinters, ARIMA

# Se define la clase DataPrice para gestionar los datos de precios
# Esta clase se encarga de parsear los datos de precios, ajustarlos y predecirlos
# Se espera que los datos de precios contengan las columnas 'Year', 'Week' y 'EUR_kg'
# La columna 'Year' contiene el año de la observación
# La columna 'Week' contiene la semana del año de la observación
# La columna 'EUR_kg' contiene el precio en euros por kilogramo
# La clase también se encarga de gestionar los errores que puedan ocurrir durante el proceso
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

    # Se obtiene el dataframe de precios procesados
    # Parámetros:
    # None
    # Retorna:
    # pd.DataFrame: DataFrame con los datos de precios procesados
    def getPriceData(self):       
        return self._price_data
    
    # Se obtiene el dataframe de precios de la predicción
    # Parámetros:
    # None
    # Retorna:
    # pd.DataFrame: DataFrame con los datos de precios de la predicción
    def getPriceDataForecast(self):        
        return self._price_data_forescast
    
    # Se asignan los datos de precios a la variable de instancia
    # Parámetros:
    # data (pd.DataFrame): DataFrame que contiene los datos de precios
    # Se espera que el dataframe contenga las columnas 'Year', 'Week' y 'EUR_kg'
    # Retorna:
    # bool: True si se asignaron correctamente, False en caso contrario
    def setPriceData(self, data):        
        self._price_data = data.copy()
        # Se procesan los datos de precios
        if not self.parsePrice(self._price_data):
            return False
        return True
    
    # Se ajusta el modelo de precios utilizando los datos de precios procesados
    # Parámetros:
    # start_date (datetime): Fecha inicial para el ajuste del modelo
    # end_date (datetime): Fecha final para el ajuste del modelo
    # horizon_days (int): Número de días para la predicción
    # Retorna:
    # bool: True si se ajustó correctamente, False en caso contrario   
    def fit_price(self, slider_value, start_date=None, end_date=None, horizon_days=365):
        self.lastError = None
        if self._price_data is None:
             self.lastError = cfg.PRICEMODEL_FIT_NO_DATA
             return False
        try:
            # Filter data based on the selected dates
            filtered_data = self._price_data.copy()
            filtered_data['timestamp'] = pd.to_datetime(filtered_data['timestamp'], errors='coerce')
            # Eliminar filas con valores NaT (fechas inválidas)
            filtered_data = filtered_data.dropna(subset=['timestamp'])           
            
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
                self.lastError = cfg.PRICEMODEL_NOT_ENOUGHT_DATA
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
            percent=slider_value/100

            delta_days = (end_date - start_date).days
            if delta_days > 0:  # Protect against division by zero
                current_day_offset = int(delta_days * percent)
                current_date = start_date + timedelta(days=current_day_offset)
            
            # Divide el DataFrame
            train = data[data['ds'].dt.date <= current_date]
            test = data[data['ds'].dt.date > current_date]
            
            # **************** Ajusta el modelo AutoARIMA ****************
            modelo = AutoARIMA(                
                seasonal=True,                        
                stepwise=False,
                trace=True,                                                        
            )

            modelos = [
                ARIMA(order=(1, 1, 4),alias='ARIMA(1,1,4)'),  # El modelo que AutoARIMA seleccionó previamente                
                HoltWinters(season_length=int(52*percent),alias='HoltWinters_seasonal'),  # Modelo Holt-Winters con estacionalidad anual (para datos semanales)                
            ]
            
            sf = StatsForecast(
                models=modelos,
                freq='W', # Especifica la frecuencia de tus datos (en este caso, semanal 'W')                 
                verbose=True  # Activar modo detallado para ver el progreso
            )

            # Ajusta el modelo a los datos de entrenamiento
            # Se utiliza el DataFrame filtrado como datos de entrenamiento            
            sf.fit(train)

            horizon_weeks = int(horizon_days*percent/7)
            self._price_data_forescast = sf.predict(h=horizon_weeks)  # Cambia el horizonte según tus necesidades

            # Importante: añadir las fechas de predicción
            # 1. Obtener la última fecha de los datos de entrenamiento
            last_date = train['ds'].iloc[-1]
        
            # 2. Generar un rango de fechas futuras semanales
            future_dates = pd.date_range(
                start=last_date, #+ pd.Timedelta(days=7),  # Una semana después de la última fecha
                periods=horizon_weeks,  # Número de semanas a predecir
                freq='W'  # Frecuencia semanal
            )

            # 3. Añadir las fechas al DataFrame de predicción
            self._price_data_forescast['ds'] = future_dates            
            varianza = self._price_data_forescast['ARIMA(1,1,4)'].astype(float) - self._price_data_forescast['HoltWinters_seasonal'].astype(float)
            self._price_data_forescast['y'] = self._price_data_forescast['ARIMA(1,1,4)'].astype(float) + varianza**2
            
            self._price_data_test = test.copy()
            self._price_data_train = train.copy()
            return True

        except ValueError as e:
            self.lastError= cfg.PRICEMODEL_FIT_ERROR.format(e=e.args[0])
            return False

