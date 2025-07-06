"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import config as cfg
import pandas as pd
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

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

            # Convertir las columnas 'ds' y 'y' a los tipos de datos correctos
            # 'ds' debe ser de tipo datetime y 'y' debe ser de tipo numérico
            # Esto es necesario para que la libreria Prophet pueda trabajar con los datos
            # Se recorre cada región y se convierte el tipo de dato de las columnas
            for region in self.index_region:
                self.data_regions[region]['ds'] = pd.to_datetime(self.data_regions[region]['ds'])
                self.data_regions[region]['y'] = pd.to_numeric(self.data_regions[region]['y'])
            return True
        except ValueError as e:
            self.lastError=("Error:" + e.__str__())
            return False
        except Exception as e:
            self.lastError=("Error:" + e.__str__())
            return False

    # Devuelve los datos de temperatura para la región seleccionada    
    def getTemperatureData(self, region):
        try:
            # Devolver el indice de la cadena region
            for i in self.index_region.keys():
                if region == self.index_region[i]:
                    return self.data_regions[i]
            self.lastError=cfg.REGION_NOT_FOUND.format(region)
            return None
        except KeyError:
            self.lastError=cfg.REGION_NOT_FOUND.format(region)
            return None
        
    # Predice los datos de temperatura para un futuro periodo de tiempo en dias
    def fitTempData(self,tempData,periods):
        data_forecast = None

        try:
            # Crea un objeto de la libreria Prophet
            # Análisis detallado:
            # Prophet es una potente biblioteca de forecasting desarrollada por Facebook (Meta) que descompone
            # las series temporales en tres componentes principales: tendencia, estacionalidad y efectos de festividades. 
            # yearly_seasonality=2 indica que se espera que la estacionalidad anual tenga un efecto significativo en los datos.
            # Esto es útil para datos que tienen patrones estacionales claros, como las temperaturas del mar, que pueden variar
            # significativamente a lo largo del año.
            # En este caso, se ha establecido yearly_seasonality=2 para permitir que el modelo capture patrones estacionales más complejos.
        
            p = Prophet(yearly_seasonality=2)           

            # Ajusta el modelo a los datos de temperatura
            p.fit(tempData)
        
            # Predice los datos de temperatura para el futuro
            # Se crea un DataFrame con las fechas futuras para la predicción
            # periods: número de días a predecir
            # freq: frecuencia de la predicción (D para días)
            # include_history: si se incluyen los datos históricos en la predicción
            future_data = p.make_future_dataframe(periods, freq='D', include_history=False)        
            data_forecast = p.predict(future_data)

            # Limita la previsión con un máximo de 30ºC y un mínimo de 0ºC
            data_forecast['yhat'] = data_forecast['yhat'].clip(lower=0, upper=30)
            data_forecast['yhat_lower'] = data_forecast['yhat_lower'].clip(lower=0, upper=30)
            data_forecast['yhat_upper'] = data_forecast['yhat_upper'].clip(lower=0, upper=30)

            #--- Debug ---
            # Plot the forecast
            #fig_data_forecast = p.plot(data_forecast)
            # Add the changepoints to the forecast plot
            #add_changepoints_to_plot(fig_data_forecast.gca(), p, data_forecast)
            # Plot the forecast components
            #fig_data_forecast_components = p.plot_components(data_forecast)
            #return data_forecast,fig_data_forecast,fig_data_forecast_components
            #--- End Debug ---
        except ValueError as e:
            self.lastError=("Error:" + e.__str__())
            return None

        return data_forecast


        