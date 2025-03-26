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
            return None
        except KeyError:
            self.lastError=cfg.REGION_NOT_FOUND.format(region)
            return None
        
    # Predice los datos de temperatura en tempData
    def fitTempData(self,tempData,changepoint_range,changepoint_prior_scale,yearly_seasonality,periods):
        data_forecast = None
        fig_data_forecast = None
        fig_data_forecast_components = None

        #changepoint_range % of the data to be considered as changepoints (default 0.8) 80% of the data
        #changepoint_prior_scale parameter which determines the flexibility of the trend (default 0.05)
        #yearly_seasonality parameter which determines the flexibility of the yearly seasonality (default 10)

        # Crea un objeto de la libreria Prophet
        p = Prophet(changepoint_range=changepoint_range,changepoint_prior_scale=changepoint_prior_scale,yearly_seasonality=yearly_seasonality)
        
        # Ajusta el modelo a los datos de temperatura
        p.fit(tempData)
        
        # Predice los datos de temperatura para el futuro
        future_data = p.make_future_dataframe(periods)        
        data_forecast = p.predict(future_data)

        #--- Debug ---
        # Plot the forecast
        #fig_data_forecast = p.plot(data_forecast)
        # Add the changepoints to the forecast plot
        #add_changepoints_to_plot(fig_data_forecast.gca(), p, data_forecast)
        # Plot the forecast components
        #fig_data_forecast_components = p.plot_components(data_forecast)
        #return data_forecast,fig_data_forecast,fig_data_forecast_components
        #--- End Debug ---

        return data_forecast


        