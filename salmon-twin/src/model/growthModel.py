"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import numpy as np
import pandas as pd

class GrowthModel:

    def __init__(self):
        None

    # Thyholdt (2014) función de crecimiento
    # t: tiempo en meses
    # T: temperatura del mar in grados Celsius
    # alpha: peso máximo asintótico en gramos
    # beta: coeficiente de pendiente
    # mu(mi): punto de inflexión en meses
    # return W: peso medio del salmon en el tiempo t
    def _thyholdt_function(self, t, T, alpha, beta, mu):        
        
        if t > 0:
            result = (alpha/1000.0) / (1 + np.exp(-(beta * T) * (t - mu)))
        else:
            result = 0.0
        
        return result
    
    # Mortandad de los salmones
    # initial_number_fishes: número inicial de salmones
    # mortality_percent: tasa de mortandad en tanto por ciento de la población por unidad de tiempo (por mes)
    # t: tiempo en meses
    def _mortality(self, initial_number_fishes, mortality_percent, t):
        return initial_number_fishes * (1 -mortality_percent) ** t      

    # Thyholdt (2014) modelo de crecimiento
    # data: Dataframe con las columnas 'ds' (fecha) e 'y' (temperatura)
    # dataForescast: Dataframe con las columnas 'ds' (fecha) e 'y' (temperatura)
    # alpha: peso máximo asintótico en gramos
    # beta: coeficiente de pendiente
    # mu (mi): punto de inflexión en meses
    # mortality_rate: tasa de mortandad en tanto por ciento de la población por unidad de tiempo (por mes)
    # initial_weight: peso inicial del salmon en gramos
    # initial_number_fishes: número inicial de salmones
    def thyholdt_growth(self, historicalTemp, forescastTemp, alpha, beta, mu, mortality_rate, initial_weight, initial_number_fishes):
       # Crea una copia del DataFrame y reinicia los índices para asegurar que sean 0, 1, 2...
        historicalTemp = historicalTemp.copy().reset_index(drop=True)
        forescastTemp = forescastTemp.copy().reset_index(drop=True)
        # Convertir las columnas 'ds' a formato datetime
        historicalTemp['ds'] = pd.to_datetime(historicalTemp['ds'])
        forescastTemp['ds'] = pd.to_datetime(forescastTemp['ds'])
        # Fecha inicial
        start_date = historicalTemp['ds'].iloc[0]
        growth_factor = 0

        # Agregar columnas para almacenar los resultados
        historicalTemp['function'] = 0.0
        historicalTemp['growth'] = 0.0
        historicalTemp['number_fishes'] = 0.0
        historicalTemp['biomass'] = 0.0

        # Agregar columnas para almacenar los resultados de la predicción
        forescastTemp['function'] = 0.0
        forescastTemp['growth'] = 0.0
        forescastTemp['number_fishes'] = 0.0
        forescastTemp['biomass'] = 0.0      

        # Calculo del modelo de crecimiento histórico
        for i in range(len(historicalTemp)):
            # Calcula el tiempo en meses desde la fecha inicial
            date_inc = (historicalTemp.loc[i,'ds'] - start_date).days / 30
            # Calcula la tasa de crecimiento usando la función de Thyholdt
            f = self._thyholdt_function(date_inc, historicalTemp.loc[i,'y'], alpha, beta, mu)           
            historicalTemp.loc[i,'function'] = f            
            # Calcula el factor de crecimiento usando el peso inicial y la función de crecimiento y el factor de crecimiento del mes anterior
            growth_factor = growth_factor + (1 + f) * initial_weight
            # Limita el factor de crecimiento al peso máximo asintótico
            if growth_factor > alpha:
                growth_factor = alpha
            historicalTemp.loc[i,'growth'] = growth_factor
            # Calcula el número de peces
            number_fishes = self._mortality(initial_number_fishes, mortality_rate, date_inc)
            historicalTemp.loc[i,'number_fishes'] = number_fishes
            # Calcula el peso total de la biomasa en KG           
            historicalTemp.loc[i,'biomass'] = growth_factor * number_fishes / 1000.0

        # Inicializa el crecimiento y el número de peces para la predicción        
        initial_number_fishes = historicalTemp.iloc[-1]['number_fishes']
        last_growth_factor = historicalTemp.iloc[-1]['growth']
        last_historical_date = historicalTemp.iloc[-1]['ds']
        last_historical_function = historicalTemp.iloc[-1]['function']

        # Calculo del modelo de crecimiento futuro
        for i in range(len(forescastTemp)):
            # Calcula el tiempo en meses desde la fecha inicial
            date_inc = (forescastTemp.loc[i,'ds'] - last_historical_date).days / 30
            
            f = self._thyholdt_function(date_inc, forescastTemp.loc[i,'yhat'], alpha, beta, mu)       
            
            # Calcula el factor de crecimiento usando el peso inicial y la función de crecimiento y el factor de crecimiento del mes anterior
            growth_factor = growth_factor + (1 + f) * initial_weight

            # Limita el factor de crecimiento al peso máximo asintótico
            if growth_factor > alpha:
                growth_factor = alpha

            forescastTemp.loc[i,'function'] = f 
            forescastTemp.loc[i,'growth'] = growth_factor
            # Calcula el número de peces
            number_fishes = self._mortality(initial_number_fishes, mortality_rate, date_inc)
            forescastTemp.loc[i,'number_fishes'] = number_fishes
            # Calcula el peso total de la biomasa en KG           
            forescastTemp.loc[i,'biomass'] = growth_factor * number_fishes / 1000.0

        return historicalTemp, forescastTemp
    