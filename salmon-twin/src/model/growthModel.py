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
    def thyholdt_growth(self, data, dataForescast, alpha, beta, mu, mortality_rate, initial_weight, initial_number_fishes):
       # Crea una copia del DataFrame y reinicia los índices para asegurar que sean 0, 1, 2...
        data = data.copy().reset_index(drop=True)
        dataForescast = dataForescast.copy().reset_index(drop=True)
        # Convertir las columnas 'ds' a formato datetime
        data['ds'] = pd.to_datetime(data['ds'])
        dataForescast['ds'] = pd.to_datetime(dataForescast['ds'])
        # Fecha inicial
        start_date = data['ds'].iloc[0]
        growth_factor = 0

        # Agregar columnas para almacenar los resultados
        data['function'] = 0.0
        data['growth'] = 0.0
        data['number_fishes'] = 0.0
        data['biomass'] = 0.0

        # Agregar columnas para almacenar los resultados de la predicción
        dataForescast['function'] = 0.0
        dataForescast['growth'] = 0.0
        dataForescast['number_fishes'] = 0.0
        dataForescast['biomass'] = 0.0      

        # Calculo del modelo de crecimiento histórico
        for i in range(len(data)):
            # Calcula el tiempo en meses desde la fecha inicial
            date_inc = (data.loc[i,'ds'] - start_date).days / 30
            # Calcula la tasa de crecimiento usando la función de Thyholdt
            f = self._thyholdt_function(date_inc, data.loc[i,'y'], alpha, beta, mu)           
            data.loc[i,'function'] = f            
            # Calcula el factor de crecimiento usando el peso inicial y la función de crecimiento y el factor de crecimiento del mes anterior
            growth_factor = growth_factor + (1 + f) * initial_weight
            # Limita el factor de crecimiento al peso máximo asintótico
            if growth_factor > alpha:
                growth_factor = alpha
            data.loc[i,'growth'] = growth_factor
            # Calcula el número de peces
            number_fishes = self._mortality(initial_number_fishes, mortality_rate, date_inc)
            data.loc[i,'number_fishes'] = number_fishes
            # Calcula el peso total de la biomasa en KG           
            data.loc[i,'biomass'] = growth_factor * number_fishes / 1000.0

        # Inicializa el crecimiento y el número de peces para la predicción
        initial_weight = data.iloc[-1]['growth']
        initial_number_fishes = data.iloc[-1]['number_fishes']
        last_growth_factor = data.iloc[-1]['growth']
        last_historical_date = data.iloc[-1]['ds']
        last_historical_function = data.iloc[-1]['function']

        # Calculo del modelo de crecimiento futuro
        for i in range(len(dataForescast)):
            # Calcula el tiempo en meses desde la fecha inicial
            date_inc = (dataForescast.loc[i,'ds'] - last_historical_date).days / 30

            # Factor de suavizado con una ventana temporal más amplia (2 meses en lugar de 0.5)
            smoothing_factor = min(1.0, date_inc / 2.0)

            # Calcula la tasa de crecimiento usando la función de Thyholdt            
            if i == 0:
                # Usar exactamente el mismo valor de función para el primer punto
                f = last_historical_function
                # Mantener exactamente el mismo crecimiento para el primer punto
                growth_factor = last_growth_factor
            else:
                # Mezcla gradual entre último valor histórico y nuevo valor calculado
                new_f = self._thyholdt_function(date_inc, dataForescast.loc[i,'yhat'], alpha, beta, mu)
        
                # Aplicar suavizado durante un período más largo (primeros 10 puntos)
                if i <= 10:  
                    f = last_historical_function * (1-smoothing_factor) + new_f * smoothing_factor
            
                    # Aplicar suavizado directamente al valor de crecimiento, no solo al incremento
                    new_growth = last_growth_factor + (1 + f) * initial_weight
                    growth_factor = last_growth_factor * (1-smoothing_factor) + new_growth * smoothing_factor
                else:
                    f = new_f
                    growth_factor = growth_factor + (1 + f) * initial_weight

            # Limita el factor de crecimiento al peso máximo asintótico
            if growth_factor > alpha:
                growth_factor = alpha

            dataForescast.loc[i,'function'] = f 
            dataForescast.loc[i,'growth'] = growth_factor
            # Calcula el número de peces
            number_fishes = self._mortality(initial_number_fishes, mortality_rate, date_inc)
            dataForescast.loc[i,'number_fishes'] = number_fishes
            # Calcula el peso total de la biomasa en KG           
            dataForescast.loc[i,'biomass'] = growth_factor * number_fishes / 1000.0

        return data, dataForescast
    