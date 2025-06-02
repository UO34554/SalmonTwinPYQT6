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
    def thyholdt_growth(self, dataHistoricalTemp, dataForescastTemp, alpha, beta, mu, mortality_rate, initial_weight, initial_number_fishes):
       # Crea una copia del DataFrame y reinicia los índices para asegurar que sean 0, 1, 2...
        historicalGrowth = dataHistoricalTemp.copy().reset_index(drop=True)
        last_historicalGrowth = historicalGrowth['ds'].iloc[-1]
        forescastGrowth = dataForescastTemp.copy().reset_index(drop=True)
        # Convertir las columnas 'ds' a formato datetime
        # Asegurar frecuencia mensual en ambos DataFrames
        historicalGrowth['ds'] = pd.to_datetime(historicalGrowth['ds'])
        historicalGrowth = historicalGrowth.set_index('ds').resample('MS').mean().reset_index()
        forescastGrowth['ds'] = pd.to_datetime(forescastGrowth['ds'])
        forescastGrowth = forescastGrowth.set_index('ds').resample('MS').mean().reset_index()
        # Eliminar del pronóstico cualquier fecha que ya exista en el histórico
        fechas_historicas = set(historicalGrowth['ds'])
        forescastGrowth = forescastGrowth[~forescastGrowth['ds'].isin(fechas_historicas)].reset_index(drop=True)                

        # Fecha inicial
        start_date = historicalGrowth['ds'].iloc[0]
        growth_factor = 0

        # Agregar columnas para almacenar los resultados
        historicalGrowth['function'] = 0.0
        historicalGrowth['growth'] = 0.0
        historicalGrowth['number_fishes'] = 0.0
        historicalGrowth['biomass'] = 0.0

        # Agregar columnas para almacenar los resultados de la predicción
        forescastGrowth['function'] = 0.0
        forescastGrowth['growth'] = 0.0
        forescastGrowth['number_fishes'] = 0.0
        forescastGrowth['biomass'] = 0.0      

        # Calculo del modelo de crecimiento histórico
        for i in range(len(historicalGrowth)):
            # Calcula el tiempo en meses desde la fecha inicial
            delta = pd.to_datetime(historicalGrowth.loc[i, 'ds']) - pd.to_datetime(start_date)
            # se convierte a meses
            # Aproximación de 365.24 / 12 = 30.44 días por mes
            date_inc = delta.total_seconds() / (30.44 * 24 * 3600)
            # Calcula la tasa de crecimiento usando la función de Thyholdt
            f = self._thyholdt_function(date_inc, historicalGrowth.loc[i,'y'], alpha, beta, mu)           
            historicalGrowth.loc[i,'function'] = f
            number_fishes = self._mortality(initial_number_fishes, mortality_rate, date_inc)            
            # Calcula el factor de crecimiento usando el peso inicial y la función de crecimiento y el factor de crecimiento del mes anterior
            growth_factor = growth_factor + (1 + f) * initial_weight
            # Limita el factor de crecimiento al peso máximo asintótico
            if growth_factor > alpha:
                growth_factor = alpha
            historicalGrowth.loc[i,'growth'] = growth_factor
            historicalGrowth.loc[i,'number_fishes'] = number_fishes
            # Calcula el peso total de la biomasa en KG           
            historicalGrowth.loc[i,'biomass'] = growth_factor * number_fishes / 1000.0        

        # Calculo del modelo de crecimiento futuro
        for i in range(len(forescastGrowth)):
            # Calcula el tiempo en meses desde la fecha inicial
            delta = pd.to_datetime(forescastGrowth.loc[i, 'ds']) - pd.to_datetime(start_date)
            # se convierte a meses
            # Aproximación de 365.24 / 12 = 30.44 días por mes
            date_inc = delta.total_seconds() / (30.44 * 24 * 3600)
            
            if i > -1:
                f = self._thyholdt_function(date_inc, forescastGrowth.loc[i,'yhat'], alpha, beta, mu)
                growth_factor = growth_factor + (1 + f) * initial_weight
                number_fishes = self._mortality(initial_number_fishes, mortality_rate, date_inc)
            else:
                f = historicalGrowth.iloc[-1]['function']
                growth_factor = historicalGrowth.iloc[-1]['growth']
                number_fishes = historicalGrowth.iloc[-1]['number_fishes']

            # Limita el factor de crecimiento al peso máximo asintótico
            if growth_factor > alpha:
                growth_factor = alpha

            forescastGrowth.loc[i,'function'] = f 
            forescastGrowth.loc[i,'growth'] = growth_factor
            forescastGrowth.loc[i,'number_fishes'] = number_fishes
            # Calcula el peso total de la biomasa en KG           
            forescastGrowth.loc[i,'biomass'] = growth_factor * number_fishes / 1000.0

        # --- Añadir el punto de unión intermedio SOLO para la gráfica ---
        # Si hay hueco entre el último histórico y el primer forecast, añade el último punto histórico como primer punto del forecast
        if not forescastGrowth.empty:
            last_hist_date = historicalGrowth['ds'].iloc[-1]
            first_fore_date = forescastGrowth['ds'].iloc[0]
            if last_hist_date < first_fore_date:
                # Copia el último punto histórico y lo añade al principio del forecast
                punto_union = historicalGrowth.iloc[[-1]].copy()
                # Si la columna 'yhat' existe en forescastGrowth, iguala 'y' a 'yhat' para el punto de unión
                if 'yhat' in forescastGrowth.columns:
                    punto_union['yhat'] = punto_union['y']
                forescastGrowth = pd.concat([punto_union, forescastGrowth], ignore_index=True)

        return historicalGrowth, forescastGrowth
