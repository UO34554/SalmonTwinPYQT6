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
        if not (last_historicalGrowth.day == 1 and last_historicalGrowth.hour == 0 and last_historicalGrowth.minute == 0):
            # Interpolamos la biomasa y otros valores entre el último punto histórico y el primer punto forecast
            # Tomamos el último punto histórico y el primer punto forecast
            last_hist = historicalGrowth.iloc[-1]
            first_fore = forescastGrowth.iloc[0]
            # Interpolación lineal usando numpy para cada valor
            x = np.array([last_hist['ds'].toordinal(), first_fore['ds'].toordinal()])
            union_x = last_historicalGrowth.toordinal()
            union_point = {
                'ds': last_historicalGrowth,
                'y': np.interp(union_x, x, [last_hist['y'], first_fore['yhat']]),
                'function': np.interp(union_x, x, [last_hist['function'], first_fore['function']]),
                'growth': np.interp(union_x, x, [last_hist['growth'], first_fore['growth']]),
                'number_fishes': np.interp(union_x, x, [last_hist['number_fishes'], first_fore['number_fishes']]),
                'biomass': np.interp(union_x, x, [last_hist['biomass'], first_fore['biomass']])
            }
            # Añadir el punto de unión al DataFrame histórico para la gráfica
            historicalGrowth = pd.concat([historicalGrowth, pd.DataFrame([union_point])], ignore_index=True)
            forescastGrowth = pd.concat([pd.DataFrame([union_point]), forescastGrowth], ignore_index=True)

        return historicalGrowth, forescastGrowth
    