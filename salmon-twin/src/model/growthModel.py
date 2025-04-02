"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import numpy as np

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
    def thyholdt_function(self, t, T, alpha, beta, mu):        
        
        result = alpha / (1 + np.exp(-(beta * T) * (t - mu)))        
        
        return result
    
    # Mortandad de los salmones
    # initial_number_fises: número inicial de salmones
    # mortality_percent: tasa de mortandad en tanto por ciento de la población por unidad de tiempo (por mes)
    # t: tiempo en meses
    def mortality(self, initial_number_fises, mortality_percent, t):
        return initial_number_fises * (1 -mortality_percent) ** t      

    # Thyholdt (2014) modelo de crecimiento
    # data: Dataframe con las columnas 'ds' (fecha) e 'y' (temperatura)
    # alpha: peso máximo asintótico en gramos
    # beta: coeficiente de pendiente
    # mu (mi): punto de inflexión en meses
    # mortality_rate: tasa de mortandad en tanto por ciento de la población por unidad de tiempo (por mes)
    # initial_weight: peso inicial del salmon en gramos
    # initial_number_fishes: número inicial de salmones
    def thyholdt_growth(self, data, alpha, beta, mu, mortality_rate, initial_weight, initial_number_fishes):

        # Crea una copia del DataFrame
        data = data.copy()
        # Fecha inicial
        start_date = data['ds'].iloc[0]
        growth_factor = 0       

        # Calculo del modelo de crecimiento
        for i in range(0, len(data)-1):
            # Calcula el tiempo en meses desde la fecha inicial
            date_inc = (data.loc[i,'ds'] - start_date).days / 30
            # Calcula la tasa de crecimiento usando la función de Thyholdt
            f = self.thyholdt_function(date_inc, data.loc[i,'y'], alpha, beta, mu)
            data.loc[i,'function'] = f            
            # Calcula el factor de crecimiento usando el peso inicial y la función de crecimiento y el factor de crecimiento del mes anterior
            growth_factor = growth_factor + (1 + f) * initial_weight
            # Limita el factor de crecimiento al peso máximo asintótico
            if growth_factor > alpha:
                growth_factor = alpha
            data.loc[i,'growth'] = growth_factor
            # Calcula el número de peces
            number_fishes = self.mortality(initial_number_fishes, mortality_rate, date_inc)
            data.loc[i,'number_fishes'] = number_fishes
            # Calcula el peso total de la biomasa            
            data.loc[i,'biomass'] = growth_factor * number_fishes

        return data
    