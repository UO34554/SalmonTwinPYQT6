import pytest
import pandas as pd
import numpy as np
from model.growthModel import GrowthModel

class TestGrowthModel:
    
    @pytest.fixture
    def growth_model(self):
        """Instancia del modelo de crecimiento"""
        return GrowthModel()
    
    @pytest.fixture
    def thyholdt_params(self):
        """Parámetros estándar de la función Thyholdt para todas las pruebas"""
        return {
            'alpha': 7000.0,    # Peso máximo asintótico (gramos)
            'beta': 0.018,      # Tasa de crecimiento
            'mu': 19.0          # Punto de inflexión temporal (mes 19)
        }
    
    @pytest.fixture
    def sample_temperature_data(self):
        """Datos de temperatura histórica"""
        dates = pd.date_range(start='2023-01-01', periods=12, freq='MS')
        return pd.DataFrame({
            'ds': dates,
            'y': [8.5, 8.2, 9.1, 10.5, 12.3, 14.8, 16.2, 15.9, 13.7, 11.4, 9.8, 8.9]
        })
    
    @pytest.fixture
    def sample_forecast_data(self):
        """Datos de temperatura de predicción"""
        dates = pd.date_range(start='2024-01-01', periods=6, freq='MS')
        return pd.DataFrame({
            'ds': dates,
            'yhat': [8.7, 8.4, 9.3, 10.8, 12.6, 15.1]
        })
    
    # Pruebas unitarias ****
    def test_thyholdt_function_basic_calculation(self, growth_model: GrowthModel, thyholdt_params: dict):
        """
        UT-GM-001: Validar cálculo básico de la función Thyholdt
        """
        # Arrange
        t_months = 72  # 6 años tiempo suficiente para ver diferencias (> μ = 19)
        # Parámetros de la función Thyholdt
        temperature = 10.0  # Temperatura en grados Celsius        
        
        # Act
        result = growth_model._thyholdt_function(
            t_months, temperature, 
            thyholdt_params['alpha'], 
            thyholdt_params['beta'], 
            thyholdt_params['mu']
        )
        
        # Assert
        result = result * 1000  # Convertir a gramos
        assert isinstance(result, (int, float))
        assert result > 0
        assert result <= thyholdt_params['alpha']  # No debe exceder el peso máximo asintótico

    def test_thyholdt_function_temperature_sensitivity(self, growth_model: GrowthModel, thyholdt_params: dict):
        """
        UT-GM-002: Verificar sensibilidad a cambios de temperatura
        """
        # Arrange
        t_months = 24   # ✅ Tiempo suficiente para ver diferencias (> μ = 19)        
        temp_low = 5.0
        temp_high = 15.0
        
        # Act
        growth_low = growth_model._thyholdt_function(
            t_months, temp_low, 
            thyholdt_params['alpha'], 
            thyholdt_params['beta'], 
            thyholdt_params['mu']
        )
        growth_high = growth_model._thyholdt_function(
            t_months, temp_high, 
            thyholdt_params['alpha'], 
            thyholdt_params['beta'], 
            thyholdt_params['mu']
        )
        
        # Assert
        assert growth_high > growth_low  # Mayor temperatura = mayor crecimiento

    def test_mortality_calculation(self, growth_model: GrowthModel):
        """
        UT-GM-003: Verificar cálculo de mortandad exponencial
        """
        # Arrange
        initial_fishes = 1000
        mortality_rate = 0.015  # 1.5% mensual
        time_months = 12
        
        # Act
        surviving_fishes = growth_model._mortality(initial_fishes, mortality_rate, time_months)
        
        # Assert
        assert surviving_fishes < initial_fishes  # Debe haber mortalidad
        assert surviving_fishes > 0  # Pero algunos deben sobrevivir
        # Verificar fórmula: N(t) = N₀ × (1 - r)ᵗ modelo discreto
        # donde N₀ es el número inicial de peces, r es la tasa de mortalidad y t es el tiempo en meses
        expected = initial_fishes * (1 - mortality_rate) ** time_months
        assert abs(surviving_fishes - expected) < 1e-10    

    def test_thyholdt_function_zero_values(self, growth_model: GrowthModel):
        """
        UT-GM-006: Verificar manejo de valores cero en parámetros
        """
        # Test con alpha = 0 (peso máximo cero)        
        result_alpha_zero = growth_model._thyholdt_function(12, 10.0, 0.0, 0.02, 17.0)
        # Con alpha=0, el crecimiento debería ser cero
        assert result_alpha_zero == 0, "Resultado con alpha=0 debe ser cero"
        
        # Test con beta = 0 (tasa de crecimiento cero)
        beta = 0.0000002  # Un valor muy pequeño para beta
        result_beta_zero = growth_model._thyholdt_function(12, 10.0, 7000.0, beta, 17.0)
        # Con beta=0, no debería haber crecimiento o debería ser mínimo
        # uso un beta muy pequeño para calcular el resultado esperado
        # calculo el resultado previsto
                       
        expected_growth = (7000.0/1000.0) / (1 + np.exp(-(beta * 10) * (12.0 - 17.0)))
        assert np.isclose(result_beta_zero, expected_growth), "Resultado con beta=0 debe ser igual al crecimiento esperado"
        # Verifico que el resultado sea no negativo
        # Esto es importante porque la función puede devolver valores negativos si no se maneja correctamente
        assert result_beta_zero >= 0, "Resultado con beta=0 debe ser no negativo"
        
        # Test con tiempo = 0
        result_time_zero = growth_model._thyholdt_function(0, 10.0, 7000.0, 0.02, 17.0)
        assert result_time_zero >= 0, "Resultado con tiempo=0 debe ser no negativo"
        
        # Test con temperatura = 0
        temperature = 0.0000002  # Temperatura cero
        result_temp_zero = growth_model._thyholdt_function(12, temperature, 7000.0, 0.02, 17.0)
        expected_growth = (7000.0/1000.0) / (1 + np.exp(temperature))  # Temperatura cero no afecta el crecimiento
        assert np.isclose(result_temp_zero,expected_growth), "Resultado con temperatura=0 debe ser igual al crecimiento esperado"

    def test_thyholdt_function_negative_values(self, growth_model: GrowthModel):
        """
        UT-GM-007: Verificar manejo de valores negativos
        """     

        # Test con tiempo negativo        
        result_negative_time=growth_model._thyholdt_function(-12, 10.0, 7000.0, 0.02, 17.0)
        assert isinstance(result_negative_time, (int, float)), "Debe retornar un número"
        assert(result_negative_time >= 0), "El resultado con tiempo negativo debe ser no negativo"
        
        # Test con alpha negativo
        result_negative_alpha = growth_model._thyholdt_function(12, 10.0, -7000.0, 0.02, 17.0)
        assert isinstance(result_negative_alpha, (int, float)), "Debe retornar un número"
        assert(result_negative_alpha >= 0), "El resultado con alpha negativo debe ser no negativo"
        
        # Test con beta negativo
        result_negative_beta = growth_model._thyholdt_function(12, 10.0, 7000.0, -0.02, 17.0)
        assert isinstance(result_negative_beta, (int, float)), "Debe retornar un número"
        assert(result_negative_beta >= 0), "El resultado con beta negativo debe ser no negativo"
        
        # Test con temperatura negativa (podría ser válida en algunas condiciones)
        result_temp_negative = growth_model._thyholdt_function(12, -5.0, 7000.0, 0.02, 17.0)
        assert isinstance(result_temp_negative, (int, float)), "Debe retornar un número"
        assert not np.isnan(result_temp_negative), "No debe retornar NaN"
        
        # Test con mu negativo
        result_mu_negative = growth_model._thyholdt_function(12, 10.0, 7000.0, 0.02, -17.0)
        assert isinstance(result_mu_negative, (int, float)), "Debe retornar un número"
        assert result_mu_negative >= 0, "El crecimiento debe ser no negativo"

    def test_thyholdt_function_extreme_values(self, growth_model: GrowthModel):
        """
        UT-GM-008: Verificar manejo de valores extremos
        """
        # Test con valores muy grandes
        result_large = growth_model._thyholdt_function(1000, 100.0, 1e6, 1.0, 17.0)
        assert isinstance(result_large, (int, float)), "Debe retornar un número válido"
        assert not np.isinf(result_large), "No debe retornar infinito"
        assert not np.isnan(result_large), "No debe retornar NaN"
        
        # Test con valores muy pequeños
        result_small = growth_model._thyholdt_function(1, 0.1, 10.0, 0.001, 17.0)
        assert isinstance(result_small, (int, float)), "Debe retornar un número válido"
        assert result_small >= 0, "El crecimiento debe ser no negativo"
        
        # Test con beta muy grande (crecimiento muy rápido)
        result_large_beta = growth_model._thyholdt_function(24, 15.0, 7000.0, 10.0, 17.0)
        assert not np.isinf(result_large_beta), "No debe producir overflow"
        assert result_large_beta > 0, "Debe haber crecimiento positivo"

    def test_thyholdt_function_mathematical_edge_cases(self, growth_model: GrowthModel):
        """
        UT-GM-009: Verificar casos extremos matemáticos
        """
        # Test que podría generar overflow en el exponencial
        try:
            result_overflow = growth_model._thyholdt_function(1000, 1000.0, 7000.0, 1.0, 17.0)
            assert not np.isinf(result_overflow), "Debe manejar overflow correctamente"
            assert not np.isnan(result_overflow), "No debe retornar NaN"
        except OverflowError:
            pytest.skip("Overflow esperado para valores extremos")
        
        # Test con tiempo exactamente igual a mu (punto de inflexión)
        result_at_inflection = growth_model._thyholdt_function(17.0, 10.0, 7000.0, 0.02, 17.0)
        assert isinstance(result_at_inflection, (int, float)), "Debe manejar el punto de inflexión"
        assert result_at_inflection >= 0, "Crecimiento debe ser no negativo en punto de inflexión"
        
        # Test con tiempo muy cercano a mu
        result_near_inflection = growth_model._thyholdt_function(16.999, 10.0, 7000.0, 0.02, 17.0)
        assert isinstance(result_near_inflection, (int, float)), "Debe manejar valores cerca del punto de inflexión"

    def test_thyholdt_function_nan_infinity_handling(self, growth_model: GrowthModel):
        """
        UT-GM-010: Verificar manejo de NaN e infinitos
        """
        # Test con NaN en tiempo
        result = growth_model._thyholdt_function(np.nan, 10.0, 7000.0, 0.02, 17.0)
        assert (result==0.0), "Resultado con NaN en tiempo debe ser cero"
        
        # Test con NaN en temperatura
        result = growth_model._thyholdt_function(12, np.nan, 7000.0, 0.02, 17.0)
        assert (result==0.0), "Resultado con NaN en temperatura debe ser cero"
        
        # Test con NaN en alpha
        result = growth_model._thyholdt_function(12, 10.0, np.nan, 0.02, 17.0)
        assert (result==0.0), "Resultado con NaN en alpha debe ser cero"
        
        # Test con infinito en tiempo
        alpha = 7000.0  # Valor de alpha para la prueba
        result = growth_model._thyholdt_function(np.inf, 10.0, alpha, 0.02, 17.0)
        expected_growth = (alpha/1000.0)
        assert (result==expected_growth), "Resultado con infinito en tiempo debe ser igual a alpha"
        
        # Test con infinito en temperatura
        result = growth_model._thyholdt_function(12, np.inf, 7000.0, 0.02, 17.0)
        assert (result==0.0), "Resultado con infinito en temperatura debe ser cero"

    def test_mortality_function_edge_cases(self, growth_model: GrowthModel):
        """
        UT-GM-011: Verificar casos extremos en función de mortalidad
        """
        # Test con mortalidad cero
        result_no_mortality = growth_model._mortality(1000, 0.0, 12)
        assert result_no_mortality == 1000, "Sin mortalidad, población debe mantenerse"
        
        # Test con mortalidad 100% (tasa muy alta)
        result_high_mortality = growth_model._mortality(1000, 1.0, 12)
        assert result_high_mortality == 0.0, "Con mortalidad del 100%, no debe haber peces sobrevivientes"        
        
        # Test con tiempo cero
        result_time_zero = growth_model._mortality(1000, 0.015, 0)
        assert result_time_zero == 1000, "Sin tiempo, no hay mortalidad"
        
        # Test con población inicial cero
        result_no_fish = growth_model._mortality(0, 0.015, 12)
        assert result_no_fish == 0, "Sin población inicial, resultado debe ser cero"
        
        # Test con valores negativos
        result = growth_model._mortality(-100, 0.015, 12)
        assert result == 0.0, "Población negativa debe ser tratada como cero"        
        result = growth_model._mortality(1000, -0.015, 12)
        assert result == 0.0, "Tasa de mortalidad negativa debe ser tratada como cero"        
        result = growth_model._mortality(1000, 0.015, -12)
        assert result == 0.0, "Tiempo negativo debe ser tratado como cero"

    def test_thyholdt_function_input_validation(self, growth_model: GrowthModel):
        """
        UT-GM-012: Verificar validación de tipos de entrada
        """
        # Test con tipos incorrectos
        result = growth_model._thyholdt_function("12", 10.0, 7000.0, 0.02, 17.0)        
        assert result is None, "Debe retornar None para tipos incorrectos"
        result = growth_model._thyholdt_function(12, "10.0", 7000.0, 0.02, 17.0)
        assert result is None, "Debe retornar None para tipos incorrectos"        
        result = growth_model._thyholdt_function(12, 10.0, "7000.0", 0.02, 17.0)
        assert result is None, "Debe retornar None para tipos incorrectos"
        # Test con valores None
        result = growth_model._thyholdt_function(None, None, None, None, None)
        assert result is None, "Debe retornar None para valores None"

    def test_thyholdt_function_boundary_conditions(self, growth_model: GrowthModel):
        """
        UT-GM-013: Verificar condiciones de frontera específicas
        """
        # Test con tiempo ligeramente antes del punto de inflexión
        result_before_mu = growth_model._thyholdt_function(16.9, 10.0, 7000.0, 0.02, 17.0)
        
        # Test con tiempo ligeramente después del punto de inflexión
        result_after_mu = growth_model._thyholdt_function(17.1, 10.0, 7000.0, 0.02, 17.0)
        
        # Ambos deben ser válidos y el posterior debe ser mayor o igual
        assert isinstance(result_before_mu, (int, float))
        assert isinstance(result_after_mu, (int, float))
        assert result_after_mu >= result_before_mu, "Crecimiento debe ser monótono"
        
        # Test con temperatura en el límite de congelación
        result_freezing = growth_model._thyholdt_function(12, 0.1, 7000.0, 0.02, 17.0)
        assert result_freezing >= 0, "Crecimiento debe ser no negativo cerca del punto de congelación"
        
        # Test con temperatura muy alta (límite de supervivencia)
        result_hot = growth_model._thyholdt_function(12, 30.0, 7000.0, 0.02, 17.0)
        assert isinstance(result_hot, (int, float)), "Debe manejar temperaturas altas"

    # Pruebas de integración ****
    def test_thyholdt_growth_complete_model(self, growth_model: GrowthModel, sample_temperature_data: pd.DataFrame, sample_forecast_data: pd.DataFrame,thyholdt_params: dict):
        """
        IT-GM-004: Verificar modelo completo de crecimiento Thyholdt
        """
        # Arrange
        alpha = thyholdt_params['alpha']
        beta = thyholdt_params['beta']
        mu = thyholdt_params['mu']
        mortality_rate = 0.015
        initial_weight = 100.0
        initial_number_fishes = 1000
        
        # Act
        historical_growth, forecast_growth = growth_model.thyholdt_growth(
            sample_temperature_data, sample_forecast_data,
            alpha, beta, mu, mortality_rate, initial_weight, initial_number_fishes
        )
        
        # Assert
        # Verificar estructura de datos
        assert not historical_growth.empty
        assert not forecast_growth.empty
        
        # Verificar columnas requeridas
        required_cols = ['ds', 'function', 'growth', 'number_fishes', 'biomass']
        for col in required_cols:
            assert col in historical_growth.columns
            assert col in forecast_growth.columns
        
        # Verificar valores lógicos
        assert all(historical_growth['growth'] >= 0)
        assert all(historical_growth['biomass'] >= 0)
        assert all(historical_growth['number_fishes'] > 0)
        
        # Verificar tendencia de crecimiento
        assert historical_growth['growth'].iloc[-1] > historical_growth['growth'].iloc[0]

    def test_thyholdt_growth_continuity(self, growth_model: GrowthModel, sample_temperature_data: pd.DataFrame, sample_forecast_data: pd.DataFrame,thyholdt_params: dict):
        """
        IT-GM-005: Verificar continuidad entre datos históricos y de predicción
        """
        # Arrange
        alpha = thyholdt_params['alpha']
        beta = thyholdt_params['beta']
        mu = thyholdt_params['mu']
        mortality_rate = 0.015
        initial_weight = 100.0
        initial_number_fishes = 1000
        
        # Act
        historical_growth, forecast_growth = growth_model.thyholdt_growth(
            sample_temperature_data, sample_forecast_data,
            alpha, beta, mu, mortality_rate, initial_weight, initial_number_fishes
        )
        
        # Assert
        # Verificar que el forecast tiene punto de unión
        if not forecast_growth.empty and not historical_growth.empty:
            last_historical_date = historical_growth['ds'].iloc[-1]
            first_forecast_date = forecast_growth['ds'].iloc[0]
            
            # Si hay gap, debería existir punto de unión
            if last_historical_date < first_forecast_date:
                # El primer punto del forecast debería ser el último histórico
                assert len(forecast_growth) > len(sample_forecast_data)