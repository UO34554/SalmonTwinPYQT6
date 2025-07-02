import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from model.growthModel import GrowthModel

class TestGrowthModel:
    
    @pytest.fixture
    def growth_model(self):
        """Instancia del modelo de crecimiento"""
        return GrowthModel()
    
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

    def test_thyholdt_function_basic_calculation(self, growth_model):
        """
        UT-GM-001: Validar cálculo básico de la función Thyholdt
        """
        # Arrange
        t_months = 12
        temperature = 10.0
        alpha = 7000.0
        beta = 0.02
        mu = 17.0
        
        # Act
        result = growth_model._thyholdt_function(t_months, temperature, alpha, beta, mu)
        
        # Assert
        assert isinstance(result, (int, float))
        assert result > 0
        assert result <= alpha  # No debe exceder el peso máximo asintótico

    def test_thyholdt_function_temperature_sensitivity(self, growth_model):
        """
        UT-GM-002: Verificar sensibilidad a cambios de temperatura
        """
        # Arrange
        t_months = 12
        alpha = 7000.0
        beta = 0.02
        mu = 17.0
        temp_low = 5.0
        temp_high = 15.0
        
        # Act
        growth_low = growth_model._thyholdt_function(t_months, temp_low, alpha, beta, mu)
        growth_high = growth_model._thyholdt_function(t_months, temp_high, alpha, beta, mu)
        
        # Assert
        assert growth_high > growth_low  # Mayor temperatura = mayor crecimiento

    def test_mortality_calculation(self, growth_model):
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
        # Verificar fórmula: N(t) = N0 * exp(-r*t)
        expected = initial_fishes * np.exp(-mortality_rate * time_months)
        assert abs(surviving_fishes - expected) < 1e-10

    def test_thyholdt_growth_complete_model(self, growth_model, sample_temperature_data, sample_forecast_data):
        """
        UT-GM-004: Verificar modelo completo de crecimiento Thyholdt
        """
        # Arrange
        alpha = 7000.0
        beta = 0.02
        mu = 17.0
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

    def test_thyholdt_growth_continuity(self, growth_model, sample_temperature_data, sample_forecast_data):
        """
        UT-GM-005: Verificar continuidad entre datos históricos y de predicción
        """
        # Arrange
        alpha = 7000.0
        beta = 0.02
        mu = 17.0
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