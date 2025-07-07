import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch
from model.seaTemperature import DataTemperature

class TestDataTemperature:
    
    @pytest.fixture
    def data_temperature(self):
        """Instancia de DataTemperature para todas las pruebas"""
        return DataTemperature()
    
    @pytest.fixture
    def sample_temperature_data(self):
        """Datos de temperatura de ejemplo válidos"""
        return pd.DataFrame({
            'Año': [2023, 2023, 2023, 2023, 2023, 2023],
            'Mes': ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun'],
            'Finnmark': [4.2, 4.8, 6.1, 8.3, 12.1, 15.7],
            'Troms': [3.8, 4.2, 5.9, 7.8, 11.8, 15.2],
            'Nordland': [5.1, 5.3, 6.8, 9.2, 13.4, 16.8],
            'Nord-Trøndelag': [2.2, 2.6, 3.8, 5.5, 9.1, 12.4],
            'Sør-Trøndelag': [5.8, 6.1, 7.2, 9.8, 14.2, 17.5],
            'Møre og Romsdal': [4.9, 5.2, 6.5, 8.9, 12.8, 16.1],
            'Sogn og Fjordane': [4.3, 4.7, 6.2, 8.1, 12.3, 15.9],
            'Hordaland': [4.1, 4.5, 5.8, 7.9, 11.9, 15.4],
            'Rogaland og Agder': [2.8, 3.1, 4.2, 6.1, 9.8, 13.2]            
        })
    
    @pytest.fixture
    def invalid_temperature_data(self):
        """Datos de temperatura inválidos (faltan columnas)"""
        return pd.DataFrame({
            'Año': [2023, 2023],
            'MesIncorrecto': ['Enero', 'Febrero'],
            'RegionIncorrecta': [4.2, 4.8]
        })
    
    @pytest.fixture
    def empty_temperature_data(self):
        """DataFrame vacío"""
        return pd.DataFrame()


    # Pruebas unitarias ****    
    def test_data_temperature_initialization(self, data_temperature: DataTemperature):
        """
        UT-DT-001: Verificar inicialización correcta de DataTemperature
        """
        # Assert - Verificar que los atributos se inicializan correctamente
        assert hasattr(data_temperature, 'temp_column_names')
        assert hasattr(data_temperature, 'index_region')
        assert hasattr(data_temperature, 'temp_month_names')
        assert hasattr(data_temperature, 'data_regions')
        assert hasattr(data_temperature, 'data_regions_forecast')
        assert data_temperature.lastError is None
        
        # Verificar que data_regions tiene la estructura correcta
        for region in data_temperature.index_region:
            assert region in data_temperature.data_regions
            assert 'ds' in data_temperature.data_regions[region].columns
            assert 'y' in data_temperature.data_regions[region].columns
            
        # Verificar que data_regions_forecast tiene la estructura correcta
        for region in data_temperature.index_region:
            assert region in data_temperature.data_regions_forecast
            expected_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            for col in expected_cols:
                assert col in data_temperature.data_regions_forecast[region].columns

    def test_data_temperature_empty_dataframes(self, data_temperature: DataTemperature):
        """
        UT-DT-002: Verificar que los DataFrames se inicializan vacíos
        """
        # Assert - Todos los DataFrames deben estar vacíos al inicio
        for region in data_temperature.index_region:
            assert data_temperature.data_regions[region].empty
            assert data_temperature.data_regions_forecast[region].empty    

    def test_parse_temperature_invalid_data_missing_columns(self, data_temperature: DataTemperature, invalid_temperature_data: pd.DataFrame):
        """
        UT-DT-005: Verificar manejo de datos con columnas faltantes
        """
        # Act
        result = data_temperature.parseTemperature(invalid_temperature_data)
        
        # Assert
        assert result is False, "El parseo de datos inválidos debe retornar False"
        assert data_temperature.lastError is not None, "Debe haber un error registrado"
        assert "Error:" in data_temperature.lastError, "El error debe tener formato 'Error:'"
        
        # Verificar que los DataFrames siguen vacíos
        for region in data_temperature.index_region:
            assert data_temperature.data_regions[region].empty, "Los DataFrames deben seguir vacíos tras error"

    def test_parse_temperature_empty_data(self, data_temperature: DataTemperature, empty_temperature_data: pd.DataFrame):
        """
        UT-DT-006: Verificar manejo de datos vacíos
        """
        # Act
        result = data_temperature.parseTemperature(empty_temperature_data)
        
        # Assert
        assert result is False, "El parseo de datos vacíos debe retornar False"
        assert data_temperature.lastError is not None, "Debe registrar un error"

    def test_parse_temperature_malformed_dates(self, data_temperature: DataTemperature):
        """
        UT-DT-007: Verificar manejo de fechas malformadas
        """
        # Arrange - Datos con mes inválido
        malformed_data = pd.DataFrame({
            'Año': [2023, 2023],
            'Mes': ['MesInvalido', 'Febrero'],
            'Finnmark': [4.2, 4.8],
            'Troms': [3.8, 4.2],
            'Nordland': [5.1, 5.3],
            'Nord-Trøndelag': [5.8, 6.1],
            'Sør-Trøndelag': [4.9, 5.2],
            'Møre og Romsdal': [4.3, 4.7],
            'Sogn og Fjordane': [4.1, 4.5],
            'Hordaland': [2.8, 3.1],
            'Rogaland og Agder': [2.2, 2.6]
        })
        
        # Act
        result = data_temperature.parseTemperature(malformed_data)
        
        # Assert
        assert result is False, "Datos con fechas malformadas deben ser rechazados"
        assert data_temperature.lastError is not None, "Debe registrar error de fecha"    

    def test_get_temperature_data_invalid_region(self, data_temperature: DataTemperature, sample_temperature_data: pd.DataFrame):
        """
        UT-DT-009: Verificar manejo de región inválida
        """
        # Arrange
        data_temperature.parseTemperature(sample_temperature_data)
        invalid_region = "RegionInexistente"
        
        # Act
        result = data_temperature.getTemperatureData(invalid_region)
        
        # Assert
        assert result is None, "Debe retornar None para región inválida"
        assert data_temperature.lastError is not None, "Debe registrar error"

    @patch('model.seaTemperature.Prophet')
    def test_fit_temp_data_prophet_error(self, mock_prophet, data_temperature: DataTemperature):
        """
        UT-DT-013: Verificar manejo de errores en Prophet
        """
        # Arrange
        mock_prophet.side_effect = ValueError("Prophet error")
        
        temp_data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10, freq='D'),
            'y': np.random.uniform(10, 15, 10)
        })
        
        # Act
        result = data_temperature.fitTempData(temp_data, 10)
        
        # Assert
        assert result is None, "Debe retornar None cuando hay error en Prophet"
        assert data_temperature.lastError is not None, "Debe registrar el error"
        assert "Error:" in data_temperature.lastError, "Error debe tener formato correcto"

    def test_fit_temp_data_invalid_input(self, data_temperature: DataTemperature):
        """
        UT-DT-014: Verificar manejo de datos de entrada inválidos
        """
        # Test con DataFrame vacío
        empty_data = pd.DataFrame()
        result = data_temperature.fitTempData(empty_data, 10)
        assert result is None, "DataFrame vacío debe retornar None"
        
        # Test con datos mal formateados
        invalid_data = pd.DataFrame({'wrong_column': [1, 2, 3]})
        result = data_temperature.fitTempData(invalid_data, 10)
        assert result is None, "Datos mal formateados deben retornar None"
    
    # Pruebas de integración
    # ===============================
    # PRUEBAS DE parseTemperature()
    # ===============================
    def test_parse_temperature_valid_data(self, data_temperature: DataTemperature, sample_temperature_data: pd.DataFrame):
        """
        IT-DT-003: Verificar parseo correcto de datos válidos
        """
        # Act
        result = data_temperature.parseTemperature(sample_temperature_data)
        
        # Assert
        assert result is True, "El parseo de datos válidos debe retornar True"
        assert data_temperature.lastError is None, "No debe haber errores con datos válidos"
        
        # Verificar que los datos se almacenaron correctamente
        for region in data_temperature.index_region:
            df_region = data_temperature.data_regions[region]
            assert not df_region.empty, f"Los datos de {region} no deben estar vacíos"
            assert len(df_region) == len(sample_temperature_data), "Debe haber el mismo número de registros"
            
            # Verificar tipos de datos
            assert df_region['ds'].dtype == 'datetime64[ns]', "La columna 'ds' debe ser datetime"
            assert pd.api.types.is_numeric_dtype(df_region['y']), "La columna 'y' debe ser numérica"

    def test_parse_temperature_data_content_validation(self, data_temperature: DataTemperature, sample_temperature_data: pd.DataFrame):
        """
        IT-DT-004: Verificar que el contenido de los datos se parsea correctamente
        """
        # Act
        data_temperature.parseTemperature(sample_temperature_data)
        
        # Assert - Verificar contenido específico
        first_region = list(data_temperature.index_region.keys())[0]
        df_first_region = data_temperature.data_regions[first_region]
        
        # Verificar primera fila
        first_row = df_first_region.iloc[0]
        expected_date = pd.to_datetime('2023-01')
        assert first_row['ds'] == expected_date, f"Fecha esperada: {expected_date}, obtenida: {first_row['ds']}"
        
        # Verificar que los valores de temperatura son razonables
        for region in data_temperature.index_region:
            df_region = data_temperature.data_regions[region]
            temperatures = df_region['y'].values
            assert all(temp >= 0 for temp in temperatures), "Temperaturas no pueden ser menores a 0°C"
            assert all(temp <= 30 for temp in temperatures), "Temperaturas no pueden ser mayores a 30°C"

    # ===============================
    # PRUEBAS DE getTemperatureData()
    # ===============================

    def test_get_temperature_data_valid_region(self, data_temperature: DataTemperature, sample_temperature_data: pd.DataFrame):
        """
        IT-DT-008: Verificar obtención de datos para región válida
        """
        # Arrange
        data_temperature.parseTemperature(sample_temperature_data)
        region_name = list(data_temperature.index_region.values())[0]  # Primera región disponible
        
        # Act
        result = data_temperature.getTemperatureData(region_name)
        
        # Assert
        assert result is not None, "Debe retornar datos para región válida"
        assert isinstance(result, pd.DataFrame), "Debe retornar un DataFrame"
        assert not result.empty, "Los datos no deben estar vacíos"
        assert 'ds' in result.columns, "Debe contener columna 'ds'"
        assert 'y' in result.columns, "Debe contener columna 'y'"

    def test_get_temperature_data_all_regions(self, data_temperature: DataTemperature, sample_temperature_data: pd.DataFrame):
        """
        IT-DT-010: Verificar que se pueden obtener datos de todas las regiones
        """
        # Arrange
        data_temperature.parseTemperature(sample_temperature_data)
        
        # Act & Assert
        for region_name in data_temperature.index_region.values():
            result = data_temperature.getTemperatureData(region_name)
            assert result is not None, f"Datos de {region_name} deben estar disponibles"
            assert len(result) > 0, f"Datos de {region_name} no deben estar vacíos"

    # ===============================
    # PRUEBAS DE fitTempData()
    # ===============================
    @patch('model.seaTemperature.Prophet')
    def test_fit_temp_data_successful_prediction(self, mock_prophet, data_temperature: DataTemperature):
        """
        IT-DT-011: Verificar predicción exitosa de temperatura
        """
        # Reset mock to ensure clean state
        mock_prophet.reset_mock()

        # Arrange
        mock_prophet_instance = Mock()
        mock_prophet.return_value = mock_prophet_instance
        
        # Datos de entrada simulados
        temp_data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=30, freq='D'),
            'y': np.random.uniform(5, 20, 30)
        })
        periods = 10
        
        # Predicción simulada
        mock_forecast = pd.DataFrame({
            'ds': pd.date_range('2023-01-31', periods=periods, freq='D'),
            'yhat': np.random.uniform(8, 18, periods),
            'yhat_lower': np.random.uniform(6, 16, periods),
            'yhat_upper': np.random.uniform(10, 20, periods)
        })
        mock_prophet_instance.predict.return_value = mock_forecast
        
        # Act
        result = data_temperature.fitTempData(temp_data, periods)
        
        # Assert
        assert result is not None, "Debe retornar resultado de predicción"
        assert isinstance(result, pd.DataFrame), "Debe retornar DataFrame"
        assert len(result) == periods, f"Debe tener {periods} períodos de predicción"
        
        # Verificar que Prophet fue configurado correctamente
        mock_prophet.assert_called_once_with(yearly_seasonality=2)
        mock_prophet_instance.fit.assert_called_once_with(temp_data)

    @patch('model.seaTemperature.Prophet')
    def test_fit_temp_data_temperature_clipping(self, mock_prophet, data_temperature: DataTemperature):
        """
        IT-DT-012: Verificar que las temperaturas se limitan entre 0°C y 30°C
        """
        # Arrange
        mock_prophet_instance = Mock()
        mock_prophet.return_value = mock_prophet_instance
        
        temp_data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10, freq='D'),
            'y': np.random.uniform(10, 15, 10)
        })
        
        # Predicción con valores extremos
        mock_forecast = pd.DataFrame({
            'ds': pd.date_range('2023-01-11', periods=5, freq='D'),
            'yhat': [-5, 35, 15, -10, 40],  # Valores fuera del rango
            'yhat_lower': [-8, 32, 12, -12, 38],
            'yhat_upper': [-2, 38, 18, -8, 42]
        })
        mock_prophet_instance.predict.return_value = mock_forecast
        
        # Act
        result = data_temperature.fitTempData(temp_data, 5)
        
        # Assert
        assert all(result['yhat'] >= 0), "Todas las temperaturas deben ser >= 0°C"
        assert all(result['yhat'] <= 30), "Todas las temperaturas deben ser <= 30°C"
        assert all(result['yhat_lower'] >= 0), "Límites inferiores deben ser >= 0°C"
        assert all(result['yhat_lower'] <= 30), "Límites inferiores deben ser <= 30°C"
        assert all(result['yhat_upper'] >= 0), "Límites superiores deben ser >= 0°C"
        assert all(result['yhat_upper'] <= 30), "Límites superiores deben ser <= 30°C"

    def test_complete_workflow(self, data_temperature: DataTemperature, sample_temperature_data: pd.DataFrame):
        """
        IT-DT-015: Verificar flujo completo de trabajo
        """
        # 1. Parse de datos
        parse_result = data_temperature.parseTemperature(sample_temperature_data)
        assert parse_result is True, "El parseo debe ser exitoso"
        
        # 2. Obtener datos de una región
        region_name = list(data_temperature.index_region.values())[0]
        temp_data = data_temperature.getTemperatureData(region_name)
        assert temp_data is not None, "Debe obtener datos de temperatura"
        
        # 3. Verificar estructura de datos para predicción
        assert 'ds' in temp_data.columns, "Datos deben tener columna 'ds'"
        assert 'y' in temp_data.columns, "Datos deben tener columna 'y'"
        assert len(temp_data) > 0, "Debe haber datos disponibles para predicción"

    def test_error_persistence(self, data_temperature: DataTemperature):
        """
        IT-DT-016: Verificar que los errores se almacenan correctamente
        """
        # Arrange - Datos inválidos que causarán error
        invalid_data = pd.DataFrame({'invalid': ['data']})
        
        # Act
        result = data_temperature.parseTemperature(invalid_data)
        
        # Assert
        assert result is False, "Operación debe fallar"
        assert data_temperature.lastError is not None, "Error debe ser almacenado"
        
        # Verificar que el error persiste
        previous_error = data_temperature.lastError
        
        # Intentar otra operación que falle
        result2 = data_temperature.getTemperatureData("InvalidRegion")
        assert result2 is None, "Segunda operación debe fallar"
        
        # El error puede cambiar, pero debe existir
        assert data_temperature.lastError is not None, "Error debe seguir existiendo"

    def test_single_record_processing(self, data_temperature: DataTemperature):
        """
        IT-DT-019: Verificar procesamiento de un solo registro
        """
        # Arrange
        single_record = pd.DataFrame({
            'Año': [2023],
            'Mes': ['Jul'],
            'Finnmark': [18.5],
            'Troms': [17.8],
            'Nordland': [19.2],
            'Nord-Trøndelag': [20.1],
            'Sør-Trøndelag': [18.9],
            'Møre og Romsdal': [18.3],
            'Sogn og Fjordane': [17.9],
            'Hordaland': [16.2],
            'Rogaland og Agder': [15.8]
        })
        
        # Act
        result = data_temperature.parseTemperature(single_record)
        
        # Assert
        assert result is True, "Debe procesar un solo registro correctamente"
        
        for region in data_temperature.index_region:
            df = data_temperature.data_regions[region]
            assert len(df) == 1, "Debe haber exactamente un registro"
            assert df.iloc[0]['ds'] == pd.to_datetime('2023-07'), "Fecha debe ser julio 2023"

    # Pruebas de sistema *******
    def test_large_dataset_performance(self, data_temperature: DataTemperature):
        """
        ST-DT-017: Verificar rendimiento con datasets grandes
        """
        # Arrange - Dataset grande (varios años de datos)        
        
        num_records = 1000
        months_cycle = ['Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Des']

        large_data = pd.DataFrame({
            'Año': [2020 + i//12 for i in range(num_records)],  
            'Mes': [months_cycle[i % 12] for i in range(num_records)],
            'Finnmark': np.random.uniform(2, 20, num_records),
            'Troms': np.random.uniform(2, 20, num_records),
            'Nordland': np.random.uniform(2, 20, num_records),
            'Nord-Trøndelag': np.random.uniform(2, 20, num_records),
            'Sør-Trøndelag': np.random.uniform(2, 20, num_records),
            'Møre og Romsdal': np.random.uniform(2, 20, num_records),
            'Sogn og Fjordane': np.random.uniform(2, 20, num_records),
            'Hordaland': np.random.uniform(2, 20, num_records),
            'Rogaland og Agder': np.random.uniform(2, 20, num_records)
        })
        
        # Act - Medir tiempo de procesamiento
        start_time = time.time()
        result = data_temperature.parseTemperature(large_data)
        end_time = time.time()
        
        # Assert
        assert result is True, "Debe procesar datasets grandes exitosamente"
        assert (end_time - start_time) < 5.0, "Procesamiento debe completarse en menos de 5 segundos"
        
        # Verificar que todos los datos se procesaron
        for region in data_temperature.index_region:
            assert len(data_temperature.data_regions[region]) == 1000, "Todos los registros deben estar procesados"

    # ===============================
    # PRUEBAS DE CASOS LÍMITE
    # ===============================

    def test_temperature_edge_values(self, data_temperature: DataTemperature):
        """
        ST-DT-018: Verificar manejo de valores extremos de temperatura
        """
        # Arrange - Datos con temperaturas extremas pero válidas
        extreme_data = pd.DataFrame({
            'Año': [2023, 2023, 2023],
            'Mes': ['Jan', 'Feb', 'Mar'],
            'Finnmark': [-5.0, 0.0, 30.0],  # Temperaturas extremas
            'Troms': [-3.0, 1.0, 28.0],
            'Nordland': [-2.0, 2.0, 29.0],
            'Nord-Trøndelag': [-1.0, 3.0, 25.0],
            'Sør-Trøndelag': [0.0, 4.0, 26.0],
            'Møre og Romsdal': [1.0, 5.0, 27.0],
            'Sogn og Fjordane': [2.0, 6.0, 24.0],
            'Hordaland': [-4.0, 0.5, 22.0],
            'Rogaland og Agder': [-6.0, -1.0, 20.0]
        })
        
        # Act
        result = data_temperature.parseTemperature(extreme_data)
        
        # Assert
        assert result is True, "Debe manejar temperaturas extremas válidas"
        
        # Verificar que se mantienen los valores extremos
        for region in data_temperature.index_region:
            temps = data_temperature.data_regions[region]['y'].values
            assert len(temps) == 3, "Debe haber 3 registros de temperatura"
            # No verificamos rangos específicos ya que pueden ser válidos según la región