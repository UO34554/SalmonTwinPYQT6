"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from PySide6.QtWidgets import QLabel, QDialog, QFileDialog, QGraphicsView, QGraphicsScene, QWidget, QSlider, QGridLayout, QVBoxLayout, QPushButton, QProgressDialog, QTextEdit
from PySide6.QtCore import Qt, QTimer, QThread, QMutexLocker, QMutex, Signal
from PySide6.QtGui import QPen, QBrush, QColor, QFont, QTextCursor
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import pandas as pd
import random
import config as cfg
import locale
import math
from model.seaTemperature import DataTemperature
from model.growthModel import GrowthModel
from model.priceModel import DataPrice
from utility.utility import OptionsDialog, auxTools, DataLoader
from datetime import datetime, timedelta

# Controlodador de la vista de dashboard
class dashBoardController:
    def __init__(self,view,raftController):
        self._view = view        
        self.lastError = None
        self.raftCon = raftController
        self.tempModel = DataTemperature()
        self.priceModel = DataPrice()
        self.dataLoader = DataLoader()
        self.growthModel = GrowthModel()       
        self.lastRaftName = None

        # Inicializar referencias a sliders (se crearán al dibujar la balsa)
        self.dateSliderCurrent = None
        self.dateSliderForecast = None

        # --- Inicialización de la vista ---        
        # Configura el idioma a español (España) para las fechas
        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')        
        # Crear un QLabel para el mensaje de estado
        self.label_estado = QLabel()
        self._view.statusbar.addPermanentWidget(self.label_estado)
        # Cargar las balsas marinas
        self.load_rafts_from_controller()
        # Inicializar los datos del gráfico 3D
        self.fish_items = []
        self.fish_count = 50
        self.timer = QTimer()
        # variable para rastrear la conexión del temporizador
        self.timer_connected = False

        # --- Conectar señales de la vista con manejadores de eventos ---
        self._view.actionConfigurar.triggered.connect(self.on_raft_config)
        self._view.actionVer.triggered.connect(self.on_raft_view)
        self._view.actionCSV.triggered.connect(self.on_temperature_load_csv)
        self._view.actionPredecir.triggered.connect(self.on_temperature_predict)
        self._view.actionCSVprecio.triggered.connect(self.on_price_load_csv)
        self._view.actionPredecirPrecio.triggered.connect(self.on_price_predict)
        self._view.actionBuscarPredictor.triggered.connect(self.on_search_price_predictor)
        self._view.actionPredecirCrecimiento.triggered.connect(self.on_growth_predict)      
    
    # --- Eventos de la vista ---
    def show(self):
        # Mostrar mensaje permanente de información de las balsas marinas        
        self.label_estado.setText(cfg.RAFTS_LOADED_MESSAGE.format(count=self.raftCon.count_rafts()))
        self._view.show()

    def on_growth_predict(self):
        raft = self.choice_raft_list_dialog()
        if raft is None:
            return
        
        # Agregar datos al gráfico si existen
        df_temperature = raft.getTemperature()
        if df_temperature is None or df_temperature.empty:            
            auxTools.show_error_message(cfg.DASHBOARD_NO_TEMP_DATA_ERROR)
            return
        else:            
            # Convertir la columna 'ds' a formato datetime si no está ya
            df_temperature['ds'] = pd.to_datetime(df_temperature['ds'], errors='coerce')
            # Eliminar valores NaT antes de filtrar
            df_temperature = df_temperature.dropna(subset=['ds'])
            # Filtrar los datos de temperatura con la fecha inicial y la fecha actual
            df_temperature = df_temperature[(df_temperature['ds'].dt.date >= raft.getStartDate())]
            if df_temperature is None or df_temperature.empty:
                auxTools.show_error_message(cfg.DASHBOARD_NO_TEMP_DATA_ERROR)
                return
            # Implementar la predicción del crecimiento del salmón según indica el slider
            if self.dateSliderCurrent is None:
                sliderValue = 0
            else:
                sliderValue = self.dateSliderCurrent.value()            
            # Guardar el valor del slider en la balsa como porcentaje
            raft.setPerCentage(sliderValue)
            # Calcular fecha de inicio de predicción
            # El porcentaje se divide por 1000 para ajustarlo al modelo de crecimiento
            percent = raft.getPerCentage()
            delta_days = (raft.getEndDate() - raft.getStartDate()).days
            days = int(delta_days * percent / 1000)
            fecha_actual = raft.getStartDate() + timedelta(days)
            df_temperature = self._filter_and_interpolate_temperature_data(df_temperature, fecha_actual)
            if df_temperature is None or df_temperature.empty:
                auxTools.show_error_message(cfg.DASHBOARD_NO_TEMP_DATA_ERROR)
                return
            
            # Parámetros del modelo Thyholdt (estos valores pueden ser ajustados según tus necesidades)
            alpha = 7000.0                                  # Peso máximo asintótico en gramos (7kg)
            beta = 0.02004161                               # Coeficiente de pendiente
            mu = 17.0                                       # Punto de inflexión en meses
            mortality_rate = 0.015                          # Tasa mensual de mortandad (1,5%)
            initial_weight = 100.0                          # Peso inicial del salmón en gramos (100g)            
            initial_number_fishes = raft.getNumberFishes()  # Cantidad inicial de peces
            
            # Aplicar el modelo de crecimiento de Thyholdt devuelve el peso en KG
            df_forecast_temperature = raft.getTemperatureForecast()
            df_forecast_temperature['ds'] = pd.to_datetime(df_forecast_temperature['ds'], errors='coerce')
            df_forecast_temperature = df_forecast_temperature.dropna(subset=['ds'])           
            df_forecast_temperature = df_forecast_temperature[(df_forecast_temperature['ds'].dt.date >= fecha_actual)]
            if df_forecast_temperature is None or df_forecast_temperature.empty:
                # Mostrar un mensaje de error temporal
                auxTools.show_error_message(cfg.DASHBOARD_NO_TEMP_FORECAST_DATA_ERROR)
                return            
            # Aplicar el modelo de crecimiento de Thyholdt
            growth_data, growth_data_forescast = self.growthModel.thyholdt_growth(df_temperature, df_forecast_temperature,
                                                    alpha, 
                                                    beta, 
                                                    mu, 
                                                    mortality_rate, 
                                                    initial_weight, 
                                                    initial_number_fishes)
            
            raft.setGrowth(growth_data)
            raft.setGrowthForecast(growth_data_forescast)
            # Actualizar la balsa en la lista de balsas
            if self.raftCon.update_rafts_biomass(raft):    
                auxTools.show_info_dialog(cfg.DASHBOARD_PREDICT_GROWTH_SUCCESS)
            else:
                auxTools.show_error_message(cfg.DASHBOARD_PREDICT_GROWTH_ERROR)

    def on_raft_view(self):        
        self.load_rafts_from_controller()        
        data = self.raftCon.get_name_rafts()
        # Mostrar un diálogo para seleccionar una balsa
        option = self.aux_list_dialog(data, title=cfg.DASHBOARD_LIST_TITLE_RAFT, message=cfg.DASHBOARD_SELECT_RAFT_MESSAGE)
        if option:
            # Actualizar el mensaje de estado permanente
            self.label_estado.setText(cfg.RAFTS_LOADED_MESSAGE.format(count=self.raftCon.count_rafts()))
            self._draw_raft(option)
        else:
            # Mostrar mensaje de error temporal
            self._view.statusbar.showMessage(cfg.DASHBOARD_SELECT_RAFT_ERORR_MESSAGE)            

    def on_raft_config(self):
        self.raftCon.show()
        # Actualizar el mensaje de estado permanente
        self.label_estado.setText(cfg.RAFTS_LOADED_MESSAGE.format(count=self.raftCon.count_rafts()))

    def on_temperature_load_csv(self):
        # Cargar la temperatura
        options = QFileDialog.Options()
        file_name = QFileDialog.getOpenFileName(
            None,
            cfg.DASHBOARD_LOAD_TEMP_FILE_MSG,
            "",
            "CSV Files (*.csv)",
            options=options
            )
        if self.load_dataTemperature_from_file("csv", file_name[0], ';'):
            if self._save_raft_temperature():             

                auxTools.show_info_dialog(cfg.DASHBOARD_LOAD_TEMP_FILE_SUCCESS)
        else:
            auxTools.show_error_message(cfg.DASHBOARD_LOAD_TEMP_FILE_ERROR)

    def on_price_load_csv(self):
        # Cargar el precio
        options = QFileDialog.Options()
        file_name = QFileDialog.getOpenFileName(
            None,
            cfg.DASHBOARD_LOAD_PRICE_FILE_MSG,
            "",
            "CSV Files (*.csv)",
            options=options
            )
        if self.load_dataPrice_from_file("csv", file_name[0], ';'):
            if self._save_salmon_price():
                auxTools.show_info_dialog(cfg.DASHBOARD_LOAD_PRICE_FILE_SUCCESS)            
        else:
            auxTools.show_error_message(cfg.DASHBOARD_LOAD_PRICE_FILE_ERROR)

    def choice_raft_list_dialog(self):
        self.load_rafts_from_controller()        
        data = self.raftCon.get_name_rafts()
        option = self.aux_list_dialog(data,title=cfg.DASHBOARD_LIST_TITLE_RAFT, message=cfg.DASHBOARD_SELECT_RAFT_MESSAGE)
        # Mostrar un diálogo para seleccionar una balsa
        if not option:
            auxTools.show_error_message(cfg.DASHBOARD_SELECT_RAFT_ERORR_MESSAGE)
            return None       
        else:
            # Usar la balsa actualmente seleccionada
            raft = self.raftCon.get_raft_by_name(option)
            return raft

    # Muestra un diálogo para seleccionar un estimador de precios    
    def _select_top_estimator_dialog(self):
        # Cargar los mejores estimadores de precios
        items = self.priceModel.get_saved_top_estimators()
        if len(items) == 0:
            auxTools.show_error_message(cfg.DASHBOARD_PREDICT_GET_ESTIMATORS)
            auxTools.show_error_message(cfg.DASHBOARD_SELECT_ESTIMATOR_MESSAGE)
            return None
        
        descs = []
        for i, est in enumerate(items):
            desc = f"#{i+1} | Score: {est['score']:.4f} | MAE: {est['mae']:.3f} | RMSE: {est['rmse']:.3f} | DIR: {est['dir_acc']} | Stats: {est['stats']} | Windows: {est['windows']}"
            descs.append(desc)

        option = self.aux_list_dialog(descs, title=cfg.DASHBOARD_LIST_TITLE_ESTIMATOR, message=cfg.DASHBOARD_LIST_ESTIMATOR_MESSAGE)
        # Mostrar un diálogo para seleccionar una balsa
        if not option:
            auxTools.show_error_message(cfg.DASHBOARD_SELECT_ESTIMATOR_MESSAGE)
            return None       
        else:
            # Usar el estimador seleccionado
            idx = descs.index(option)
            selected_estimator = items[idx]            
            return selected_estimator

    def on_temperature_predict(self):
        raft = self.choice_raft_list_dialog()
        if raft is None:
            return
        
        # Agregar datos al gráfico si existen
        dataTemp = raft.getTemperature()
        if dataTemp.empty:
            # Mostrar un mensaje de error temporal
            self._view.statusbar.showMessage(cfg.DASHBOARD_NO_TEMP_DATA_ERROR)
            return
        else:
            # Implementar la predicción de la temperatura del mar según indica el slider
            if self.dateSliderCurrent is None:
                sliderValue = 0
            else:
                sliderValue = self.dateSliderCurrent.value()            
            
            # Guardar el valor del slider en la balsa como porcentaje
            raft.setPerCentage(sliderValue)
            perCent = raft.getPerCentage()/1000

            # Calcular fecha de inicio de predicción
            delta_days = (raft.getEndDate() - raft.getStartDate()).days
            forescast_start_date = raft.getStartDate() + timedelta(delta_days * perCent)

            # --- AÑADIR DATOS HISTÓRICOS PREVIOS AL ENTRENAMIENTO ---
            # Fechas de la balsa
            start_date = raft.getStartDate()
            end_date = raft.getEndDate()
            # Calcular la fecha de inicio del rango previo
            dataTemp['ds'] = pd.to_datetime(dataTemp['ds'], errors='coerce')
            prev_start_date = dataTemp['ds'].min().date()  # Fecha mínima de los datos de temperatura
            # Filtrar históricos previos (mismo rango de días que la balsa, justo antes)
            dataTemp['ds'] = pd.to_datetime(dataTemp['ds'], errors='coerce')
            dataTemp = dataTemp.dropna(subset=['ds'])
            df_hist = dataTemp[(dataTemp['ds'].dt.date >= prev_start_date) & (dataTemp['ds'].dt.date < start_date)]
            df_hist = df_hist.sort_values('ds')
            # Filtrar históricos de la balsa
            df_balsa = dataTemp[(dataTemp['ds'].dt.date >= start_date) & (dataTemp['ds'].dt.date <= end_date)]
            # Unir el último punto de df_hist con el primero de df_balsa si ambos existen
            if not df_hist.empty and not df_balsa.empty:
                first_balsa = df_balsa.iloc[[0]]
                df_hist = pd.concat([df_hist, first_balsa], ignore_index=True)
            # Concatenar históricos previos y de la balsa
            if not df_hist.empty:
                dataTemp = pd.concat([df_hist, df_balsa], ignore_index=True)
            else:
                dataTemp = df_balsa

            # Verificar que tenemos suficientes datos para la predicción
            # Al menos 6 meses de datos para el modelo temperatura
            # La temperatura tiene una frecuencia mensual, por lo que 6 meses son 6 puntos
            if len(dataTemp)< 6:
                auxTools.show_error_message(cfg.DASHBOARD_NO_TEMP_PERIOD_ERROR)
                return

            # Dias de predicción
            forescastDays = (raft.getEndDate() - forescast_start_date).days

            dataTemp = self._filter_and_interpolate_temperature_data(dataTemp, forescast_start_date)
            if dataTemp is None:
                auxTools.show_error_message(cfg.DASHBOARD_TEMP_FILTER_ERROR)
                return         
            data_forecast = self.tempModel.fitTempData(dataTemp,forescastDays)
            if data_forecast is not None:
                raft.setTemperatureForecast(data_forecast)
                # Actualizar la balsa en la lista de balsas
                if self.raftCon.update_rafts_temp_forecast(raft):    
                    auxTools.show_info_dialog(cfg.DASHBOARD_PREDICT_TEMP_SUCCESS)
                else:
                    auxTools.show_error_message(cfg.DASHBOARD_PREDICT_TEMP_ERROR)
            else:
                auxTools.show_error_message(cfg.DASHBOARD_PREDICT_TEMP_ERROR.format(error=self.tempModel.lastError))        

    def on_price_predict(self):
        raft = self.choice_raft_list_dialog()
        if raft is None:
            return
        
        # Obtener las fechas inicial y final de la balsa
        start_date = raft.getStartDate()
        end_date = raft.getEndDate()        

        # Obtener todos los datos de precios
        price_data = raft.getPriceData()
        if price_data is None or price_data.empty:
            auxTools.show_error_message(cfg.DASHBOARD_PREDICT_PRICE_ERROR.format(error="No hay datos de precios"))
            return
        
        # Convertir la columna 'timestamp' a formato datetime si no está ya
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], errors='coerce')
        price_data = price_data.dropna(subset=['timestamp'])

        # --- Añadir datos históricos previos al entrenamiento ---
        prev_start_date = price_data['timestamp'].min().date()
        df_hist = price_data[(price_data['timestamp'].dt.date >= prev_start_date) & (price_data['timestamp'].dt.date < start_date)]
        df_hist = df_hist.sort_values('timestamp')
        df_balsa = price_data[(price_data['timestamp'].dt.date >= start_date) & (price_data['timestamp'].dt.date <= end_date)]
        # Unir el último punto de df_hist con el primero de df_balsa si ambos existen
        if not df_hist.empty and not df_balsa.empty:
            first_balsa = df_balsa.iloc[[0]]
            df_hist = pd.concat([df_hist, first_balsa], ignore_index=True)
        # Concatenar históricos previos y de la balsa
        if not df_hist.empty:
            price_data = pd.concat([df_hist, df_balsa], ignore_index=True)
        else:
            price_data = df_balsa

        # Establecer los datos en el modelo
        self.priceModel.setPriceData(price_data)        
                                  
        # Verificar slider
        if self.dateSliderCurrent is None:
            sliderValue = 0
            auxTools.show_error_message(cfg.DASHBOARD_NO_TEMP_PERIOD_ERROR)
            return
        else:
            sliderValue = self.dateSliderCurrent.value()

        # Dias de predicción
        perCent = raft.getPerCentage()/1000
        # Permitir elegir estimador si hay guardados
        estimator = self._select_top_estimator_dialog()
        if estimator is not None:
            adjust = True
        else:
            adjust = False

        # Llamar al método fit_price con las fechas específicas
        if self.priceModel.fit_price(perCent, start_date, end_date, adjust, estimator, prev_start_date=prev_start_date):
            # Guardar los datos de precios en la balsa
            raft.setPerCentage(sliderValue)           
            raft.setPriceForecast(self.priceModel.getPriceDataForecast())
            # Actualizar la balsa en la lista de balsas
            if self.raftCon.update_rafts_price_forecast(raft):
                auxTools.show_info_dialog(cfg.DASHBOARD_PREDICT_PRICE_SUCCESS)
        else:            
            auxTools.show_error_message(cfg.DASHBOARD_PREDICT_PRICE_ERROR.format(error=self.priceModel.lastError))

    def on_search_price_predictor(self):
        raft = self.choice_raft_list_dialog()
        if raft is None:
            return
    
        # Obtener las fechas y configurar datos en el modelo
        start_date = raft.getStartDate()
        end_date = raft.getEndDate()
        
        # --- Añadir datos históricos previos al entrenamiento ---
        price_data = raft.getPriceData()
        if price_data is None or price_data.empty:
            auxTools.show_error_message(cfg.DASHBOARD_PREDICT_PRICE_ERROR.format(error="No hay datos de precios"))
            return
        
        # Convertir la columna 'timestamp' a formato datetime si no está ya
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], errors='coerce')
        price_data = price_data.dropna(subset=['timestamp'])

        # Filtrar históricos previos (mismo rango de días que la balsa, justo antes)
        prev_start_date = price_data['timestamp'].min().date()
        df_hist = price_data[(price_data['timestamp'].dt.date >= prev_start_date) & (price_data['timestamp'].dt.date < start_date)]
        df_hist = df_hist.sort_values('timestamp')
        df_balsa = price_data[(price_data['timestamp'].dt.date >= start_date) & (price_data['timestamp'].dt.date <= end_date)]
        # Unir el último punto de df_hist con el primero de df_balsa si ambos existen
        if not df_hist.empty and not df_balsa.empty:
            first_balsa = df_balsa.iloc[[0]]
            df_hist = pd.concat([df_hist, first_balsa], ignore_index=True)
        # Concatenar históricos previos y de la balsa
        if not df_hist.empty:
            price_data = pd.concat([df_hist, df_balsa], ignore_index=True)
        else:
            price_data = df_balsa

        # Establecer los datos en el modelo
        self.priceModel.setPriceData(price_data)
        
        # Verificar slider
        if self.dateSliderCurrent is None:
            auxTools.show_error_message(cfg.DASBOARD_NO_TEMP_PERIOD_ERROR)
            return
    
        # Calcular porcentaje
        perCent = raft.getPerCentage() / 1000

        # Se fija el número de iteraciones
        num_iterations = 800

        # Crear worker thread
        # Si hay datos históricos previos y de la balsa, usar el rango previo
        if not df_hist.empty and not df_balsa.empty:            
            self.search_worker = PricePredictorSearchWorker(
                self.priceModel, perCent, start_date, end_date, prev_start_date, n_iterations=num_iterations
            )
        else:
            self.search_worker = PricePredictorSearchWorker(
                self.priceModel, perCent, start_date, end_date, None, n_iterations=num_iterations
            )

        # Crear y mostrar diálogo de progreso
        self.search_dialog = PredictorSearchDialog(self._view,self.search_worker)

        # Conectar señales
        self.search_worker.progress_updated.connect(self.search_dialog.update_progress)
        self.search_worker.status_updated.connect(self.search_dialog.update_status)
        self.search_worker.print_message.connect(self.search_dialog.print_mensage)
        self.search_worker.result_found.connect(self.search_dialog.add_result)
        self.search_worker.finished_signal.connect(self._on_search_finished)
        self.search_worker.finished_signal.connect(self.search_dialog.search_finished)        

        # Conectar cancelación al botón integrado
        self.search_dialog.cancel_button.clicked.connect(self.search_worker.stop)
        self.search_dialog.cancel_button.clicked.connect(self.search_dialog.cancel_search)

        # Iniciar worker thread
        self.search_worker.start()

        # Mostrar diálogo
        self.search_dialog.exec()

    # Callback cuando termina la búsqueda
    def _on_search_finished(self, success, message):        
        if success:
            # Actualizar la balsa con los nuevos datos
            raft = self.raftCon.get_raft_by_name(self.lastRaftName)
            if raft:
                raft.setPerCentage(self.dateSliderCurrent.value())
                raft.setPriceForecast(self.priceModel.getPriceDataForecast())
            
            if self.raftCon.update_rafts_price_forecast(raft):
                # Solo mostrar popup para éxito
                auxTools.show_info_dialog("Actualizada la balsa con los nuevos datos")
            else:
                # Solo mostrar popup para errores críticos de actualización
                auxTools.show_error_message("Error actualizando la balsa con los nuevos datos")
        else:
            # NO mostrar popup para errores de búsqueda - ya se muestran en la ventana
            pass
        
        # Limpiar referencias
        self.search_worker = None
        self.search_dialog = None

    # --- Métodos de la lógica de negocio

    """
    Filtrar datos de temperatura por fecha y realizar interpolación lineal
    usando np.interp cuando la fecha de inicio de predicción cae entre dos puntos mensuales
    
    Args:
        dataTemp: DataFrame con datos de temperatura
        forescast_start_date: Fecha de inicio para la predicción
        
    Returns:
        DataFrame filtrado con punto interpolado si es necesario
    """
    def _filter_and_interpolate_temperature_data(self, dataTemp, forescast_start_date):
        try:
             # Crear copia para evitar modificar el original
            filtered_data = dataTemp.copy()        
            # Asegurar que 'ds' esté en formato datetime
            filtered_data['ds'] = pd.to_datetime(filtered_data['ds'], errors='coerce')        
            # Eliminar valores NaT
            filtered_data = filtered_data.dropna(subset=['ds'])        
            # Verificar que tenemos datos después de la limpieza
            if filtered_data.empty:
                raise ValueError("No hay datos válidos después de la conversión de fechas")            
            # Verificar si la fecha exacta existe en los datos
            if hasattr(forescast_start_date, 'date'):          
                forecast_date = forescast_start_date.date()        
            else:
                forecast_date = forescast_start_date
            exact_match = filtered_data[filtered_data['ds'].dt.date == forecast_date]
            forecast_timestamp = pd.Timestamp(forecast_date)
            if not exact_match.empty:
                # Si existe la fecha exacta, usar los datos hasta esa fecha (incluyéndola)
                historical_data = filtered_data[filtered_data['ds'] <= forecast_timestamp]                
                return historical_data
            # Si no existe fecha exacta, necesitamos interpolar
            # Filtrar datos anteriores (estrictamente menores) a la fecha de inicio de predicción
            before_data = filtered_data[filtered_data['ds'] < forecast_timestamp]
            # Filtrar datos posteriores (estrictamente mayores) a la fecha de inicio de predicción
            after_data = filtered_data[filtered_data['ds'] > forecast_timestamp]
            if before_data.empty:
                raise ValueError(f"No hay datos de temperatura antes de {forecast_date}")
            if after_data.empty:
                raise ValueError(f"No hay datos de temperatura después de {forecast_date}")
            # Obtener los puntos para interpolación
            before_point = before_data.iloc[-1]  # Último punto antes
            after_point = after_data.iloc[0]     # Primer punto después
            # Convertir fechas a timestamps numéricos para interpolación
            x_points = np.array([before_point['ds'].timestamp(), after_point['ds'].timestamp()])
            y_points = np.array([before_point['y'], after_point['y']])
            x_target = forecast_timestamp.timestamp()
            # Interpolar el valor de temperatura para la fecha de inicio de predicción
            interpolated_value = np.interp(x_target, x_points, y_points)
            # Crear un DataFrame con el punto interpolado
            interpolated_data = pd.DataFrame({
                'ds': [forescast_start_date],
                'y': [interpolated_value]
            })
            # Concatenar los datos históricos con el punto interpolado
            historical_data = pd.concat([before_data, interpolated_data], ignore_index=True)
            # Asegurar formato uniforme
            historical_data['ds'] = pd.to_datetime(historical_data['ds'])
            return historical_data
        except Exception as e:
            auxTools.show_error_message(cfg.DASHBOARD_TEMP_FILTER_ERROR_MSG.format(error=str(e)))
            return None        

    def _save_raft_temperature(self):
        if self.lastRaftName is None:
           raft = self.choice_raft_list_dialog()
           if raft is None:
            return
        else:
            raft = self.raftCon.get_raft_by_name(self.lastRaftName)
           
        # Guardar los datos de temperatura en la balsa
        raft.setTemperature(self.tempModel.getTemperatureData(raft.getSeaRegion()))
        # Actualizar la balsa en la lista de balsas
        return self.raftCon.update_rafts_temp(raft)            

    # Guardar los datos de precios de salmón en un archivo JSON   
    def _save_salmon_price(self):
        if self.lastRaftName is None:
           raft = self.choice_raft_list_dialog()
           if raft is None:
            return
        else:
            raft = self.raftCon.get_raft_by_name(self.lastRaftName)

        # Guardar los datos de precios en la balsa
        raft.setPrice(self.priceModel.getPriceData())
        # Actualizar la balsa en la lista de balsas
        return self.raftCon.update_rafts_price(raft)        
    
    # Borra todos los widgets del layout central
    def _clear_dashboard(self):
        # Detener el temporizador para evitar actualizaciones después de eliminar widgets
        self.timer.stop()
        # Desconectar específicamente el método que sabemos que se conecta en _create_fish
        if self.timer_connected:
            try:
                self.timer.timeout.disconnect(self._update_fish_positions)
                self.timer_connected = False  # Marcar como desconectado
            except TypeError:
                pass

        # Limpiar las referencias a las líneas verticales antes de borrar widgets
        self._clear_vertical_lines()
    
        # Borrar todos los widgets del layout
        for i in reversed(range(self._view.centralwidget.layout().count())): 
            self._view.centralwidget.layout().itemAt(i).widget().deleteLater()
    
        # Limpiar las referencias a los peces
        self.fish_items = []
        self.fish_positions = []

    def _clear_vertical_lines(self):
        """Limpia las referencias a las líneas verticales para evitar errores"""
        # Limpiar líneas de temperatura
        if hasattr(self, 'date_vline'):
            self.date_vline = None
        if hasattr(self, 'date_vline_forescast'):
            self.date_vline_forescast = None
    
        # Limpiar líneas de crecimiento
        if hasattr(self, 'growth_vline'):
            self.growth_vline = None
        if hasattr(self, 'growth_vline_forescast'):
            self.growth_vline_forescast = None
    
        # Limpiar líneas de precio
        if hasattr(self, 'price_vline'):
            self.price_vline = None
        if hasattr(self, 'price_vline_forescast'):
            self.price_vline_forescast = None

    # Dibujar la balsa seleccionada
    def _draw_raft(self,raftName):
        # Mostrar mensaje temporal
        self._view.statusbar.showMessage(cfg.DASHBOARD_RAFT_SELECTED_MESSAGE.format(raftName))
        self.lastRaftName = raftName
        # Buscar la balsa seleccionada
        raft = self.raftCon.get_raft_by_name(raftName)
        # Limpiar la vista
        self._clear_dashboard()        
        # Dibujar la balsa
        self._draw_infopanel(0,0,raft)
        self._draw_miniDash(0,1,raft,self.raftCon.get_rafts())
        self._draw_graph_temperature(1,0,raft)
        self._draw_growth_model(1,1,raft)
        self._draw_price(2,0,raft)
        self._draw_schematic_3d(2,1,raft)

    #Actualiza los ticks del eje X cuando cambia el rango visible
    def _update_price_axis_ticks(self, range_vals):
        # Obtener el rango visible actual
        min_x, max_x = range_vals[0]
        x_range = max_x - min_x
    
        # Obtener el ancho del gráfico en píxeles
        plot_width = self._price_plot_widget.width()
    
        # Calcular cuántos ticks pueden caber basado en el ancho disponible
        # Asumiendo que cada label necesita aproximadamente 130px para ser legible
        max_ticks_by_width = max(3, int(plot_width / 130))
    
        # Filtrar los índices visibles dentro del rango actual
        visible_indices = np.where((self._price_x_values >= min_x) & (self._price_x_values <= max_x))[0]
    
        # Determinar si estamos en un nivel de zoom grande
        is_high_zoom = len(visible_indices) < 5 and x_range > 86400  # Al menos un día y pocos puntos visibles
    
        if is_high_zoom and x_range > 0:
            # Caso de zoom elevado: generar ticks interpolados a intervalos regulares
            optimal_tick_count = min(max_ticks_by_width, 12)
        
            # Generar timestamps interpolados uniformemente distribuidos
            interpolated_timestamps = np.linspace(min_x, max_x, optimal_tick_count)
        
            # Crear los ticks con las fechas formateadas
            if plot_width < 500:
                ticks = [(ts, self._format_date_compact(ts)) for ts in interpolated_timestamps]
            else:
                ticks = [(ts, self._format_date(ts)) for ts in interpolated_timestamps]
    
        elif len(visible_indices) > 0:
            # Caso normal: usar los puntos de datos existentes
            time_based_ticks = min(10, max(4, int(x_range / (86400 * 7))))
            num_ticks = min(time_based_ticks, max_ticks_by_width)
        
            step = max(1, len(visible_indices) // num_ticks)
            tick_indices = visible_indices[::step]
        
            # Usar formato de fecha más compacto cuando hay poco espacio
            if plot_width < 500:
                ticks = [(self._price_x_values[i], self._format_date_compact(self._price_x_values[i])) for i in tick_indices]
            else:
                ticks = [(self._price_x_values[i], self._format_date(self._price_x_values[i])) for i in tick_indices]
        else:
            # No hay puntos visibles
            return
        
        # Actualizar los ticks del eje X
        self._price_plot_widget.getAxis('bottom').setTicks([ticks])

    # Dibujar el precio del salmón
    def _draw_price(self, pos_i, pos_j, raft):
        plot_widget = pg.PlotWidget()        
        plot_widget.setTitle(title="Precio del Salmón EUR/kg", color='k')
        plot_widget.setLabels(left="EUR/kg")
        plot_widget.showGrid(x=True, y=True)        
        # Color negro para los ejes
        plot_widget.getAxis('left').setPen('k')  # Color negro para el eje y
        plot_widget.getAxis('left').setTextPen('k')  # Color negro para las etiquetas del eje y
        plot_widget.getAxis('bottom').setPen('k')  # Color negro para el eje x
        plot_widget.getAxis('bottom').setTextPen('k')  # Color negro para las etiquetas del eje x
        # Fondo con tema claro
        plot_widget.setBackground((240, 240, 240, 180))        
        legend = plot_widget.addLegend()

        # Obtener los datos de precios
        price_data = raft.getPriceData()        
        price_data_forescast = raft.getPriceForecastData()
    
        if price_data is None or price_data.empty:
            # Mostrar una 'X' roja si no hay datos de precios
            plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
            # Agregar el widget al layout
            self._view.centralwidget.layout().addWidget(plot_widget, pos_i, pos_j)
            return
        else:
            # Filtrar los datos según las fechas de la balsa
            start_date = raft.getStartDate()
            end_date = raft.getEndDate()
            delta_days = (end_date - start_date).days
        
        # Convertir la columna 'timestamp' a formato datetime si no está ya
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], errors='coerce')
        # Eliminar valores NaT antes de filtrar
        price_data = price_data.dropna(subset=['timestamp'])

        # --- Datos históricos previos a la balsa (mismo rango de días que la balsa) ---
        #prev_start_date = start_date - timedelta(days=delta_days)
        prev_start_date = price_data['timestamp'].min().date()
        df_hist = price_data[(price_data['timestamp'].dt.date >= prev_start_date) & (price_data['timestamp'].dt.date < start_date)]
        df_hist = df_hist.sort_values('timestamp')
        
        # --- Datos de la balsa ---
        filtered_price = price_data[(price_data['timestamp'].dt.date >= start_date) & 
                                   (price_data['timestamp'].dt.date <= end_date)]
        
        # --- UNIR EL ÚLTIMO PUNTO DE df_hist CON EL PRIMERO DE filtered_price ---
        if not df_hist.empty and not filtered_price.empty:
            first_balsa = filtered_price.iloc[[0]]
            df_hist = pd.concat([df_hist, first_balsa], ignore_index=True)
        
        if filtered_price.empty:
            # Mostrar una 'X' roja si no hay datos en el rango de fechas
            plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
            plot_widget.setTitle("Precio del Salmón (sin datos en el rango seleccionado)")
        else:
            # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
            if not df_hist.empty:
                x_hist = df_hist['timestamp'].map(pd.Timestamp.timestamp).values
                y_hist = df_hist['EUR_kg'].values
            else:
                x_hist = np.array([])
                y_hist = np.array([])

            x = filtered_price['timestamp'].map(pd.Timestamp.timestamp).values
            y = filtered_price['EUR_kg'].values

            # Graficar los datos históricos previos en amarillo
            if x_hist.size > 0:
                plot_widget.plot(x_hist, y_hist, pen=pg.mkPen(color='#FFA500', width=2), name="Histórico previo", color='#FFD700')

            # Graficar los datos históricos de precio
            plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2),name="Histórico")
            
            # Guardar referencia al widget y a los valores de X para actualizaciones posteriores
            self._price_plot_widget = plot_widget
            self._price_x_values = np.concatenate([x_hist, x]) if x_hist.size > 0 else x

            # Conectar la función para actualizar los ticks cuando cambia el rango
            # Se filtra el objeto vb: El objeto ViewBox que cambió que se pasa como argumento al no ser necesario
            # Se conecta la señal sigRangeChanged a la función _update_temperature_axis_ticks
            plot_widget.getViewBox().sigRangeChanged.connect(lambda vb, range_vals: self._update_price_axis_ticks(range_vals))

            # Establecer los ticks iniciales
            all_x = np.concatenate([x_hist, x]) if x_hist.size > 0 else x
            all_y = np.concatenate([y_hist, y]) if y_hist.size > 0 else y
            self._update_price_axis_ticks([[all_x.min(), all_x.max()], [all_y.min(), all_y.max()]])

            # Crear un ScatterPlotItem para ver los puntos de datos
            scatter = pg.ScatterPlotItem(x=all_x, y=all_y, pen=pg.mkPen(color='k'), brush=pg.mkBrush(255, 255, 255, 120), size=7)
            plot_widget.addItem(scatter)

            if price_data_forescast is not None and not price_data_forescast.empty:                
                # Asegurar que la columna 'ds' sea de tipo datetime con el formato correcto
                price_data_forescast['ds'] = pd.to_datetime(price_data_forescast['ds'], errors='coerce')
    
                # Eliminar valores NaT que pudieran haberse generado
                price_data_forescast = price_data_forescast.dropna(subset=['ds'])
    
                # Método alternativo para convertir fechas a timestamps
                x_forecast = np.array([pd.Timestamp(date).timestamp() 
                         for date in price_data_forescast['ds'] 
                         if not pd.isna(date)])


                # Asegurarse de que 'y' tiene la misma longitud que x_forecast
                y_forecast = price_data_forescast['y'].iloc[:len(x_forecast)].values                


                # Graficar los datos de precio pronosticados
                plot_widget.plot(x_forecast, y_forecast, pen=pg.mkPen(color='r', width=2, style=Qt.DashLine), 
                                 name="Predicción")                              
                
                # Configurar el rango de visualización para mostrar desde la fecha inicial a la fecha final
                min_x = min(all_x.min(), x_forecast.min())
                max_x = max(all_x.max(), x_forecast.max())
                min_y = min(all_y.min(), y_forecast.min())
                max_y = max(all_y.max(), y_forecast.max())
                
                plot_widget.setXRange(min_x, max_x, padding=0.1)
                plot_widget.setYRange(min_y, max_y, padding=0.1)

                # Personalizar los ticks del eje X
                axis = plot_widget.getAxis('bottom')                
                axis.setLabel("", units="")

                # Añadir línea vertical para la fecha actual
                self.price_vline_forescast = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='r', width=2, style=Qt.DashLine))
                plot_widget.addItem(self.price_vline_forescast)
                self.price_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='b', width=2, style=Qt.DashLine))
                plot_widget.addItem(self.price_vline) 
                # Establecer posición inicial                
                initial_pos = x_forecast[0]
                self.price_vline.setPos(initial_pos)
                self.price_vline_forescast.setPos(initial_pos)

            else:
                # Si no hay datos de predicción, solo mostrar los históricos
                plot_widget.setXRange(x.min(), x.max(), padding=0.1)
                plot_widget.setYRange(y.min(), y.max(), padding=0.1)
                
                axis = plot_widget.getAxis('bottom')                
                axis.setLabel("", units="")
                
                # Añadir línea vertical para la fecha actual                
                self.price_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='b', width=2, style=Qt.DashLine))
                initial_pos = x[0]
                self.price_vline.setPos(initial_pos)
                plot_widget.addItem(self.price_vline)

        # Establecer color negro para la leyenda
        for item in legend.items:
            label = item[1]
            texto_original = label.text
            label.setText(texto_original, color='k')
                
        # Agregar el widget al layout
        self._view.centralwidget.layout().addWidget(plot_widget, pos_i, pos_j)

    def _update_growth_axis_ticks(self,range_vals):
        # Obtener el rango visible actual
        min_x, max_x = range_vals[0]
        x_range = max_x - min_x

        # Obtener el ancho del gráfico en píxeles
        plot_width = self._growth_widget.width()

        # Calcular cuántos ticks pueden caber basado en el ancho disponible
        # Asumiendo que cada label necesita aproximadamente 150px para ser legible
        max_ticks_by_width = max(3, int(plot_width / 150))

        # Filtrar los índices visibles dentro del rango actual
        visible_indices = np.where((self._growth_x_values >= min_x) & (self._growth_x_values <= max_x))[0]
    
        # Determinar si estamos en un nivel de zoom grande
        is_high_zoom = len(visible_indices) < 5 and x_range > 86400  # Al menos un día y pocos puntos visibles

        if is_high_zoom and x_range > 0:
            # Caso de zoom elevado: generar ticks interpolados a intervalos regulares
            optimal_tick_count = min(max_ticks_by_width, 12)
        
            # Generar timestamps interpolados uniformemente distribuidos
            interpolated_timestamps = np.linspace(min_x, max_x, optimal_tick_count)
        
            # Crear los ticks con las fechas formateadas
            if plot_width < 500:
                ticks = [(ts, self._format_date_compact(ts)) for ts in interpolated_timestamps]
            else:
                ticks = [(ts, self._format_date(ts)) for ts in interpolated_timestamps]
    
        elif len(visible_indices) > 0:
            # Caso normal: usar los puntos de datos existentes
            time_based_ticks = min(10, max(4, int(x_range / (86400 * 7))))
            num_ticks = min(time_based_ticks, max_ticks_by_width)
        
            step = max(1, len(visible_indices) // num_ticks)
            tick_indices = visible_indices[::step]
        
            # Usar formato de fecha más compacto cuando hay poco espacio
            if plot_width < 500:
                ticks = [(self._growth_x_values[i], self._format_date_compact(self._growth_x_values[i])) for i in tick_indices]
            else:
                ticks = [(self._growth_x_values[i], self._format_date(self._growth_x_values[i])) for i in tick_indices]
        else:
            # No hay puntos visibles
            return
        
        # Actualizar los ticks del eje X
        self._growth_widget.getAxis('bottom').setTicks([ticks])

    # Dibujar el modelo de crecimiento de la balsa
    def _draw_growth_model(self,pos_i,pos_j,raft):        
        # Crear un widget de gráfico de PyQtGraph
        plot_widget = pg.PlotWidget()
        plot_widget.setTitle("Modelo de Crecimiento de Biomasa Kg", color='k')
        plot_widget.setLabels(left="Biomasa (kg)")
        plot_widget.showGrid(x=True, y=True)        
        # Color negro para los ejes
        plot_widget.getAxis('left').setPen('k')  # Color negro para el eje y
        plot_widget.getAxis('left').setTextPen('k')  # Color negro para las etiquetas del eje y
        plot_widget.getAxis('bottom').setPen('k')  # Color negro para el eje x
        plot_widget.getAxis('bottom').setTextPen('k')  # Color negro para las etiquetas del eje x
        # Fondo con tema claro
        plot_widget.setBackground((240, 240, 240, 180))        
        legend = plot_widget.addLegend()
    
        # Agregar datos al gráfico si existen
        if raft is None or raft.getTemperature().empty:
            # Mostrar una 'X' roja si no hay datos de temperatura
            plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
        else:
            # Obtener los datos de temperatura de la balsa
            df_temperature = raft.getTemperature()
            # Convertir la columna 'ds' a formato datetime si no está ya
            df_temperature['ds'] = pd.to_datetime(df_temperature['ds'], errors='coerce')
            # Eliminar valores NaT antes de filtrar
            df_temperature = df_temperature.dropna(subset=['ds'])
            # Filtrar los datos de temperatura con la fecha inicial y la fecha actual
            percent = raft.getPerCentage()
            delta_days = (raft.getEndDate() - raft.getStartDate()).days
            days = int(delta_days * percent / 1000)
            fecha_actual = raft.getStartDate() + timedelta(days)
            df_temperature = df_temperature[(df_temperature['ds'].dt.date >= raft.getStartDate()) & 
                                        (df_temperature['ds'].dt.date <= fecha_actual)]


            if df_temperature.empty:
                # Mostrar una 'X' roja si no hay datos de temperatura
                plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
            else:
                # Obtener los datos de crecimiento
                growth_data = raft.getGrowthData()
                growth_data_forescast = raft.getGrowthForecastData()
                if growth_data is None or growth_data.empty:
                    # Mostrar una 'X' roja si no hay datos de crecimiento
                    plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
                    self._view.centralwidget.layout().addWidget(plot_widget, pos_i, pos_j)
                    return
            
                # Convertir fechas a objetos datetime primero
                x_dates = pd.to_datetime(growth_data['ds'], errors='coerce')
                xf_dates = pd.to_datetime(growth_data_forescast['ds'], errors='coerce')

                # Luego convertir a timestamps numéricos para graficar y etiquetas
                x = x_dates.map(pd.Timestamp.timestamp).values
                xf = xf_dates.map(pd.Timestamp.timestamp).values                
                y_biomass = growth_data['biomass'].values
                y_biomass_f = growth_data_forescast['biomass'].values
                y_number = growth_data['number_fishes'].values
                y_number_f = growth_data_forescast['number_fishes'].values

                # Configurar el rango de visualización para mostrar desde la fecha inicial a la fecha final
                min_x = min(x.min(), xf.min())
                max_x = max(x.max(), xf.max())
                min_y = min(y_biomass.min(), y_biomass_f.min(),y_number.min(), y_number_f.min())
                max_y = max(y_biomass.max(), y_biomass_f.max(),y_number.max(), y_number_f.max())

                plot_widget.setXRange(min_x, max_x, padding=0.1)
                plot_widget.setYRange(min_y, max_y, padding=0.1)
            
                # Personalizar los ticks del eje X
                axis = plot_widget.getAxis('bottom')                
                axis.setLabel("", units="")

                # Guardar referencia al widget y a los valores de X para actualizaciones posteriores
                self._growth_widget = plot_widget
                self._growth_x_values = np.concatenate((x, xf))

                plot_widget.getViewBox().sigRangeChanged.connect(lambda vb, range_vals: self._update_growth_axis_ticks(range_vals))

                # Establecer los ticks iniciales
                self._update_growth_axis_ticks([[min_x, max_x], [min_y, max_y]])
            
                # Graficar los datos de biomasa, crecimiento individual y número de peces
                plot_widget.plot(x, y_biomass, pen=pg.mkPen(color='b', width=2), name="Biomasa historica")
                plot_widget.plot(xf, y_biomass_f, pen=pg.mkPen(color='r', width=2, style=Qt.DashLine), name="Biomasa Predicción")
                plot_widget.plot(x, y_number, pen=pg.mkPen(color='darkred', width=2), name="Nº de Peces histórico")
                plot_widget.plot(xf, y_number_f, pen=pg.mkPen(color='darkred', width=2, style=Qt.DashLine), name="Nº de Peces Predicción")

                # Añadir línea vertical para la fecha actual
                self.growth_vline_forescast = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='r', width=2, style=Qt.DashLine))
                plot_widget.addItem(self.growth_vline_forescast)
                self.growth_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='b', width=2, style=Qt.DashLine))
                plot_widget.addItem(self.growth_vline)
                                
                # Establecer posición inicial
                initial_pos = xf[0]
                self.growth_vline.setPos(initial_pos)
                self.growth_vline_forescast.setPos(initial_pos)
        
        # Establecer color negro para la leyenda
        for item in legend.items:
            label = item[1]
            texto_original = label.text
            label.setText(texto_original, color='k')
            
        self._view.centralwidget.layout().addWidget(plot_widget, pos_i, pos_j)

    # --- Grafico 3d ---   
    def _draw_schematic_3d(self,pos_i,pos_j,raft):        
        # Crear un widget 3D
        view = gl.GLViewWidget()
        view.setBackgroundColor(pg.mkColor(210, 225, 230, 255))
        # Configurar el rango inicial de la cámara
        view.setCameraPosition(distance=40, elevation=20, azimuth=45)
        # --- Añadir Título al Gráfico 3D ---
        title_text = "Biomasa en " + raft.getName()
        text_item = gl.GLTextItem(text=title_text, color=(0,0,0,255), font=QFont("Arial", 18))
        text_item.setData(pos=np.array([0, 0, 5]), text=title_text)
        view.addItem(text_item)
       
        # Agregar una cuadrícula para referencia
        grid = gl.GLGridItem()
        grid.setColor((100,100,100,255))
        grid.scale(1, 1, 1)
        view.addItem(grid)        
        # Dibujar las redes bajo el agua
        self._create_nets(view)
        # Dibujar flotadores alrededor de la balsa
        self._create_flotadores(view)
        # Dibujar peces dentro de la red
        self._create_fish(view,raft)
        # Mostrar el widget 3D        
        self._view.centralwidget.layout().addWidget(view,pos_i,pos_j)

    # Helper para generar los datos de malla de un toroide
    def _create_torus_mesh_data(self, major_radius, minor_radius, z_offset, major_segments, minor_segments):
        vertices = []
        faces = []
        
        # Generar vértices
        for i in range(major_segments + 1): # u (ángulo alrededor del eje Z principal del toroide)
            u_angle = 2 * np.pi * i / major_segments
            cos_u = np.cos(u_angle)
            sin_u = np.sin(u_angle)
            
            for j in range(minor_segments + 1): # v (ángulo alrededor del círculo del tubo del toroide)
                v_angle = 2 * np.pi * j / minor_segments
                cos_v = np.cos(v_angle)
                sin_v = np.sin(v_angle)
                
                # Coordenadas del toroide
                x = (major_radius + minor_radius * cos_v) * cos_u
                y = (major_radius + minor_radius * cos_v) * sin_u
                z = z_offset + minor_radius * sin_v # z_offset es el nivel Z del centro del toroide
                vertices.append([x, y, z])

        vertices = np.array(vertices)

        # Generar caras (formando quads, divididos en dos triángulos)
        for i in range(major_segments):
            for j in range(minor_segments):
                # Índices de los 4 vértices que forman un quad en la superficie del toroide
                v1_idx = i * (minor_segments + 1) + j
                v2_idx = (i + 1) * (minor_segments + 1) + j
                v3_idx = (i + 1) * (minor_segments + 1) + (j + 1)
                v4_idx = i * (minor_segments + 1) + (j + 1)
                
                # Crear dos triángulos para el quad
                faces.append([v1_idx, v2_idx, v4_idx]) # Triángulo 1
                faces.append([v2_idx, v3_idx, v4_idx]) # Triángulo 2
                
        faces = np.array(faces)
        return gl.MeshData(vertexes=vertices, faces=faces)

    # Crear redes bajo el agua
    def _create_nets(self, view):
        # Parámetros generales de la red
        radius = 10                 # Radio de la balsa, será el radio mayor de los toroides
        height = 5                  # Altura total de la red
        num_vertical_supports = 24  # Número de cilindros verticales
        num_horizontal_rings = 3    # Número de anillos horizontales intermedios (total num_horizontal_rings + 2 con top/bottom)
        net_thickness = 0.1         # Grosor de los tubos de la red (radio menor de toroides y radio de cilindros)
        
        # Parámetros específicos para los toroides (anillos horizontales)
        torus_major_segments = 40   # Segmentos a lo largo del perímetro del toroide (más = más suave)
        torus_minor_segments = 12   # Segmentos para la sección transversal del tubo del toroide (más = tubo más redondo)
        
        # Color para la red (azul oscuro opaco)
        net_color_tuple = (50/255, 70/255, 150/255, 255/255) # (0.196, 0.275, 0.588, 1.0)

        # 1. Soportes Verticales (Cilindros)
        for i in range(num_vertical_supports):
            angle = 2 * np.pi * i / num_vertical_supports
            x_pos = radius * np.cos(angle)
            y_pos = radius * np.sin(angle)

            mesh_data_vertical = gl.MeshData.cylinder(rows=2, cols=12, 
                                                      radius=[net_thickness, net_thickness], 
                                                      length=height)
            vertical_support = gl.GLMeshItem(meshdata=mesh_data_vertical, smooth=True, 
                                             color=net_color_tuple, shader='shaded')
            
            vertical_support.translate(x_pos, y_pos, -height) 
            view.addItem(vertical_support)

        # 2. Anillos Horizontales (Toroides)
        # Se incluyen los perímetros superior (z=0) e inferior (z=-height), más los intermedios.
        ring_z_levels = np.linspace(0, -height, num_horizontal_rings + 2)

        for z_level in ring_z_levels:
            # Crear datos de malla para el toroide
            torus_mesh_data = self._create_torus_mesh_data(
                major_radius=radius,           # Radio grande del toroide (radio de la balsa)
                minor_radius=net_thickness,    # Radio pequeño del toroide (grosor del tubo)
                z_offset=z_level,              # Nivel Z del centro del toroide
                major_segments=torus_major_segments,
                minor_segments=torus_minor_segments
            )
            
            torus_item = gl.GLMeshItem(meshdata=torus_mesh_data, smooth=True,
                                         color=net_color_tuple, shader='shaded')
            
            # El toroide ya se genera en su posición y orientación correctas.
            view.addItem(torus_item)

    # Crear flotadores (esferas distribuidas en el círculo)
    def _create_flotadores(self,view):
        # Flotadores (esferas distribuidas en el círculo)
        radius = 10
        flotador_positions = 8
        flotador_radius = 0.5 # Radio de la esfera del flotador
        flotador_color = (0.2, 0.8, 0.2, 1) # Verde más brillante y opaco

        for i in range(flotador_positions):
            angle = 2 * np.pi / flotador_positions * i
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0 # Los flotadores están en la superficie (z=0)

            # Crear una malla de esfera para cada flotador
            mesh_data_flotador = gl.MeshData.sphere(rows=10, cols=10, radius=flotador_radius)
            sphere_item = gl.GLMeshItem(meshdata=mesh_data_flotador, smooth=True,
                                        color=flotador_color, shader='shaded')
            sphere_item.translate(x, y, z)
            view.addItem(sphere_item)    

    # Crear peces (esferas pequeñas dentro de la red)
    def _create_fish(self, view, raft):
        # Crear peces (esferas pequeñas dentro de la red)
        self.fish_items = []
        self.fish_positions = []
        self.fish_orientations = []
        # Número de peces
        self.fish_count = int(raft.getCurrentNumberFishes())        
        # Cuerpo de pez
        body_mesh_data = gl.MeshData.sphere(rows=8, cols=8, radius=0.2)
        # La cola del pez
        tail_mesh_data = gl.MeshData.sphere(rows=6, cols=6, radius=0.2)
        body_color = (0.9, 0.5, 0.1, 1)  # Naranja/amarillo opaco para el cuerpo
        tail_color = (0.7, 0.3, 0.05, 1) # Marrón opaco para la cola
        # Parámetros de forma
        body_scale_x = 2.5
        body_radius = 0.2
        tail_offset_x = - (body_radius * body_scale_x * 0.55) - 0.1
        for _ in range(self.fish_count):
            x = random.uniform(-7, 7)
            y = random.uniform(-7, 7)
            z = random.uniform(-5, -0.2)
            body_item = gl.GLMeshItem(meshdata=body_mesh_data, smooth=True, shader='shaded', color=body_color)
            tail_item = gl.GLMeshItem(meshdata=tail_mesh_data, smooth=True, shader='shaded', color=tail_color)
            # Orientación inicial aleatoria
            yaw = random.uniform(0, 360)    # Rotación alrededor del eje Z (vertical)            
            pitch = random.uniform(-15, 15) # Inclinación ligera hacia arriba/abajo            

            body_item.resetTransform()
            body_item.scale(body_scale_x, 1.0, 1.0)
            body_item.rotate(yaw, 0, 0, 1, local=False)
            body_item.rotate(pitch, 0, 1, 0, local=False) # Rotar pitch sobre el eje Y local (ya afectado por yaw)
            body_item.translate(x, y, z, local=False)
            tail_item.resetTransform()
            tail_item.scale(1.2, 0.1, 0.8)
            tail_item.translate(tail_offset_x, 0, 0, local=True) # Posicionar detrás del cuerpo (en el eje X local del cuerpo)
            tail_item.rotate(yaw, 0, 0, 1, local=False)    # Aplicar la misma orientación que el cuerpo
            tail_item.rotate(pitch, 0, 1, 0, local=False)
            tail_item.translate(x, y, z, local=False)
            self.fish_items.append({'body': body_item, 'tail': tail_item})
            self.fish_positions.append([x, y, z])
            self.fish_orientations.append({'yaw': yaw, 'pitch': pitch, 'body_scale_x': body_scale_x, 'tail_offset_x': tail_offset_x})
            view.addItem(body_item)
            view.addItem(tail_item)                       

        # Configurar un temporizador para animar los peces
        self.timer.stop()        
        # Desconectar primero si ya estaba conectado
        if self.timer_connected:
            try:
                self.timer.timeout.disconnect(self._update_fish_positions)
            except TypeError:
                pass
        self.timer.timeout.connect(self._update_fish_positions)
        self.timer_connected = True  # Marcar como conectado
        # Actualizar la posición de los peces cada 500 ms
        self.timer.start(500)  
    
    # Actualizar la posición de los peces
    def _update_fish_positions(self):
         # Exit if fish items or positions are not initialized.
        if not hasattr(self, 'fish_items') or not hasattr(self, 'fish_positions') or not hasattr(self, 'fish_orientations'):
            return
        
        swim_speed = 0.1
        net_radius = 8
        net_depth_max = -0.2
        net_depth_min = -5.0

        for i, fish in enumerate(self.fish_items):
            # Obtener el cuerpo y la cola del pez
            body_item = fish['body']
            tail_item = fish['tail']
            # Obtener la posición actual
            current_pos = self.fish_positions[i]
            orientation = self.fish_orientations[i]
            x, y, z = current_pos
            previous_yaw = orientation['yaw']
            pitch = orientation['pitch']
            body_scale_x = orientation['body_scale_x']
            tail_offset_x = orientation['tail_offset_x']

            # Calcular pequeños desplazamientos (deltas)
            delta_turn_angle = random.uniform(0, 10)
            yaw_for_this_frame = (previous_yaw + delta_turn_angle) % 360
            delta_pitch_angle = random.uniform(0, 1)
            pitch_for_this_frame = (pitch + delta_pitch_angle) % 360            
            yaw_rad_for_movement = np.radians(yaw_for_this_frame)
            pitch_rad_for_movement = np.radians(pitch_for_this_frame)
            delta_x_pos = swim_speed * np.cos(yaw_rad_for_movement)
            delta_y_pos = swim_speed * np.sin(yaw_rad_for_movement)
            delta_z_pos = swim_speed * np.sin(pitch_rad_for_movement)

            # --- Lógica de movimiento vertical (Z) mejorada ---
            z_range = net_depth_max - net_depth_min
            edge_zone_percentage = 0.15 # Porcentaje del rango Z considerado "cerca del borde"
            edge_zone_threshold = z_range * edge_zone_percentage
            
            min_z_movement_away = 0.01  # Mínimo movimiento para alejarse del borde
            max_z_movement_away = 0.03  # Máximo movimiento para alejarse del borde
            general_z_movement = 0.02   # Rango de movimiento general en Z

            if z <= net_depth_min + edge_zone_threshold:
                # Cerca del fondo, tender a moverse hacia arriba
                delta_z_pos = random.uniform(min_z_movement_away, max_z_movement_away)
            elif z >= net_depth_max - edge_zone_threshold:
                # Cerca de la superficie, tender a moverse hacia abajo
                delta_z_pos = random.uniform(-max_z_movement_away, -min_z_movement_away)
            else:
                # En la zona media, movimiento aleatorio normal
                delta_z_pos = random.uniform(-general_z_movement, general_z_movement)

            # Calcular nuevas posiciones
            new_x = x + delta_x_pos
            new_y = y + delta_y_pos
            new_z = z + delta_z_pos

             # Limitar posiciones dentro de los límites de la red
            final_yaw_to_use = yaw_for_this_frame
            if np.sqrt(new_x**2 + new_y**2) >= net_radius:
                # Si golpea el borde, invertir el componente de velocidad que lo llevó allí
                # y/o cambiar drásticamente el yaw
                new_x = np.clip(new_x, -net_radius, net_radius)
                new_y = np.clip(new_y, -net_radius, net_radius)
                final_yaw_to_use = (yaw_for_this_frame + 30) % 360 # Girar 30 grados
                push_strength = 0.1
                dist_from_origin = np.sqrt(new_x**2 + new_y**2) # Debería ser net_radius aquí
                if dist_from_origin > 0: # Evitar división por cero
                    direction_to_center_x = -new_x / dist_from_origin
                    direction_to_center_y = -new_y / dist_from_origin                    
                    new_x += direction_to_center_x * push_strength
                    new_y += direction_to_center_y * push_strength


            if new_z > net_depth_max or new_z < net_depth_min:
                new_z = np.clip(new_z, net_depth_min, net_depth_max)

            orientation['yaw'] = final_yaw_to_use

            # Actualizar la posición del pez
            self.fish_positions[i] = [new_x, new_y, new_z]            
            # Actualizar transformación del cuerpo
            body_item.resetTransform()
            body_item.scale(body_scale_x, 1.0, 1.0)
            body_item.rotate(final_yaw_to_use, 0, 0, 1, local=False)
            body_item.rotate(pitch, 0, 1, 0, local=False)
            body_item.translate(new_x, new_y, new_z, local=False)
            # Actualizar transformación de la cola
            tail_item.resetTransform()
            tail_item.scale(1.2, 0.1, 0.8)           
            tail_item.translate(tail_offset_x, 0, 0, local=True) # Offset local respecto al cuerpo
            tail_item.rotate(final_yaw_to_use, 0, 0, 1, local=False)    # Misma orientación que el cuerpo
            tail_item.rotate(pitch, 0, 1, 0, local=False)
            tail_item.translate(new_x, new_y, new_z, local=False)
    # --- Fin Grafico 3d ---
    
    # Calcular la fecha óptima de cosecha
    def _calculate_optimal_harvest_date(self, raft):
        try:            
            # Buscar el punto donde se maximiza biomasa*precio de fecha menor
            # Obtener las fechas de inicio y fin de la balsa
            start_date_raft = pd.to_datetime(raft.getCurrentDate())
            end_date_raft = pd.to_datetime(raft.getEndDate())
            dateOffset = pd.DateOffset(days=7)  # Avanzar en intervalos de 7 días
            deltaMinPrice = 0.01  # Margen mínimo de precio para considerar un cambio significativo            

            if start_date_raft is None or end_date_raft is None:
                return None
            
            optimal_date = start_date_raft
            current_date = start_date_raft
            maxBiomassPrice = raft.getBiomassForecast(current_date) * raft.getPriceForecast(current_date)
            while current_date <= end_date_raft:
                biomass = raft.getBiomassForecast(current_date)
                if biomass is None or biomass <= 0:
                    # Si no hay biomasa, saltar a la siguiente fecha
                    current_date += dateOffset
                    continue                
                price = raft.getPriceForecast(current_date)
                if price is None or price <= 0:
                    # Si no hay precio, saltar a la siguiente fecha
                    current_date += dateOffset
                    continue                
                # Calcular biomasa * precio
                biomass_price = biomass * price
                # Si es mayor que el máximo encontrado, actualizar
                inc = abs(biomass_price - maxBiomassPrice) / biomass_price
                if inc > deltaMinPrice and biomass_price > maxBiomassPrice:
                    maxBiomassPrice = biomass_price
                    optimal_date = current_date
                # Avanzar a la siguiente fecha
                current_date += dateOffset           
                    
            # Obtener la biomasa y precio en esa fecha
            biomass = raft.getBiomassForecast(optimal_date)
            price = raft.getPriceForecast(optimal_date)
        
            # Buscar el número de peces correspondiente a esa fecha
            nFishes = raft.getNumberFishesForecast(optimal_date)
        
            # Calcular el valor total
            total = maxBiomassPrice

            return optimal_date, biomass, price, nFishes, total

        except Exception as e:
            print(f"Error calculando fecha óptima: {str(e)}")
            return None       

    # --- Grafico 2d ---
    def _draw_raft_2D(self, scene, scene_size, cage_radius, deltaY=0.0):
        # Definir el área de la escena
        scene.setSceneRect(-scene_size/2, -scene_size/2, scene_size, scene_size)
        # Colores        
        net_color = QColor(100, 100, 255, 150)        # Azul semitransparente
        float_color = QColor(200, 200, 200)           # Gris claro
        support_color = QColor(150, 150, 150)         # Gris oscuro
        # Pluma y pincel comunes
        pen = QPen(Qt.black)
        net_brush = QBrush(net_color)
        float_brush = QBrush(float_color)        

        # 1. Estructura Flotante (Círculo principal)
        floating_structure = scene.addEllipse(-cage_radius*3, -cage_radius*3 + deltaY,
                                               cage_radius*4,  cage_radius*4,
                                                  pen, float_brush)
        floating_structure.setToolTip(cfg.DASHBOARD_GRAPH_MAINSTRUCTURE_MSG)
        # 2. Red de la Jaula
        net = scene.addEllipse(-cage_radius*3 + 5, -cage_radius*3 + 5 + deltaY,
                                     4 * cage_radius - 10, 4 * cage_radius - 10,
                                     pen, net_brush)
        net.setToolTip(cfg.DASHBOARD_GRAPH_NET_MSG)
        # 3. Soportes (Ejemplo: líneas radiales)
        num_supports = 8
        support_length = cage_radius + 15 + deltaY / 2  # Longitud de los soportes
        for i in range(num_supports):
            angle = 360 / num_supports * i            
            x1 = -cage_radius
            y1 = -cage_radius + deltaY
            x2 = x1 + support_length * math.cos(math.radians(angle))
            y2 = y1 + support_length * math.sin(math.radians(angle))
            support = scene.addLine(x1, y1, x2, y2, QPen(support_color, 2))
            support.setToolTip(cfg.DASHBOARD_GRAPH_PILLARS_MSG)
        # 4. Anclajes (Ejemplo: pequeños rectángulos en los extremos de los soportes)
        anchor_size = 5
        for i in range(num_supports):
            angle = 360 / num_supports * i            
            x = -cage_radius + (support_length + anchor_size) * math.cos(math.radians(angle))
            y = -cage_radius + deltaY + (support_length + anchor_size) * math.sin(math.radians(angle))
            anchor = scene.addRect(x - anchor_size / 2, y - anchor_size / 2,
                                         anchor_size, anchor_size, pen, QBrush(Qt.blue))
            anchor.setToolTip(cfg.DASHBOARD_GRAPH_ANCHOR_MSG)

    def _draw_miniDash(self,pos_i,pos_j,currentRaft,rafts):
        # Contenedor principal
        main_widget = QWidget()
        grid_layout = QGridLayout(main_widget)
        grid_layout.setSpacing(0)  # Espacio entre elementos del grid
        # Espaciador para que los widgets no se estiren demasiado
        grid_layout.setRowStretch(1, 1)        
        col = 0
        for raft in rafts:
            # Detectar si es la balsa actual
            isCrrent = (raft.getId() == currentRaft.getId())            
            # Crear un QGraphicsView para mostrar la información de la balsa
            view = QGraphicsView()            
            scene = QGraphicsScene()
            # Aplicar un estilo con fondo semitransparente
            view.setStyleSheet("""                
                QGraphicsView {
                    background-color: rgba(200, 200, 200, 150); /* Gris claro semitransparente */
                    border: 1px solid black; /* Borde negro opcional */
                }
                """)
            # Configurar un tamaño inicial para la escena
            deltaY = 25      
            if isCrrent:
                view.setMaximumWidth(350)
                view.setMaximumHeight(165)
                scene_size = 150
                cage_radius = 20
                font_name_size = 14
                font_title_size = 12
                font_info_size = 11
                multiplier_pos = 5
                self._draw_raft_2D(scene,scene_size,cage_radius, deltaY)        
                view.setScene(scene)       
            else:
                view.setMaximumWidth(270)
                view.setMaximumHeight(140)
                scene_size = 130
                cage_radius = scene_size / 10
                font_name_size = 10
                font_title_size = 9
                font_info_size = 8
                multiplier_pos = 5
                self._draw_raft_2D(scene,scene_size,cage_radius)        
                view.setScene(scene)        
        
            # 5. Añadir cajas informativas con valor esperado y fecha óptima
            # Intentar calcular los valores
            result = self._calculate_optimal_harvest_date(raft)
            date, biomass, price, nFishes, total = result
            if result is not None and total>0:                
                optimal_date = date.strftime("%d/%m/%Y") if hasattr(date, 'strftime') else "N/A"
                expected_value = total if total is not None else 0
            else:
                # Valores predeterminados cuando no se puede calcular
                optimal_date = "N/A"
                expected_value = "N/A"

            if isCrrent:
                # Nombre de la balsa
                raft_name = scene.addText(raft.getName(), QFont("Arial", font_name_size, QFont.Bold))
                raft_name.setDefaultTextColor(QColor(0, 0, 0))
                raft_name.setPos(-cage_radius*4, -cage_radius*5 + deltaY)  # Ajustar la posición vertical según el deltaY
            else:
                # Nombre de la balsa
                raft_name = scene.addText(raft.getName(), QFont("Arial", font_name_size, QFont.Bold))
                raft_name.setDefaultTextColor(QColor(0, 0, 0))
                raft_name.setPos(-cage_radius*4, -cage_radius*5 - deltaY/3)  # Ajustar la posición vertical según el deltaY

            if not isCrrent:
                # Título        
                title_text = scene.addText("Información de Cosecha", QFont("Arial", font_title_size, QFont.Bold))
                title_text.setDefaultTextColor(QColor(0, 0, 0))
                title_text.setPos(-cage_radius*multiplier_pos, cage_radius*1.5)    
            
                # Valor esperado
                if expected_value == "N/A":
                    value_text = scene.addText("Valor esperado: N/A", QFont("Arial", font_info_size))
                    value_text.setDefaultTextColor(QColor(150, 0, 0)) # Rojo
                else:
                    value_text = scene.addText(f"Valor esperado: {expected_value:.2f} EUR", QFont("Arial", font_info_size))
                    value_text.setDefaultTextColor(QColor(0, 100, 0))  # Verde
                value_text.setPos(-cage_radius*multiplier_pos, cage_radius*2.5)
    
                # Fecha óptima
                if optimal_date == "N/A":
                    date_text = scene.addText("Recogida óptima: N/A", QFont("Arial", font_info_size))
                    date_text.setDefaultTextColor(QColor(150, 0, 0)) # Rojo
                else:
                    date_text = scene.addText(f"Recogida óptima: {optimal_date}", QFont("Arial", font_info_size))
                    date_text.setDefaultTextColor(QColor(0, 100, 0))  # Verde
                date_text.setPos(-cage_radius*multiplier_pos, cage_radius*3.5)
    
            # Solo añadir el view al layout
            if isCrrent:
                grid_layout.addWidget(view, 0, 0, 1, 1)
            else:
                col += 1
                grid_layout.addWidget(view, 0, col, 1, 1)

        #Cuadro de información con el resultado de la balsa actual
        viewResult = QWidget()
        viewResultLayout = QGridLayout(viewResult)       
        lforescastValue = QLabel("Valor esperado: N/A")
        lcurrentDate = QLabel("Recogida óptima: N/A")
        label_style_big = """
            QLabel {
                font-size: 18px; /* Tamaño de la letra */
                background-color: rgba(200, 200, 200, 150); /* Fondo semitransparente */
                color: black; /* Color del texto */
                border: 1px solid gray; /* Opcional: borde */
                padding: 1px; /* Margen interno */
            }
        """
        lforescastValue.setStyleSheet(label_style_big)
        lcurrentDate.setStyleSheet(label_style_big)
        viewResultLayout.addWidget(lforescastValue,0,0,1,col+1)
        viewResultLayout.addWidget(lcurrentDate,1,0,1,col+1)
        grid_layout.addWidget(viewResult, 1, 0, 1, col+1)

        if currentRaft is not None:
          result = self._calculate_optimal_harvest_date(currentRaft)
          date, biomass, price, nFishes, total = result
          if result is not None and total>0:            
            optimal_date = date.strftime("%d/%m/%Y") if hasattr(date, 'strftime') else "N/A"
            expected_value = total if total is not None else 0
            # Actualizar etiquetas con los valores calculados
            lforescastValue.setText(f"Valor esperado: {expected_value:.2f} EUR")
            lcurrentDate.setText(f"Recogida óptima: {optimal_date}")
        else:
            # Si no hay balsa actual, mostrar valores predeterminados
            lforescastValue.setText("Valor esperado: N/A")
            lcurrentDate.setText("Recogida óptima: N/A")
                 
        
        self._view.centralwidget.layout().addWidget(main_widget,pos_i,pos_j)
    # --- Fin Grafico 2d ---

    def _update_forescast_date(self, perCentage, raft, lforescastDate, lforescastFishNumber, lforescastBiomass, lforescastPrice, lforescastTotalValue):
        # Mapeamos 0-1000 desde la fecha actual hasta la fecha máxima de pronosticada
        current_date = raft.getCurrentDate()
        max_date = raft.getMaxForecastDate()
        delta_days = (max_date - current_date).days        
        current_day_offset = int(delta_days * (perCentage / 1000))  # Dividir por 1000
        current_date = current_date + timedelta(days=current_day_offset)

        # Formatear la fecha y actualizar la etiqueta
        formatted_date = current_date.strftime("%d de %B de %Y")
        lforescastDate.setText("Fecha pronóstico: " + formatted_date)        
        lforescastFishNumber.setText("Pronóstico número de peces: {0:.0f}".format(raft.getNumberFishesForecast(current_date)))
        lforescastBiomass.setText("Pronóstico Biomasa: {0:.2f} kg".format(raft.getBiomassForecast(current_date)))
        lforescastPrice.setText("Pronóstico Precio: {0:.2f} EUR/kg".format(raft.getPriceForecast(current_date)))
        lforescastTotalValue.setText("Pronóstico Valor total: {0:.2f} EUR".format(raft.getTotalValueForecast(current_date)))

        # Actualizar líneas verticales en todas las gráficas SOLO si existen
        try:
            timestamp = pd.Timestamp(current_date).timestamp()
            # Verificar que las líneas existan y no hayan sido eliminadas
            if hasattr(self, 'date_vline_forescast') and self.date_vline_forescast is not None:
                try:
                    self.date_vline_forescast.setPos(timestamp)
                except RuntimeError:
                    # La línea fue eliminada, limpiar la referencia
                    self.date_vline_forescast = None                
            if hasattr(self, 'growth_vline_forescast') and self.growth_vline_forescast is not None:
                try:
                    self.growth_vline_forescast.setPos(timestamp)
                except RuntimeError:
                    # La línea fue eliminada, limpiar la referencia
                    self.growth_vline_forescast = None                
            if hasattr(self, 'price_vline_forescast') and self.price_vline_forescast is not None:
                try:
                    self.price_vline_forescast.setPos(timestamp)
                except RuntimeError:
                    # La línea fue eliminada, limpiar la referencia
                    self.price_vline_forescast = None
        except Exception as e:
            print(cfg.DASHBOARD_DATE_VLINE_FOR_ERROR.format(error=str(e)))

    # Función para actualizar la etiqueta de la fecha cuando cambia el valor del slider
    # Actualiza las lineas verticales en todas las gráficas
    def _update_current_date(self, perCentage, raft, lcurrentDate):
        start_date = raft.getStartDate()
        end_date = raft.getEndDate()
        # Mapeamos 0-1000 al rango histórico
        delta_days = (end_date - start_date).days        
        current_day_offset = int(delta_days * (perCentage / 1000))
        current_date = start_date + timedelta(days=current_day_offset)

        # Formatear la fecha y actualizar la etiqueta
        formatted_date = current_date.strftime("%d de %B de %Y")
        lcurrentDate.setText("Fecha actual: " + formatted_date)

        # Actualizar líneas verticales en todas las gráficas SOLO si existen
        try:
            timestamp = pd.Timestamp(current_date).timestamp()
            # Verificar que las líneas existan y no hayan sido eliminadas
            if hasattr(self, 'date_vline') and self.date_vline is not None:
                try:
                    self.date_vline.setPos(timestamp)
                except RuntimeError:
                    # La línea fue eliminada, limpiar la referencia
                    self.date_vline = None
                
            if hasattr(self, 'growth_vline') and self.growth_vline is not None:
                try:
                    self.growth_vline.setPos(timestamp)
                except RuntimeError:
                    # La línea fue eliminada, limpiar la referencia
                    self.growth_vline = None
                
            if hasattr(self, 'price_vline') and self.price_vline is not None:
                try:
                    self.price_vline.setPos(timestamp)
                except RuntimeError:
                    # La línea fue eliminada, limpiar la referencia
                    self.price_vline = None
        except Exception as e:
            print(cfg.DASHBOARD_DATE_VLINE_HIS_ERROR.format(error=str(e)))

    # Datos de la balsa
    def _draw_infopanel(self,pos_i,pos_j,raft):
        # Crear un widget para mostrar información de la balsa
        view = QWidget()
        # Crear un layout vertical para organizar los QLabel
        layout = QGridLayout()
        view.setLayout(layout)
        # Mostrar información de la balsa
        lName = QLabel(raft.getName())
        lRegion = QLabel("Región del mar: {0}".format(raft.getSeaRegion()))
        lLocation = QLabel("Ubicación: 12.3456, -78.9012")
        lDepth = QLabel("Profundidad: 10 m")
        lnumberFishes = QLabel("Número inicial de peces: {0}".format(raft.getNumberFishes()))
        
        # Crear un layout para el slider de la fecha actual y el pronóstico       
        sliderLayoutCurrent = QGridLayout()
        sliderViewCurrent = QWidget()
        sliderViewCurrent.setLayout(sliderLayoutCurrent)        

        sliderLayoutForescast = QGridLayout()
        sliderViewForescast = QWidget()
        sliderViewForescast.setLayout(sliderLayoutForescast)

        # Configurar el slider con el rango 0-1000
        self.dateSliderCurrent = QSlider(Qt.Horizontal)
        self.dateSliderCurrent.setMinimum(0)
        self.dateSliderCurrent.setMaximum(1000)            
        self.dateSliderCurrent.setValue(raft.getPerCentage())

        self.dateSliderForecast = QSlider(Qt.Horizontal)
        self.dateSliderForecast.setMinimum(0)
        self.dateSliderForecast.setMaximum(1000)
        # Valor inicial para el pronóstico
        self.dateSliderForecast.setValue(0)  

        # El texto inicial será para la fecha de inicio de la balsa        
        lcurrentDate = QLabel("Fecha actual: " + raft.getCurrentDate().strftime("%d de %B de %Y"))
        lcurrentFishNumber = QLabel("Número de peces: {0:.0f}".format(raft.getCurrentNumberFishes()))
        lcurrentBiomass = QLabel("Biomasa: {0:.2f} kg".format(raft.getCurrentBiomass()))
        lcurrentPrice = QLabel("Precio: {0:.2f} EUR/kg".format(raft.getCurrentPrice()))
        lcurrentTotalValue = QLabel("Valor total: {0:.2f} EUR".format(raft.getCurrentTotalValue()))

        lforescastDate = QLabel("Fecha pronóstico: " + raft.getCurrentDate().strftime("%d de %B de %Y"))
        lforescastFishNumber = QLabel("Pronóstico número de peces: {0:.0f}".format(raft.getCurrentNumberFishes()))
        lforescastBiomass = QLabel("Pronóstico Biomasa: {0:.2f} kg".format(raft.getCurrentBiomass()))
        lforescastPrice = QLabel("Pronóstico Precio: {0:.2f} EUR/kg".format(raft.getCurrentPrice()))
        lforescastTotalValue = QLabel("Pronóstico Valor total: {0:.2f} EUR".format(raft.getCurrentTotalValue()))

        sliderLayoutCurrent.addWidget(lcurrentDate,0,0)
        sliderLayoutCurrent.addWidget(lcurrentFishNumber,0,1)
        sliderLayoutCurrent.addWidget(lcurrentBiomass,0,2)
        sliderLayoutCurrent.addWidget(lcurrentPrice,0,3)
        sliderLayoutCurrent.addWidget(lcurrentTotalValue,0,4)
        sliderLayoutCurrent.addWidget(self.dateSliderCurrent,1,0,1,5)

        sliderLayoutForescast.addWidget(lforescastDate,0,0)
        sliderLayoutForescast.addWidget(lforescastFishNumber,0,1)
        sliderLayoutForescast.addWidget(lforescastBiomass,0,2)
        sliderLayoutForescast.addWidget(lforescastPrice,0,3)
        sliderLayoutForescast.addWidget(lforescastTotalValue,0,4)
        sliderLayoutForescast.addWidget(self.dateSliderForecast,1,0,1,5)
       
        # Mostrar las fechas de inicio y fin en formato de idioma castellano        
        formatted_start_date = raft.getStartDate().strftime("%d de %B de %Y")
        formatted_end_date = raft.getEndDate().strftime("%d de %B de %Y")
        lFechas = QLabel("Fechas: {0} - {1}".format(formatted_start_date, formatted_end_date))
        if raft.getTemperature().empty:
            lTemperature = QLabel("Temperatura: No disponible")
        else:
            # Obtener la temperatura de la fecha actual            
            temp = raft.geCurrentDateTemperature()
            # Mostrar la temperatura promedio            
            lTemperature = QLabel("Temperatura: {0:.2f} °C".format(temp))

        # Estilo para los QLabel
        label_style_small = """
            QLabel {
                font-size: 12px; /* Tamaño de la letra */
                background-color: rgba(200, 200, 200, 150); /* Fondo semitransparente */
                color: black; /* Color del texto */
                border: 1px solid gray; /* Opcional: borde */
                padding: 1px; /* Margen interno */
            }
        """
        label_style_medium = """
            QLabel {
                font-size: 14px; /* Tamaño de la letra */
                background-color: rgba(200, 200, 200, 150); /* Fondo semitransparente */
                color: black; /* Color del texto */
                border: 1px solid gray; /* Opcional: borde */
                padding: 1px; /* Margen interno */
            }
        """
        label_style_big = """
            QLabel {
                font-size: 18px; /* Tamaño de la letra */
                background-color: rgba(200, 200, 200, 150); /* Fondo semitransparente */
                color: black; /* Color del texto */
                border: 1px solid gray; /* Opcional: borde */
                padding: 1px; /* Margen interno */
            }
        """
        label_style_huge = """
            QLabel {
                font-size: 24px; /* Tamaño de la letra */
                background-color: rgba(200, 200, 200, 150); /* Fondo semitransparente */
                color: black; /* Color del texto */
                border: 1px solid gray; /* Opcional: borde */
                padding: 1px; /* Margen interno */
            }
        """
        lName.setStyleSheet(label_style_huge)
        lRegion.setStyleSheet(label_style_big)
        lFechas.setStyleSheet(label_style_big)

        lLocation.setStyleSheet(label_style_medium)
        lDepth.setStyleSheet(label_style_medium)        
        lTemperature.setStyleSheet(label_style_medium)
        lnumberFishes.setStyleSheet(label_style_medium)

        lcurrentDate.setStyleSheet(label_style_small)
        lcurrentFishNumber.setStyleSheet(label_style_small)
        lcurrentBiomass.setStyleSheet(label_style_small)
        lcurrentPrice.setStyleSheet(label_style_small)
        lcurrentTotalValue.setStyleSheet(label_style_medium)

        lforescastDate.setStyleSheet(label_style_small)
        lforescastFishNumber.setStyleSheet(label_style_small)
        lforescastBiomass.setStyleSheet(label_style_small)
        lforescastPrice.setStyleSheet(label_style_small)
        lforescastTotalValue.setStyleSheet(label_style_medium)

        # Añadir los QLabel al layout
        layout.addWidget(lName,0,0,1,1)
        layout.addWidget(lRegion,0,1,1,1)
        layout.addWidget(lFechas,0,2,1,2)

        layout.addWidget(lLocation,1,0,1,1)
        layout.addWidget(lDepth,1,1,1,1)        
        layout.addWidget(lTemperature,1,2,1,1)        
        layout.addWidget(lnumberFishes,1,3,1,1)
        
        # Añadir el slider al layout
        layout.addWidget(sliderViewCurrent,3,0,1,4)
        layout.addWidget(sliderViewForescast,4,0,1,4)

        # Conectar el evento de cambio de valor del slider
        self._update_current_date(raft.getPerCentage(),raft,lcurrentDate)
        self.dateSliderCurrent.valueChanged.connect(lambda value_current: self._update_current_date(value_current, raft, lcurrentDate))
        self._update_forescast_date(0,raft,lforescastDate,lforescastFishNumber,lforescastBiomass,lforescastPrice,lforescastTotalValue)
        self.dateSliderForecast.valueChanged.connect(lambda value_forecast: self._update_forescast_date(value_forecast, raft, 
                                                                                                        lforescastDate,
                                                                                                        lforescastFishNumber,
                                                                                                        lforescastBiomass,
                                                                                                        lforescastPrice,
                                                                                                        lforescastTotalValue
                                                                                                        ))

        self._view.centralwidget.layout().addWidget(view,pos_i,pos_j)

    # Formatear los ticks del eje X con el formato 'día/mes/año'
    def _format_date(self, value):
        date = datetime.fromtimestamp(value)
        return date.strftime('%d/%m/%Y')
    
    def _format_date_compact(self, value):
        """Formato compacto para fechas cuando hay poco espacio"""
        date = datetime.fromtimestamp(value)
        return date.strftime('%d/%m')  # Solo día/mes

    # Mostrar un tooltip con la fecha y la temperatura y una línea vertical
    def _mouse_move_plot(self, event, plot_widget, x, y, y_forecast, vline):
        pos = event
        if plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = plot_widget.plotItem.vb.mapSceneToView(pos)
            vline.setPos(mouse_point.x())
            vline.show()
            # Mostrar tooltip con fecha y temperatura
            closest_index = np.argmin(abs(x - mouse_point.x()))
            date = datetime.fromtimestamp(x[closest_index]).strftime('%d/%m/%Y')
            temperature = y[closest_index]
            if y_forecast is not None:
                temperature_forecast = y_forecast[closest_index]
                plot_widget.setToolTip(f"Fecha: {date}\nTemperatura: {temperature:.2f} ºC\nPredicción: {temperature_forecast:.2f} ºC")
            else:                
                plot_widget.setToolTip(f"Fecha: {date}\nTemperatura: {temperature:.2f} ºC")
        else:
            vline.hide()
            plot_widget.setToolTip(None)

    #Actualiza los ticks del eje X cuando cambia el rango visible
    def _update_temperature_axis_ticks(self, values, range_vals):
        # Obtener el rango visible actual
        min_x, max_x = range_vals
        x_range = max_x - min_x
    
        # Obtener el ancho del gráfico en píxeles
        plot_width = self._temp_plot_widget.width()
    
        # Calcular cuántos ticks pueden caber basado en el ancho disponible
        # Asumiendo que cada label necesita aproximadamente 150px para ser legible
        max_ticks_by_width = max(3, int(plot_width / 150))
    
        # Filtrar los índices visibles dentro del rango actual
        visible_indices = np.where((values >= min_x) & (values <= max_x))[0]
    
        # Determinar si estamos en un nivel de zoom grande
        is_high_zoom = len(visible_indices) < 5 and x_range > 86400  # Al menos un día y pocos puntos visibles
    
        if is_high_zoom and x_range > 0:
            # Caso de zoom elevado: generar ticks interpolados a intervalos regulares
            optimal_tick_count = min(max_ticks_by_width, 12)
        
            # Generar timestamps interpolados uniformemente distribuidos
            interpolated_timestamps = np.linspace(min_x, max_x, optimal_tick_count)
        
            # Crear los ticks con las fechas formateadas
            if plot_width < 600:
                ticks = [(ts, self._format_date_compact(ts)) for ts in interpolated_timestamps]
            else:
                ticks = [(ts, self._format_date(ts)) for ts in interpolated_timestamps]
    
        elif len(visible_indices) > 0:
            # Caso normal: usar los puntos de datos existentes
            time_based_ticks = min(10, max(4, int(x_range / (86400 * 7))))
            num_ticks = min(time_based_ticks, max_ticks_by_width)
        
            step = max(1, len(visible_indices) // num_ticks)
            tick_indices = visible_indices[::step]
        
            # Usar formato de fecha más compacto cuando hay poco espacio
            if plot_width < 600:
                ticks = [(values[i], self._format_date_compact(values[i])) for i in tick_indices]
            else:
                ticks = [(values[i], self._format_date(values[i])) for i in tick_indices]
        else:
            # No hay puntos visibles
            return
        
        # Actualizar los ticks del eje X
        self._temp_plot_widget.getAxis('bottom').setTicks([ticks])

    # Graficar una serie temporal
    def _draw_graph_temperature(self,pos_i,pos_j,raft):
        if raft is None:
            region = "------"
        else:
            region = raft.getSeaRegion()    
        plot_widget = pg.PlotWidget()
        plot_widget.setTitle(title="Temperatura del mar en la región de {0}".format(region), color='k')
        plot_widget.setLabels(left="Grados ºC")        
        plot_widget.showGrid(x=True, y=True)
        # Color negro para los ejes
        plot_widget.getAxis('left').setPen('k')  # Color negro para el eje y
        plot_widget.getAxis('left').setTextPen('k')  # Color negro para las etiquetas del eje y
        plot_widget.getAxis('bottom').setPen('k')  # Color negro para el eje x
        plot_widget.getAxis('bottom').setTextPen('k')  # Color negro para las etiquetas del eje x
        # Fondo con tema claro
        plot_widget.setBackground((240, 240, 240, 180))        
        legend = plot_widget.addLegend()

        # Agregar datos al gráfico si existen
        if raft is None or raft.getTemperature().empty:
            # Mostrar una 'X' roja si no hay datos de temperatura
            plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
            self._view.centralwidget.layout().addWidget(plot_widget,pos_i,pos_j)
            return
        else:
            # Obtener los datos de predicción de temperatura de la balsa si hay
            if not raft.getTemperatureForecast().empty:
                df_temperature_forecast = raft.getTemperatureForecast()
                # Convertir la columna 'ds' a formato timestamp si no está ya en datetime
                df_temperature_forecast['ds'] = pd.to_datetime(df_temperature_forecast['ds'], errors='coerce')
                # Eliminar valores NaT antes de filtrar
                df_temperature_forecast = df_temperature_forecast.dropna(subset=['ds'])
                # Filtrar la predicción para mostrar solo apartir de la fecha actual
                perCentage = raft.getPerCentage() / 1000
                delta_days = (raft.getEndDate() - raft.getStartDate()).days
                forescast_start_date = raft.getStartDate() + timedelta(delta_days * perCentage) - timedelta(days=30)                
                df_temperature_forecast = df_temperature_forecast[(df_temperature_forecast['ds'].dt.date >= forescast_start_date)]
            else:
                df_temperature_forecast = None

            # Obtener todos los datos de temperatura
            df_temperature = raft.getTemperature()
            df_temperature['ds'] = pd.to_datetime(df_temperature['ds'], errors='coerce')
            df_temperature = df_temperature.dropna(subset=['ds'])

            # Fechas de la balsa
            start_date = raft.getStartDate()
            end_date = raft.getEndDate()
            delta_days = (end_date - start_date).days

            # Calcular la fecha de inicio del rango previo
            prev_start_date = df_temperature['ds'].dt.date.min()

            # Datos históricos previos a la balsa (mismo rango de días que la balsa)
            df_hist = df_temperature[(df_temperature['ds'].dt.date >= prev_start_date) & (df_temperature['ds'].dt.date < start_date)]
            df_hist = df_hist.sort_values('ds')
            
             # Datos de la balsa
            df_balsa = df_temperature[(df_temperature['ds'].dt.date >= start_date) & (df_temperature['ds'].dt.date <= end_date)]

            # --- UNIR EL ÚLTIMO PUNTO DE df_hist CON EL PRIMERO DE df_balsa ---
            if not df_hist.empty and not df_balsa.empty:                
                first_balsa = df_balsa.iloc[[0]]                
                # Añadir el punto puente al inicio de df_balsa
                df_hist = pd.concat([df_hist,first_balsa], ignore_index=True)

            # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
            if not df_hist.empty:
                x_hist = df_hist['ds'].map(pd.Timestamp.timestamp).values
                y_hist = df_hist['y'].values
            else:
                x_hist = np.array([])
                y_hist = np.array([])

            x_balsa = df_balsa['ds'].map(pd.Timestamp.timestamp).values
            y_balsa = df_balsa['y'].values

            if df_temperature_forecast is not None and not df_temperature_forecast.empty:
                # Graficar histórico previo en naranja
                if x_hist.size > 0:
                    plot_widget.plot(x_hist, y_hist, pen=pg.mkPen(color='#FFA500', width=2), name="Histórico previo", color='g')

                # Graficar histórico de la balsa en azul
                plot_widget.plot(x_balsa, y_balsa, pen=pg.mkPen(color='b', width=2), name="Histórico balsa", color='b')

                # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
                self.x_forecast = df_temperature_forecast['ds'].map(pd.Timestamp.timestamp).values
                self.y_forecast = df_temperature_forecast['yhat'].values

                # Guardar referencia al widget
                self._temp_plot_widget = plot_widget
                
                # Establecer los ticks iniciales
                all_x_hist = np.concatenate([x_hist, x_balsa]) 
                all_y_hist = np.concatenate([y_hist, y_balsa])
                
                # Dejar una frecuencia de un valor por mes para los ticks de la predicción
                if self.x_forecast.size > 0:
                    # Convertir a fechas pandas para agrupar por mes
                    forecast_dates = pd.to_datetime(df_temperature_forecast['ds'])
                    # Seleccionar el primer valor de cada mes
                    forecast_months = forecast_dates.dt.to_period('M')
                    mask = ~forecast_months.duplicated(keep='first')
                    # Filtrar los arrays para quedarse solo con un valor por mes
                    x_forecast = self.x_forecast[mask.values]    
                # Filtrar los valores de predicción para evitar superposiciones con el histórico
                x_forecast_filtrado = []
                for xf in x_forecast:
                    if not np.any(np.abs(all_x_hist - xf) < 30 * 86400): # 24*60*60 = 86400 segundos 1 dia
                        x_forecast_filtrado.append(xf)
                x_forecast_filtrado = np.array(x_forecast_filtrado)
                # Concatenar los datos históricos y de predicción
                all_x = np.concatenate([all_x_hist, x_forecast_filtrado])                 

                # Convertir la columna 'ds' a formato timestamp si no está ya en datetime
                df_temperature['ds'] = pd.to_datetime(df_temperature['ds'], errors='coerce')
                # Eliminar valores NaT antes de filtrar
                df_temperature = df_temperature.dropna(subset=['ds'])
                # Filtrar los datos de temperatura con la fecha inicial y final de la balsa
                df_temperature = df_temperature[(df_temperature['ds'].dt.date >= raft.getStartDate()) & (df_temperature['ds'].dt.date <= raft.getEndDate())]
            
                # Personalizar los ticks del eje X
                axis = plot_widget.getAxis('bottom')                
                # Cambiar el label del eje X de manera específica
                axis.setLabel("", units="")

                # Crear un ScatterPlotItem para ver los puntos de datos
                scatter = pg.ScatterPlotItem(x=all_x_hist, y=all_y_hist, pen=pg.mkPen(color='k'), brush=pg.mkBrush(255, 255, 255, 120), size=7)
                plot_widget.addItem(scatter)                              

                # Graficar los datos de predicción de temperatura
                plot_widget.plot(self.x_forecast, self.y_forecast, pen=pg.mkPen(color='r', width=2, style=Qt.DashLine), name="Predicción", color='k')

                # Conectar la función para actualizar los ticks cuando cambia el rango
                plot_widget.getViewBox().sigRangeChanged.connect(lambda vb, range_vals: self._update_temperature_axis_ticks(all_x,range_vals[0]))
                self._update_temperature_axis_ticks(all_x,[all_x.min(), all_x.max()])

                # Ajustar los rangos de los ejes de manera dinámica
                min_x = min(all_x_hist.min(), self.x_forecast.min())
                max_x = max(all_x_hist.max(), self.x_forecast.max())
                min_y = min(all_y_hist.min(), self.y_forecast.min())
                max_y = max(all_y_hist.max(), self.y_forecast.max())
            else:
                # Graficar histórico previo en naranja
                if x_hist.size > 0:
                    plot_widget.plot(x_hist, y_hist, pen=pg.mkPen(color='#FFA500', width=2), name="Histórico previo", color='g')

                # Graficar histórico de la balsa en azul
                if x_balsa.size > 0:
                    plot_widget.plot(x_balsa, y_balsa, pen=pg.mkPen(color='b', width=2), name="Histórico balsa", color='b')

                # Guardar referencia al widget y a los valores de X para actualizaciones posteriores
                self._temp_plot_widget = plot_widget
                if x_balsa.size > 0:                         
                    self._temp_x_values = np.concatenate([x_hist, x_balsa]) if x_hist.size > 0 else x_balsa
                else:
                    self._temp_x_values = x_hist

                # Conectar la función para actualizar los ticks cuando cambia el rango
                plot_widget.getViewBox().sigRangeChanged.connect(lambda vb, range_vals: self._update_temperature_axis_ticks(self._temp_x_values,range_vals[0]))

                # Establecer los ticks iniciales
                if x_balsa.size > 0:
                    all_x = np.concatenate([x_hist, x_balsa]) if x_hist.size > 0 else x_balsa
                    all_y = np.concatenate([y_hist, y_balsa]) if y_hist.size > 0 else y_balsa
                elif x_hist.size > 0:
                    all_x = x_hist
                    all_y = y_hist
                else:
                    all_x = np.array([])
                    all_y = np.array([])                    

                if all_x.size > 0:    
                    self._update_temperature_axis_ticks(self._temp_x_values,[all_x.min(), all_x.max()])

                # Convertir la columna 'ds' a formato timestamp si no está ya en datetime
                df_temperature['ds'] = pd.to_datetime(df_temperature['ds'], errors='coerce')
                # Eliminar valores NaT antes de filtrar
                df_temperature = df_temperature.dropna(subset=['ds'])
                # Filtrar los datos de temperatura con la fecha inicial y final de la balsa
                df_temperature = df_temperature[(df_temperature['ds'].dt.date >= raft.getStartDate()) & (df_temperature['ds'].dt.date <= raft.getEndDate())]
            
                # Personalizar los ticks del eje X
                axis = plot_widget.getAxis('bottom')                
                # Cambiar el label del eje X de manera específica
                axis.setLabel("", units="")

                # Crear un ScatterPlotItem para ver los puntos de datos
                scatter = pg.ScatterPlotItem(x=all_x, y=all_y, pen=pg.mkPen(color='k'), brush=pg.mkBrush(255, 255, 255, 120), size=7)
                plot_widget.addItem(scatter)
                self.x_forecast = np.array([])
                # Si no hay predicción, usar solo los datos históricos
                if all_x.size > 0:
                    min_x = all_x.min()
                    max_x = all_x.max()
                    min_y = all_y.min()
                    max_y = all_y.max()
                else:
                    min_x = 0
                    max_x = 1
                    min_y = 0
                    max_y = 1

            # Ajustar el zoom del gráfico para que se ajuste a los datos
            plot_widget.setXRange(min_x, max_x, padding=0.1)
            plot_widget.setYRange(min_y, max_y, padding=0.1)

            # Establecer color negro para la leyenda
            for item in legend.items:
                label = item[1]
                texto_original = label.text
                label.setText(texto_original, color='k')                 

            # Añadir línea vertical para la fecha actual (usa un color diferente)
            self.date_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='b', width=2, style=Qt.DashLine))
            initial_pos = None
            if self.x_forecast.size > 0:
                initial_pos = self.x_forecast[0]
            elif len(x_balsa)>0:
                initial_pos = x_balsa[0]  # Usa el primer valor histórico si no hay predicción

            if not initial_pos is None:
                self.date_vline.setPos(initial_pos)
                # Añadir línea vertical para la predicción con la fecha actual (usa un color diferente)                
                self.date_vline_forescast = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='r', width=2, style=Qt.DashLine))
                self.date_vline_forescast.setPos(initial_pos)
                # Añadir la líneas verticales al gráfico
                plot_widget.addItem(self.date_vline_forescast)
                plot_widget.addItem(self.date_vline)                
        
        self._view.centralwidget.layout().addWidget(plot_widget,pos_i,pos_j)        

    # Cargar las balsas marinas
    def load_rafts_from_controller(self):        
        if not self.raftCon.load_rafts():    
            self.lastError = self.raftCon.lastError        

    # Diálogo auxiliar para seleccionar una opción de una lista
    def aux_list_dialog(self, data, title, message):
        dialog = OptionsDialog(data,title, message)
        # Ajustar el tamaño de la ventana según la longitud máxima de las cadenas
        max_len = max((len(str(item)) for item in data), default=60)
        width = min(max(500, max_len * 10), 1800)  # Ajusta el factor y el máximo según necesidad
        dialog.resize(width, 400)        
        if dialog.exec() == QDialog.Accepted:
            return dialog.get_selected_option()
        
    # Método para cargar datos de temperatura de un archivo
    def load_dataTemperature_from_file(self, file_type, filepath, separator):
        # Cargar los datos de temperatura desde un archivo CSV
        if file_type == "csv":            
                if self.dataLoader.load_from_csv(filepath, separator):
                    if self.tempModel.parseTemperature(self.dataLoader.getData()):
                        return True
                    else:
                        self.lastError = cfg.DASHBOARD_TEMPERATURE_PARSE_ERROR+ ":" + filepath
                        return False
                else:
                    self.lastError = self.dataLoader.lastError + ":" + filepath
                    return False
        elif file_type == "json":
            # Implementar la carga de datos desde un archivo JSON
            pass
        elif file_type == "excel":
            # Implementar la carga de datos desde un archivo Excel 
            pass

    # Método para cargar datos de precio de un archivo
    def load_dataPrice_from_file(self, file_type, filepath, separator):
        # Cargar los datos de temperatura desde un archivo CSV
        if file_type == "csv":            
                if self.dataLoader.load_from_csv(filepath, separator):
                    if self.priceModel.parsePrice(self.dataLoader.getData()):
                        if self.priceModel.smoothPriceMonthly():
                            return True
                        else:
                            self.lastError = self.priceModel.lastError
                            return False
                    else:
                        self.lastError = cfg.DASHBOARD_PRICE_PARSE_ERROR+ ":" + filepath
                        return False
                else:
                    self.lastError = self.dataLoader.lastError + ":" + filepath
                    return False
        elif file_type == "json":
            # Implementar la carga de datos desde un archivo JSON
            pass
        elif file_type == "excel":
            # Implementar la carga de datos desde un archivo Excel 
            pass

# Clase para el hilo de búsqueda de parámetros óptimos del predictor de precios
# Esta clase se encarga de ejecutar la búsqueda de parámetros óptimos en un hilo separado para no bloquear la interfaz de usuario.  
class PricePredictorSearchWorker(QThread):
    # Señales para comunicación con la interfaz principal
    print_message = Signal(str)
    progress_updated = Signal(int)
    status_updated = Signal(str)
    result_found = Signal(int, dict)
    finished_signal = Signal(bool, str)
    
    def __init__(self, price_model, percent, start_date, end_date, prev_start_date, n_iterations):
        super().__init__()
        self.price_model = price_model
        self.percent = percent
        self.start_date = start_date
        self.end_date = end_date
        self.prev_start_date = prev_start_date
        self.n_iterations = n_iterations
        self.should_stop = False
        self.mutex = QMutex()
        
    def stop(self):
        """Método para detener el hilo de forma segura"""
        with QMutexLocker(self.mutex):
            self.should_stop = True
    
    def run(self):
        """Método principal que se ejecuta en el hilo separado"""
        try:
            self.status_updated.emit("Preparando datos para optimización...")
            
            # 1. Preparar datos usando el modelo
            data_result = self.price_model.prepare_data_for_optimization(
                self.percent, self.start_date, self.end_date, self.prev_start_date
            )
            
            if data_result is None:
                # Obtener el error detallado del modelo
                detailed_error = self.price_model.lastError
                self.finished_signal.emit(False, f"Error preparando datos:\n\n{detailed_error}")
                return
            
            train_data, test_data = data_result
            
            # 2. Verificar si se debe detener
            with QMutexLocker(self.mutex):
                if self.should_stop:
                    self.finished_signal.emit(False, "Búsqueda cancelada por el usuario")
                    return
            
            # 3. Ejecutar optimización con monitoreo de progreso
            self.status_updated.emit("Ejecutando búsqueda de parámetros óptimos...")
            results = self._run_optimization_with_progress(train_data, test_data, progress_callback=self._progress_callback)
            
            if self.should_stop:
                self.finished_signal.emit(False, "Búsqueda cancelada por el usuario")
                return
            
            if results and len(results) > 0:
                # 4. Entrenar modelo final
                self.print_message.emit("Entrenando modelo final con mejores parámetros...")
                self.print_message.emit(f"Mejores parámetros encontrados: {results[0]}")
                success = self.price_model.train_final_model(results[0])
                
                if success:
                    self.finished_signal.emit(True, "Modelo entrenado correctamente")
                    # Guardar los mejores estimadores
                    if self.price_model.save_top_estimators(results):
                        self.print_message.emit("Mejores estimadores guardados correctamente")
                    else:
                        self.finished_signal.emit(False, f"Error guardando mejores estimadores: {self.price_model.lastError}")                    
                else:
                    self.finished_signal.emit(False, f"Error entrenando modelo final: {self.price_model.lastError}")
            else:
                self.finished_signal.emit(False, "No se encontraron configuraciones válidas")
                
        except Exception as e:
            self.finished_signal.emit(False, f"Error durante la búsqueda: {str(e)}")

    # Método de callback para actualizar el progreso
    def _progress_callback(self, progress, result_dict):        
        # Actualizar el mensaje de estado con la iteración actual
        self.status_updated.emit(f"Iteración {progress} de {self.n_iterations}...")
        # Actualizar la barra de progreso
        percent = int(100 * (progress) / self.n_iterations)
        self.progress_updated.emit(percent)
        # Emitir el resultado directamente a la vista
        if result_dict is not None:
            self.result_found.emit(percent, result_dict)

    # Método para ejecutar la optimización con progreso
    def _run_optimization_with_progress(self, train_data, test_data, progress_callback):
        # Ejecutar optimización usando el método del modelo
        results = self.price_model.run_parameter_optimization(
            train_data, test_data, self.n_iterations,
            fixed_stats=None, fixed_windows=None,fixed_params=None,lags=None,
            progress_callback=progress_callback
        )
            
        return results        

# Clase para el diálogo de búsqueda de predictor óptimo para precios
class PredictorSearchDialog(QDialog):
    def __init__(self, parent=None, worker=None):
        super().__init__(parent)
        self.setWindowTitle("Búsqueda de Predictor Óptimo")
        self.setFixedSize(1100, 650)
        self.setModal(True)
        self.worker = worker
        
        # Layout principal
        layout = QVBoxLayout(self)

        # Añadir barra de progreso integrada en la ventana principal
        progress_label = QLabel("📊 PROGRESO DE LA BÚSQUEDA")
        progress_label.setFont(QFont("Arial", 12, QFont.Bold))
        progress_label.setStyleSheet("color: #1E90FF; margin: 10px 0px;")
        layout.addWidget(progress_label)
        
        # Barra de progreso integrada (no popup)
        from PySide6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Iniciando búsqueda... %p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4682B4;
                border-radius: 5px;
                text-align: center;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #4682B4;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Área de texto para mostrar resultados en tiempo real
        results_label = QLabel("🔍 PROGRESO EN TIEMPO REAL")
        results_label.setFont(QFont("Arial", 11, QFont.Bold))
        results_label.setStyleSheet("color: #2E8B57; margin: 10px 0px 5px 0px;")
        layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 9))
        self.results_text.setMaximumHeight(150)  # Reducir altura para dar más espacio al ranking
        layout.addWidget(self.results_text)
        
        # Área para mostrar el ranking de mejores configuraciones
        ranking_label = QLabel("🏆 RANKING DE MEJORES CONFIGURACIONES")
        ranking_label.setFont(QFont("Arial", 12, QFont.Bold))
        ranking_label.setStyleSheet("color: #2E8B57; margin: 10px 0px;")
        layout.addWidget(ranking_label)
        
        self.ranking_text = QTextEdit()
        self.ranking_text.setReadOnly(True)
        self.ranking_text.setFont(QFont("Consolas", 9))
        self.ranking_text.setStyleSheet("""
            QTextEdit {
                background-color: #F0F8FF;
                border: 2px solid #4682B4;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.ranking_text)
        
        # Layout para botones
        button_layout = QVBoxLayout()

        # Botón de cancelar (reemplaza la funcionalidad del popup)
        self.cancel_button = QPushButton("❌ Cancelar Búsqueda")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #FF5252;
            }
        """)
        button_layout.addWidget(self.cancel_button)
        
        # Botón de ayuda (opcional)
        self.help_button = QPushButton("💡 Ayuda con Errores")
        self.help_button.clicked.connect(self.show_help)
        self.help_button.setVisible(False)
        button_layout.addWidget(self.help_button)
        
        # Botón cerrar
        self.close_button = QPushButton("✅ Cerrar")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.accept)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45A049;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
                color: #666666;
            }
        """)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
        # Variables para tracking de mejores resultados
        self.best_results = []  # Lista de mejores configuraciones
        self.max_results = 5    # Número máximo de resultados a mostrar
        self.chosen_config = None  # Configuración elegida para entrenamiento
        self.is_cancelled = False  # Flag para cancelación
        
        # Agregar mensaje inicial
        self.results_text.append("🔍 INICIANDO BÚSQUEDA DE PARÁMETROS ÓPTIMOS")
        self.results_text.append("=" * 60)
        self.ranking_text.append("⏳ Esperando resultados de la búsqueda...")

    # Evento de cierre de la ventana 
    # si el worker no está en ejecución cierra la ventana,
    # si está en ejecución no cierra la ventana para evitar problemas de concurrencia
    def closeEvent(self, event):
        # Detener el worker si sigue activo antes de cerrar la ventana
        if self.worker is not None and self.worker.isRunning():
            auxTools.show_info_dialog("No puedes cerrar la ventana mientras la búsqueda está en curso. Cancela la búsqueda primero.")
            event.ignore() 
        else:
            super().closeEvent(event)

    """Actualizar la barra de progreso integrada"""    
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    """Manejar la cancelación de la búsqueda"""
    def cancel_search(self):
        self.is_cancelled = True
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText("⏳ Cancelando...")
        self.progress_bar.setFormat("Cancelando búsqueda... %p%")
        self.worker.terminate()  # Terminar el hilo de búsqueda
        self.close_button.setEnabled(True)  # Habilitar el botón de cerrar        

    """Actualizar el mensaje de estado en la barra de progreso"""    
    def update_status(self, message):
        self.progress_bar.setFormat(f"{message} %p%")

    def print_mensage(self, message):
        print(message)

    def add_result(self, progress, result):
        """Añadir nuevo resultado y actualizar ranking"""
        # Agregar a la lista de resultados en tiempo real        
        score = result.get('score', 0)
        mae = result.get('mae', 'N/A')
        rmse = result.get('rmse', 'N/A')
        mape = result.get('mape', 'N/A')
        dir_acc = result.get('dir_acc', 'N/A')
        stats = result.get('stats', 'N/A')
        windows = result.get('windows', 'N/A')
        params = result.get('params', {})
        lags = result.get('lags', 'N/A')

        # Mostrar todos los datos en el área de progreso
        self.results_text.append(
            f"✨   {progress}%: Score {score:.6f}\n"
            f"   MAE: {mae} | RMSE: {rmse} | MAPE: {mape} | Dir: {dir_acc}\n"
            f"   Stats: {stats}\n"
            f"   Windows: {windows}\n"
            f"   Params: {params}\n"
            f"   Lags: {lags}\n"
            + "-"*50
        )
        self.best_results.append(result)
        self.best_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        self.results_text.ensureCursorVisible()
        
    def _update_ranking_display(self):
        """Actualizar la visualización del ranking"""
        self.ranking_text.clear()

        if not self.best_results:
            self.ranking_text.append("⏳ Esperando resultados de la búsqueda...")
            return

        self.ranking_text.append("🏆 TOP DE MEJORES CONFIGURACIONES ENCONTRADAS")
        self.ranking_text.append("=" * 80)

        for i, result in enumerate(self.best_results):
            position = i + 1
            score = result.get('score', 0)            
            mae = result.get('mae', 'N/A')
            rmse = result.get('rmse', 'N/A')
            mape = result.get('mape', 'N/A')
            dir_acc = result.get('dir_acc', 'N/A')
            stats = result.get('stats', 'N/A')
            windows = result.get('windows', 'N/A')
            params = result.get('params', {})
            lags = result.get('lags', 'N/A')

            # Medallas para los top 3
            if position == 1:
                medal = "🥇"
            elif position == 2:
                medal = "🥈"
            elif position == 3:
                medal = "🥉"
            else:
                medal = f"#{position}"

            self.ranking_text.append(f"\n{medal} POSICIÓN {position}")
            self.ranking_text.append(f"   Score: {score:.6f}")        
            self.ranking_text.append(f"   MAE: {mae} | RMSE: {rmse} | MAPE: {mape} | Dir: {dir_acc}")
            self.ranking_text.append(f"   Stats: {stats}")
            self.ranking_text.append(f"   Windows: {windows}")
            self.ranking_text.append(f"   Params: {params}")
            self.ranking_text.append(f"   Lags: {lags}")
            self.ranking_text.append("-" * 80)

        self.ranking_text.append("\n🔝 Ranking actualizado. Los mejores resultados están arriba.")
        # Ver la primera línea del ranking        
        cursor = self.ranking_text.textCursor()
        cursor.movePosition(QTextCursor.Start)
        self.ranking_text.setTextCursor(cursor)            

    """Llamado cuando termina la búsqueda"""       
    def search_finished(self, success, message):
        # Ocultar botón de cancelar y habilitar cerrar
        self.cancel_button.setVisible(False)
        self.close_button.setEnabled(True)

        if success:
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat("✅ Búsqueda completada - 100%")
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #4CAF50;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 3px;
                }
            """)
            
            self.results_text.append(f"\n✅ BÚSQUEDA COMPLETADA EXITOSAMENTE")
            self.results_text.append(f"📊 Se evaluaron configuraciones y se encontraron {len(self.best_results)} candidatos")
            self.results_text.append("\n🎯 La búsqueda ha finalizado. Revise el ranking de configuraciones arriba.")
            
            # Mostrar resumen estadístico
            if self.best_results:
                best_score = self.best_results[0]['score']
                worst_score = self.best_results[-1]['score']
                self.results_text.append(f"\n📈 Mejor score encontrado: {best_score:.6f}")
                self.results_text.append(f"📉 Peor score encontrado: {worst_score:.6f}")
                improvement = ((best_score - worst_score) / worst_score * 100) if worst_score > 0 else 0
                self.results_text.append(f"📊 Mejora del mejor vs el peor: {improvement:.2f}%")

            self._update_ranking_display()
        else:
            self.progress_bar.setFormat("❌ Error en la búsqueda")
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 2px solid #FF6B6B;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: #FF6B6B;
                    border-radius: 3px;
                }
            """)
            
            # Para errores, mostrar información detallada
            self.results_text.append(f"\n❌ ERROR EN LA BÚSQUEDA:")
            self.results_text.append("=" * 70)
            
            # Formatear el mensaje de error
            error_lines = message.split('\n')
            for line in error_lines:
                if line.strip():
                    if line.startswith('Sugerencias:'):
                        self.results_text.append(f"\n💡 {line}")
                    elif line.startswith('- '):
                        self.results_text.append(f"   {line}")
                    else:
                        self.results_text.append(f"   {line}")
            
            self.results_text.append("=" * 70)
            self.results_text.append("\n📋 ACCIONES RECOMENDADAS:")
            self.results_text.append("   • Revise las sugerencias mostradas arriba")
            self.results_text.append("   • Ajuste los parámetros de la balsa según sea necesario")
            self.results_text.append("   • Intente la búsqueda nuevamente después de los ajustes")
            
            # Mostrar botón de ayuda para errores
            self.help_button.setVisible(True)
            
        
        self.results_text.ensureCursorVisible()
    
    def show_help(self):
        """Mostrar ayuda adicional para resolver errores"""
        help_text = """
        GUÍA DE SOLUCIÓN DE PROBLEMAS:

        🔧 DATOS INSUFICIENTES:
                • Verifique que tiene al menos 10 registros de precios
                • Amplíe el rango de fechas de la balsa
                • Cargue más datos históricos de precios

        📅 DIVISIÓN TRAIN/TEST:
                • Use al menos 33% del rango para entrenamiento
                • Ajuste el slider de fecha actual hacia la derecha
                • Evite fechas muy tempranas que dejen pocos datos

        📊 CALIDAD DE DATOS:
                • Revise el formato de fechas en los archivos CSV
                • Elimine registros con valores faltantes
                • Asegúrese de que los precios sean números válidos

        ⚙️ CONFIGURACIÓN BALSA:
                • Verifique las fechas de inicio y fin
                • Asegúrese de que hay datos en ese rango
                • Considere usar un rango de fechas más amplio
        """
        auxTools.show_info_dialog(help_text)