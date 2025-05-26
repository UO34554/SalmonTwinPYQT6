"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from PySide6.QtWidgets import QLabel, QDialog, QFileDialog, QGraphicsView, QGraphicsScene, QWidget, QSlider, QGridLayout
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPen, QBrush, QColor, QFont
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
            percent = raft.getPerCentage()
            delta_days = (raft.getEndDate() - raft.getStartDate()).days
            days = int(delta_days * percent / 1000)
            fecha_actual = raft.getStartDate() + timedelta(days)
            df_temperature = df_temperature[(df_temperature['ds'].dt.date >= raft.getStartDate()) & 
                                            (df_temperature['ds'].dt.date <= fecha_actual)]            
        
            # Parámetros del modelo Thyholdt (estos valores pueden ser ajustados según tus necesidades)
            alpha = 7000.0                                  # Peso máximo asintótico en gramos (7kg)
            beta = 0.02004161                               # Coeficiente de pendiente
            mu = 17.0                                       # Punto de inflexión en meses
            mortality_rate = 0.05                           # Tasa mensual de mortandad (5%)
            initial_weight = 100.0                          # Peso inicial del salmón en gramos (100g)            
            initial_number_fishes = raft.getNumberFishes()  # Cantidad inicial de peces
            
            # Aplicar el modelo de crecimiento de Thyholdt devuelve el peso en KG
            df_forecast_temperature = raft.getTemperatureForecast()
            if df_forecast_temperature is None or df_forecast_temperature.empty:
                # Mostrar un mensaje de error temporal
                auxTools.show_error_message(cfg.DASHBOARD_NO_TEMP_FORECAST_DATA_ERROR)
                return
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
        option = self.aux_list_dialog(data)
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
        option = self.aux_list_dialog(data)
        # Mostrar un diálogo para seleccionar una balsa
        if not option:
            auxTools.show_error_message(cfg.DASHBOARD_SELECT_RAFT_ERORR_MESSAGE)
            return None       
        else:
            # Usar la balsa actualmente seleccionada
            raft = self.raftCon.get_raft_by_name(option)
            return raft

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
            if sliderValue==0:
                auxTools.show_error_message(cfg.DASHBOARD_NO_FORESCAST_PERIOD_ERROR)
                return
            
            raft.setPerCentage(sliderValue)
            perCent = raft.getPerCentage()/1000
            # Filtrar los datos de temperatura de entrenamiento con la fecha inicial y hasta la fecha actual
            delta_days = (raft.getEndDate() - raft.getStartDate()).days
            forescast_start_date = raft.getStartDate() + timedelta(delta_days * perCent)
            # Dias de predicción
            forescastDays = (raft.getEndDate() - forescast_start_date).days 
            dataTemp = dataTemp[dataTemp['ds'].apply(lambda x: pd.Timestamp(x) <= pd.Timestamp(forescast_start_date))]            
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
        if not self.priceModel.setPriceData(raft.getPriceData()):
            auxTools.show_error_message(cfg.DASHBOARD_PREDICT_PRICE_ERROR.format(error=self.priceModel.lastError))
            return
        # Llamar al método fit_price con las fechas específicas
        if self.dateSliderCurrent is None:
            sliderValue = 0
            auxTools.show_error_message(cfg.DASHBOARD_NO_FORESCAST_PERIOD_ERROR)
            return
        else:
            sliderValue = self.dateSliderCurrent.value()

        # Dias de predicción
        perCent = raft.getPerCentage()/1000
        if self.priceModel.fit_price(perCent,start_date, end_date, False):
            # Guardar los datos de precios en la balsa
            raft.setPerCentage(sliderValue)           
            raft.setPriceForecast(self.priceModel.getPriceDataForecast())
            # Actualizar la balsa en la lista de balsas
            if self.raftCon.update_rafts_price_forecast(raft):
                auxTools.show_info_dialog(cfg.DASHBOARD_PREDICT_PRICE_SUCCESS)
        else:            
            auxTools.show_error_message(cfg.DASHBOARD_PREDICT_PRICE_ERROR.format(error=self.priceModel.lastError))

    # --- Métodos de la lógica de negocio
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
        
        # Convertir la columna 'timestamp' a formato datetime si no está ya
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'], errors='coerce')
        # Eliminar valores NaT antes de filtrar
        price_data = price_data.dropna(subset=['timestamp'])
        
        # Filtrar los datos de precio entre las fechas de la balsa
        filtered_price = price_data[(price_data['timestamp'].dt.date >= start_date) & 
                                   (price_data['timestamp'].dt.date <= end_date)]
        
        if filtered_price.empty:
            # Mostrar una 'X' roja si no hay datos en el rango de fechas
            plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
            plot_widget.setTitle("Precio del Salmón (sin datos en el rango seleccionado)")
        else:
            # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
            x = filtered_price['timestamp'].map(pd.Timestamp.timestamp).values
            y = filtered_price['EUR_kg'].values

            # Graficar los datos históricos de precio
            plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2),name="Histórico")
            
            # Guardar referencia al widget y a los valores de X para actualizaciones posteriores
            self._price_plot_widget = plot_widget
            self._price_x_values = x

            # Conectar la función para actualizar los ticks cuando cambia el rango
            # Se filtra el objeto vb: El objeto ViewBox que cambió que se pasa como argumento al no ser necesario
            # Se conecta la señal sigRangeChanged a la función _update_temperature_axis_ticks
            plot_widget.getViewBox().sigRangeChanged.connect(lambda vb, range_vals: self._update_price_axis_ticks(range_vals))

            # Establecer los ticks iniciales
            self._update_price_axis_ticks([[x.min(), x.max()], [y.min(), y.max()]])
            
            # Crear un ScatterPlotItem para ver los puntos de datos
            scatter = pg.ScatterPlotItem(x=x, y=y, pen=pg.mkPen(color='k'), brush=pg.mkBrush(255, 255, 255, 120), size=7)
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
                min_x = min(x.min(), x_forecast.min())
                max_x = max(x.max(), x_forecast.max())
                min_y = min(y.min(), y_forecast.min())
                max_y = max(y.max(), y_forecast.max())
                
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
        view.setBackgroundColor(pg.mkColor(0, 0, 80, 255))
        # Configurar el rango inicial de la cámara
        view.setCameraPosition(distance=40, elevation=20, azimuth=45)
        # Agregar una cuadrícula para referencia
        grid = gl.GLGridItem()
        grid.setColor((150,150,150,255))
        grid.scale(1, 1, 1)
        view.addItem(grid)
        # Dibujar la estructura circular de la balsa
        self._create_balsa(view)
        # Dibujar las redes bajo el agua
        self._create_nets(view)
        # Dibujar flotadores alrededor de la balsa
        self._create_flotadores(view)
        # Dibujar peces dentro de la red
        self._create_fish(view,raft)
        # Mostrar el widget 3D        
        self._view.centralwidget.layout().addWidget(view,pos_i,pos_j)

    # Crear la estructura circular de la balsa
    def _create_balsa(self,view):
        # Estructura de la balsa (círculo)
        radius = 10
        theta = np.linspace(0, 2 * np.pi, 100)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.zeros_like(x)

        # Crear una línea circular
        balsa = gl.GLLinePlotItem(pos=np.array([x, y, z]).T, color=(1, 0, 0, 1), width=2)
        view.addItem(balsa)

    # Crear redes bajo el agua
    def _create_nets(self,view):
        # Red (cilindro bajo la balsa)
        radius = 10
        height = 5
        theta = np.linspace(0, 2 * np.pi, 50)
        z = np.linspace(0, -height, 2)
        bottom_circle = []  # Coordenadas del fondo del cilindro
        for t in theta:
            x = np.array([radius * np.cos(t), radius * np.cos(t)])
            y = np.array([radius * np.sin(t), radius * np.sin(t)])
            line = gl.GLLinePlotItem(pos=np.array([x, y, z]).T, color=(0, 0, 1, 0.5), width=1)
            view.addItem(line)
            # Agregar puntos del círculo inferior
            bottom_circle.append([radius * np.cos(t), radius * np.sin(t), -height])

        # Dibujar el fondo como un círculo cerrado
        bottom_circle = np.array(bottom_circle)
        bottom_circle = np.vstack([bottom_circle, bottom_circle[0]])  # Cerrar el círculo
        bottom = gl.GLLinePlotItem(pos=bottom_circle, color=(0, 0, 1, 0.5), width=1)
        view.addItem(bottom)

    # Crear flotadores (esferas distribuidas en el círculo)
    def _create_flotadores(self,view):
        # Flotadores (esferas distribuidas en el círculo)
        radius = 10
        flotador_positions = 8
        for i in range(flotador_positions):
            angle = 2 * np.pi / flotador_positions * i
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = 0
            sphere = gl.GLScatterPlotItem(pos=np.array([[x, y, z]]), size=20, color=(0.1, 0.6, 0.1, 1))
            view.addItem(sphere)    

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
        
            # Si hay datos de predicción tanto de crecimiento como de precio
            growth_forecast = raft.getGrowthForecastData()
            price_forecast = raft.getPriceForecastData()
        
            if growth_forecast is None or price_forecast is None or growth_forecast.empty or price_forecast.empty:
                return None
            
            # Convertir fechas a datetime
            growth_forecast['ds'] = pd.to_datetime(growth_forecast['ds'])
            price_forecast['ds'] = pd.to_datetime(price_forecast['ds'])
        
            # 1. ENCONTRAR MÁXIMOS LOCALES EN LA CURVA DE BIOMASA
            # Un punto es un máximo local si su valor es mayor que sus puntos adyacentes
            growth_forecast = growth_forecast.copy()
            growth_forecast['is_max'] = False
        
            # Para cada punto (excepto el primero y el último)
            for i in range(1, len(growth_forecast) - 1):
                if (growth_forecast['biomass'].iloc[i] > growth_forecast['biomass'].iloc[i-1] and 
                    growth_forecast['biomass'].iloc[i] > growth_forecast['biomass'].iloc[i+1]):
                    growth_forecast.iloc[i, growth_forecast.columns.get_loc('is_max')] = True
        
            # El último punto también puede ser un máximo
            if len(growth_forecast) > 1 and growth_forecast['biomass'].iloc[-1] > growth_forecast['biomass'].iloc[-2]:
                growth_forecast.iloc[-1, growth_forecast.columns.get_loc('is_max')] = True
        
            # Si no hay máximos locales, usar el punto de máxima biomasa
            if not growth_forecast['is_max'].any():
                max_biomass_idx = growth_forecast['biomass'].idxmax()
                growth_forecast.loc[max_biomass_idx, 'is_max'] = True
        
            # Filtrar solo los máximos locales
            max_points = growth_forecast[growth_forecast['is_max']].copy()
        
            # 2. BUSCAR PRECIOS CORRESPONDIENTES A ESTOS MÁXIMOS
            # Convertir fechas a timestamps numéricos para la interpolación
            max_points_timestamps = max_points['ds'].map(pd.Timestamp.timestamp).values
            price_forecast_timestamps = price_forecast['ds'].map(pd.Timestamp.timestamp).values

            # Ordenar los arrays para interpolación (np.interp requiere que xp esté ordenado)
            if not np.all(np.diff(price_forecast_timestamps) >= 0):
                sort_idx = np.argsort(price_forecast_timestamps)
                price_forecast_timestamps = price_forecast_timestamps[sort_idx]
                price_values = price_forecast['y'].values[sort_idx]
            else:
                price_values = price_forecast['y'].values
            
            # Usar np.interp para interpolación lineal simple usando .loc para evitar warnings
            max_points.loc[:, 'price'] = np.interp(
                max_points_timestamps,         # x: puntos donde interpolar
                price_forecast_timestamps,     # xp: puntos x conocidos 
                price_values                   # fp: valores y conocidos
            )

            # 3. CALCULAR BIOMASA * PRECIO
            max_points.loc[:, 'biomass_price'] = max_points['biomass'] * max_points['price']

            # 4. BUSCAR EL MÁXIMO DE BIOMASA * PRECIO
            max_biomass_price_idx = max_points['biomass_price'].idxmax()
            # Obtener la fecha correspondiente al máximo de biomasa * precio
            optimal_date = max_points.loc[max_biomass_price_idx, 'ds']
        
            # Obtener la biomasa y precio en esa fecha
            biomass = max_points.loc[max_biomass_price_idx, 'biomass']
            price = max_points.loc[max_biomass_price_idx, 'price']
        
            # Buscar el número de peces correspondiente a esa fecha
            # Encontrar la fecha más cercana en growth_forecast
            closest_idx = growth_forecast['ds'].sub(optimal_date).abs().idxmin()
            nFishes = growth_forecast.loc[closest_idx, 'number_fishes']
        
            # Calcular el valor total
            total = biomass * price
            return optimal_date, biomass, price, nFishes, total

        except Exception as e:
            print(f"Error calculando fecha óptima: {str(e)}")
            return None       

    # --- Grafico 2d ---
    def _draw_raft_2D(self, scene, scene_size, cage_radius):
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
        floating_structure = scene.addEllipse(-cage_radius*3, -cage_radius*3,
                                               cage_radius*4,  cage_radius*4,
                                                  pen, float_brush)
        floating_structure.setToolTip(cfg.DASHBOARD_GRAPH_MAINSTRUCTURE_MSG)
        # 2. Red de la Jaula
        net = scene.addEllipse(-cage_radius*3 + 5, -cage_radius*3 + 5,
                                     4 * cage_radius - 10, 4 * cage_radius - 10,
                                     pen, net_brush)
        net.setToolTip(cfg.DASHBOARD_GRAPH_NET_MSG)
        # 3. Soportes (Ejemplo: líneas radiales)
        num_supports = 8
        support_length = cage_radius + 15
        for i in range(num_supports):
            angle = 360 / num_supports * i            
            x1 = -cage_radius
            y1 = -cage_radius
            x2 = x1 + support_length * math.cos(math.radians(angle))
            y2 = y1 + support_length * math.sin(math.radians(angle))
            support = scene.addLine(x1, y1, x2, y2, QPen(support_color, 2))
            support.setToolTip(cfg.DASHBOARD_GRAPH_PILLARS_MSG)
        # 4. Anclajes (Ejemplo: pequeños rectángulos en los extremos de los soportes)
        anchor_size = 5
        for i in range(num_supports):
            angle = 360 / num_supports * i            
            x = -cage_radius + (support_length + anchor_size) * math.cos(math.radians(angle))
            y = -cage_radius + (support_length + anchor_size) * math.sin(math.radians(angle))
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
            if isCrrent:
                view.setMaximumWidth(350)
                view.setMaximumHeight(350)
                scene_size = 200
                cage_radius = scene_size / 10
                font_name_size = 14
                font_title_size = 12
                font_info_size = 11
                multiplier_pos = 5
                self._draw_raft_2D(scene,scene_size,cage_radius)        
                view.setScene(scene)       
            else:
                view.setMaximumWidth(270)
                view.setMaximumHeight(270)
                scene_size = 150
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
            if result is not None:
                date, biomass, price, nFishes, total = result
                optimal_date = date.strftime("%d/%m/%Y") if hasattr(date, 'strftime') else "N/A"
                expected_value = total if total is not None else 0
            else:
                # Valores predeterminados cuando no se puede calcular
                optimal_date = "N/A"
                expected_value = 0

            # Nombre de la balsa
            raft_name = scene.addText(raft.getName(), QFont("Arial", font_name_size, QFont.Bold))
            raft_name.setDefaultTextColor(QColor(0, 0, 0))
            raft_name.setPos(-cage_radius*4, -cage_radius*5)

            # Título        
            title_text = scene.addText("Información de Cosecha", QFont("Arial", font_title_size, QFont.Bold))
            title_text.setDefaultTextColor(QColor(0, 0, 0))
            title_text.setPos(-cage_radius*multiplier_pos, cage_radius*1.5)
    
            # Valor esperado
            value_text = scene.addText(f"Valor esperado: {expected_value:.2f} EUR", QFont("Arial", font_info_size))
            value_text.setDefaultTextColor(QColor(0, 100, 0))  # Verde
            value_text.setPos(-cage_radius*multiplier_pos, cage_radius*2.5)
    
            # Fecha óptima
            date_text = scene.addText(f"Recogida óptima: {optimal_date}", QFont("Arial", font_info_size))
            date_text.setDefaultTextColor(QColor(0, 0, 150))  # Azul
            date_text.setPos(-cage_radius*multiplier_pos, cage_radius*3.5)
    
            # Solo añadir el view al layout
            if isCrrent:
                grid_layout.addWidget(view, 0, 0, 1, 1)
            else:
                col += 1
                grid_layout.addWidget(view, 0, col, 1, 1)  
        
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
    def _update_temperature_axis_ticks(self, range_vals):
        # Obtener el rango visible actual
        min_x, max_x = range_vals[0]
        x_range = max_x - min_x
    
        # Obtener el ancho del gráfico en píxeles
        plot_width = self._temp_plot_widget.width()
    
        # Calcular cuántos ticks pueden caber basado en el ancho disponible
        # Asumiendo que cada label necesita aproximadamente 130px para ser legible
        max_ticks_by_width = max(3, int(plot_width / 130))
    
        # Filtrar los índices visibles dentro del rango actual
        visible_indices = np.where((self._temp_x_values >= min_x) & (self._temp_x_values <= max_x))[0]
    
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
                ticks = [(self._temp_x_values[i], self._format_date_compact(self._temp_x_values[i])) for i in tick_indices]
            else:
                ticks = [(self._temp_x_values[i], self._format_date(self._temp_x_values[i])) for i in tick_indices]
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
            # Obtener los datos de temperatura de la balsa            
            df_temperature = raft.getTemperature()
            # Convertir la columna 'ds' a formato timestamp si no está ya en datetime
            df_temperature['ds'] = pd.to_datetime(df_temperature['ds'], errors='coerce')
            # Eliminar valores NaT antes de filtrar
            df_temperature = df_temperature.dropna(subset=['ds'])
            # Filtrar los datos de temperatura con la fecha inicial y final de la balsa
            df_temperature = df_temperature[(df_temperature['ds'].dt.date >= raft.getStartDate()) & (df_temperature['ds'].dt.date <= raft.getEndDate())]
            if df_temperature.empty:
                # Mostrar una 'X' roja si no hay datos de temperatura
                plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
                self._view.centralwidget.layout().addWidget(plot_widget,pos_i,pos_j)
                return
            else:
                # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
                x = df_temperature['ds'].map(pd.Timestamp.timestamp).values
                y = df_temperature['y'].values

                # Personalizar los ticks del eje X
                axis = plot_widget.getAxis('bottom')                
                # Cambiar el label del eje X de manera específica
                axis.setLabel("", units="")

                # Crear un ScatterPlotItem para ver los puntos de datos
                scatter = pg.ScatterPlotItem(x=x, y=y, pen=pg.mkPen(color='k'), brush=pg.mkBrush(255, 255, 255, 120), size=7)
                plot_widget.addItem(scatter)

                # Graficar los datos de temperatura
                plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2), name="Histórico", color='k')

                # Guardar referencia al widget y a los valores de X para actualizaciones posteriores
                self._temp_plot_widget = plot_widget
                self._temp_x_values = x

                # Conectar la función para actualizar los ticks cuando cambia el rango
                # Se filtra el objeto vb: El objeto ViewBox que cambió que se pasa como argumento al no ser necesario
                # Se conecta la señal sigRangeChanged a la función _update_temperature_axis_ticks
                plot_widget.getViewBox().sigRangeChanged.connect(lambda vb, range_vals: self._update_temperature_axis_ticks(range_vals))

                # Establecer los ticks iniciales
                self._update_temperature_axis_ticks([[x.min(), x.max()], [y.min(), y.max()]])

                if df_temperature_forecast is not None and not df_temperature_forecast.empty:
                    # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
                    self.x_forecast = df_temperature_forecast['ds'].map(pd.Timestamp.timestamp).values
                    self.y_forecast = df_temperature_forecast['yhat'].values

                    # Graficar los datos de predicción de temperatura
                    plot_widget.plot(self.x_forecast, self.y_forecast, pen=pg.mkPen(color='r', width=2, style=Qt.DashLine), name="Predicción", color='k')

                    # Ajustar los rangos de los ejes de manera dinámica
                    min_x = min(x.min(), self.x_forecast.min())
                    max_x = max(x.max(), self.x_forecast.max())
                    min_y = min(y.min(), self.y_forecast.min())
                    max_y = max(y.max(), self.y_forecast.max())
                else:
                    # Si no hay predicción, usar solo los datos históricos
                    min_x = x.min()
                    max_x = x.max()
                    min_y = y.min()
                    max_y = y.max()

                plot_widget.setXRange(min_x, max_x, padding=0.1)
                plot_widget.setYRange(min_y, max_y, padding=0.1)

                # Establecer color negro para la leyenda
                for item in legend.items:
                    label = item[1]
                    texto_original = label.text
                    label.setText(texto_original, color='k') 

                '''
                # Conectar el evento de movimiento del ratón
                if df_temperature_forecast is not None:
                    def on_mouse_move(event):self._mouse_move_plot(event, plot_widget, x, y, self.y_forecast, vline)
                else:            
                    def on_mouse_move(event):self._mouse_move_plot(event, plot_widget, x, y, None, vline)
                
                # Conectar el evento de movimiento del ratón
                plot_widget.scene().sigMouseMoved.connect(on_mouse_move)
                '''

                # Añadir línea vertical para la fecha actual (usa un color diferente)
                self.date_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='b', width=2, style=Qt.DashLine))
                initial_pos = self.x_forecast[0]
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
    def aux_list_dialog(self, data):
        dialog = OptionsDialog(data,cfg.DASHBOARD_SELECT_RAFT_MESSAGE,cfg.DASHBOARD_LIST_TITLE)        
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
                        return True
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
