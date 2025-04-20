"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from PySide6.QtWidgets import QLabel, QDialog, QFileDialog, QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout, QHBoxLayout, QSlider
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPen, QBrush, QColor
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import numpy as np
import pandas as pd
import random
import config as cfg
import locale
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
    
    # --- Eventos de la vista ---
    def show(self):
        # Mostrar mensaje permanente de información de las balsas marinas        
        self.label_estado.setText(cfg.RAFTS_LOADED_MESSAGE.format(count=self.raftCon.count_rafts()))
        self._view.show()

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
            # Implementar la predicción de la temperatura del mar
            data_forecast = self.tempModel.fitTempData(dataTemp,0.8,0.05,True,365)
            if data_forecast is not None:
                raft.setTemperatureForecast(data_forecast)
                # Actualizar la balsa en la lista de balsas
                if self.raftCon.update_rafts_temp_forecast(raft):    
                    auxTools.show_info_dialog(cfg.DASHBOARD_PREDICT_TEMP_SUCCESS)
                else:
                    auxTools.show_error_message(cfg.DASHBOARD_PREDICT_TEMP_ERROR)        

    def on_price_predict(self):
        raft = self.choice_raft_list_dialog()
        if raft is None:
            return
        
        # Obtener las fechas inicial y final de la balsa
        start_date = raft.getStartDate()
        end_date = raft.getEndDate()        
        
        # Llamar al método fit_price con las fechas específicas
        if self.priceModel.fit_price(start_date=start_date, end_date=end_date, horizon_days=365):
            # Guardar los datos de precios en la balsa            
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
    
        # Borrar todos los widgets del layout
        for i in reversed(range(self._view.centralwidget.layout().count())): 
            self._view.centralwidget.layout().itemAt(i).widget().deleteLater()
    
        # Limpiar las referencias a los peces
        self.fish_items = []
        self.fish_positions = []

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
        self.temperature_plot_widget = self._draw_graph_temperature(0,1,raft)        
        self._draw_growth_model(1,1,raft)
        self._draw_price(2,1,raft)
        self._draw_schematic(0,0)
        self._draw_infopanel(1,0,raft)
        self._draw_schematic_3d(2,0)

    # Dibujar el precio del salmón
    def _draw_price(self, pos_i, pos_j, raft):
        # Crear un widget de gráfico de PyQtGraph
        plot_widget = pg.PlotWidget(title="Precio del Salmón")
        plot_widget.setLabels(left="EUR/kg", bottom="Fechas")
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setBackground((0, 0, 0, 140))
    
        # Definir una leyenda para el gráfico
        plot_widget.addLegend()

        # Obtener los datos de precios
        price_data = raft.getPrice()        
        price_data_forescast = raft.getPriceForecast()
    
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
            plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2), 
                             name="Precio Histórico EUR/kg")

            if price_data_forescast is not None and not price_data_forescast.empty:                
                # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
                x_forecast = price_data_forescast['ds'].map(pd.Timestamp.timestamp).values
                y_forecast = price_data_forescast['y'].values

                # Graficar los datos de precio pronosticados
                plot_widget.plot(x_forecast, y_forecast, pen=pg.mkPen(color='y', width=2, style=Qt.DashLine), 
                                 name="Precio Pronosticado EUR/kg")
                
                # Configurar el rango de visualización para mostrar desde la fecha inicial a la fecha final
                min_x = min(x.min(), x_forecast.min())
                max_x = max(x.max(), x_forecast.max())
                min_y = min(y.min(), y_forecast.min())
                max_y = max(y.max(), y_forecast.max())
                
                plot_widget.setXRange(min_x, max_x, padding=0.1)
                plot_widget.setYRange(min_y, max_y, padding=0.1)

                # Filtros dinámicos para los ticks utilizando todas las fechas
                all_x = np.concatenate([x, x_forecast])
                interval = max(1, len(all_x) // 7)
                sorted_x = np.sort(all_x)
                indices = np.linspace(0, len(sorted_x)-1, 7).astype(int)
                ticks = [(sorted_x[i], self._format_date(sorted_x[i])) for i in indices]

                # Personalizar los ticks del eje X
                axis = plot_widget.getAxis('bottom')
                axis.setTicks([ticks])
                axis.setLabel("", units="")

                # Añadir línea vertical para la fecha actual
                self.price_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='g', width=2, style=Qt.DashLine))
                plot_widget.addItem(self.price_vline)

                # Establecer posición inicial
                if len(x) > 0:  # Asegurarse de que hay datos
                    initial_pos = x[0] + (x[-1] - x[0]) * 0.25
                    self.price_vline.setPos(initial_pos)
            else:
                # Si no hay datos de predicción, solo mostrar los históricos
                plot_widget.setXRange(x.min(), x.max(), padding=0.1)
                plot_widget.setYRange(y.min(), y.max(), padding=0.1)
                
                # Configurar los ticks para los datos históricos
                interval = max(1, len(x) // 7)
                indices = np.linspace(0, len(x)-1, 7).astype(int)
                ticks = [(x[i], self._format_date(x[i])) for i in indices]
                
                axis = plot_widget.getAxis('bottom')
                axis.setTicks([ticks])
                axis.setLabel("", units="")
                
                # Añadir línea vertical para la fecha actual
                self.price_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='g', width=2, style=Qt.DashLine))
                plot_widget.addItem(self.price_vline)
                
                if len(x) > 0:
                    initial_pos = x[0] + (x[-1] - x[0]) * 0.25
                    self.price_vline.setPos(initial_pos)

        # Agregar el widget al layout
        self._view.centralwidget.layout().addWidget(plot_widget, pos_i, pos_j)

    # Dibujar el modelo de crecimiento de la balsa
    def _draw_growth_model(self,pos_i,pos_j,raft):        
        # Crear un widget de gráfico de PyQtGraph
        plot_widget = pg.PlotWidget()
        plot_widget.setTitle("Modelo de Crecimiento de Biomasa")
        plot_widget.setLabel("left", "Biomasa (kg)")
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setBackground((0, 0, 0, 140))
    
        # Definir una leyenda para el gráfico
        plot_widget.addLegend()
    
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
            # Filtrar los datos de temperatura con la fecha inicial y final de la balsa
            df_temperature = df_temperature[(df_temperature['ds'].dt.date >= raft.getStartDate()) & 
                                        (df_temperature['ds'].dt.date <= raft.getEndDate())]
        
            if df_temperature.empty:
                # Mostrar una 'X' roja si no hay datos de temperatura
                plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
            else:
                # Parámetros del modelo Thyholdt (estos valores pueden ser ajustados según tus necesidades)
                alpha = 7000.0               # Peso máximo asintótico en gramos (7kg)
                beta = 0.02004161            # Coeficiente de pendiente
                mu = 17.0                    # Punto de inflexión en meses
                mortality_rate = 0.05        # Tasa mensual de mortandad (5%)
                initial_weight = 100.0       # Peso inicial del salmón en gramos (100g)
                initial_number_fishes = 100  # Cantidad inicial de peces
            
                # Aplicar el modelo de crecimiento de Thyholdt devuelve el peso en KG
                growth_data = self.growthModel.thyholdt_growth(df_temperature, 
                                                         alpha, 
                                                         beta, 
                                                         mu, 
                                                         mortality_rate, 
                                                         initial_weight, 
                                                         initial_number_fishes)
            
                # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
                x = growth_data['ds'].map(pd.Timestamp.timestamp).values
                y_biomass = growth_data['biomass'].values
                y_number = growth_data['number_fishes'].values
            
                # Filtros dinámicos para los ticks
                interval = max(1, len(x) // 7) 
                ticks = [(x[i], self._format_date(x[i])) for i in range(0, len(x), interval)]
            
                # Personalizar los ticks del eje X
                axis = plot_widget.getAxis('bottom')
                axis.setTicks([ticks])
                axis.setLabel("", units="")
            
                # Graficar los datos de biomasa, crecimiento individual y número de peces
                plot_widget.plot(x, y_biomass, pen=pg.mkPen(color='g', width=2), name="Biomasa Total (kg)")
                plot_widget.plot(x, y_number, pen=pg.mkPen(color='r', width=2), name="Nº de Peces")

                # Añadir línea vertical para la fecha actual
                self.growth_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='g', width=2, style=Qt.DashLine))
                plot_widget.addItem(self.growth_vline)

                # Establecer posición inicial (25% del rango)
                if 'x' in locals() and x.size > 0:  # Asegurarse de que hay datos
                    initial_pos = x[0] + (x[-1] - x[0]) * 0.25
                    self.growth_vline.setPos(initial_pos)
            
        self._view.centralwidget.layout().addWidget(plot_widget, pos_i, pos_j)

    # --- Grafico 3d ---   
    def _draw_schematic_3d(self,pos_i,pos_j):
        # Crear un widget 3D
        view = gl.GLViewWidget()
        view.setBackgroundColor((55, 43, 38, 0))
        # Configurar el rango inicial de la cámara
        view.setCameraPosition(distance=50)
        # Agregar una cuadrícula para referencia
        grid = gl.GLGridItem()
        grid.scale(1, 1, 1)
        view.addItem(grid)
        # Dibujar la estructura circular de la balsa
        self._create_balsa(view)
        # Dibujar las redes bajo el agua
        self._create_nets(view)
        # Dibujar flotadores alrededor de la balsa
        self._create_flotadores(view)
        # Dibujar peces dentro de la red
        self._create_fish(view)
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
            sphere = gl.GLScatterPlotItem(pos=np.array([[x, y, z]]), size=20, color=(0, 1, 0, 1))
            view.addItem(sphere)    

    # Crear peces (esferas pequeñas dentro de la red)
    def _create_fish(self, view):
        # Crear peces (esferas pequeñas dentro de la red)
        self.fish_items = []
        self.fish_positions = [] 
        self.fish_count = 150  # Número de peces
        for _ in range(self.fish_count):
            x = random.uniform(-7, 7)
            y = random.uniform(-7, 7)
            z = random.uniform(-5, 0)
            fish = gl.GLScatterPlotItem(pos=np.array([[x, y, z]]), size=5, color=(1, 1, 0, 1))
            self.fish_items.append(fish)
            self.fish_positions.append([x, y, z])
            view.addItem(fish)            
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
        if not hasattr(self, 'fish_items') or not hasattr(self, 'fish_positions'):
            return

        for i, fish in enumerate(self.fish_items):
            # Obtener la posición actual
            current_pos = self.fish_positions[i]
            x, y, z = current_pos

            # Calcular pequeños desplazamientos (deltas)
            delta_x = random.uniform(-0.1, 0.1)  # Movimiento suave en X
            delta_y = random.uniform(-0.1, 0.1)  # Movimiento suave en Y
            delta_z = random.uniform(-0.1, 0.1)  # Movimiento suave en Z

            # Limitar posiciones dentro de los límites de la red
            new_x = np.clip(x + delta_x, -7, 7)
            new_y = np.clip(y + delta_y, -7, 7)
            new_z = np.clip(z + delta_z, -5, 0)

            # Actualizar la posición en la lista
            self.fish_positions[i] = [new_x, new_y, new_z]

            # Actualizar la posición del pez
            if fish.visible():
                fish.setData(pos=np.array([[new_x, new_y, new_z]]))
            else:
                print(cfg.DASHBOARD_FISH_3D_ERROR.format(i))
    # --- Fin Grafico 3d ---   

    # --- Grafico 2d ---
    def _draw_schematic(self,pos_i,pos_j):
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

        view.setScene(scene)

        # Dimensiones y colores
        cage_radius = 70
        net_color = QColor(100, 100, 255, 150)  # Azul semitransparente
        float_color = QColor(200, 200, 200)      # Gris claro
        support_color = QColor(150, 150, 150)    # Gris oscuro

        # Pluma y pincel comunes
        pen = QPen(Qt.black)
        net_brush = QBrush(net_color)
        float_brush = QBrush(float_color)        

        # 1. Estructura Flotante (Círculo principal)
        floating_structure = scene.addEllipse(-cage_radius, -cage_radius,
                                                  2 * cage_radius, 2 * cage_radius,
                                                  pen, float_brush)
        floating_structure.setToolTip(cfg.DASHBOARD_GRAPH_MAINSTRUCTURE_MSG)

        # 2. Red de la Jaula
        net = scene.addEllipse(-cage_radius + 10, -cage_radius + 10,
                                     2 * cage_radius - 20, 2 * cage_radius - 20,
                                     pen, net_brush)
        net.setToolTip(cfg.DASHBOARD_GRAPH_NET_MSG)

        # 3. Soportes (Ejemplo: líneas radiales)
        num_supports = 8
        support_length = cage_radius + 30
        for i in range(num_supports):
            angle = 360 / num_supports * i
            import math
            x1 = 0
            y1 = 0
            x2 = support_length * math.cos(math.radians(angle))
            y2 = support_length * math.sin(math.radians(angle))
            support = scene.addLine(x1, y1, x2, y2, QPen(support_color, 2))
            support.setToolTip(cfg.DASHBOARD_GRAPH_PILLARS_MSG)

        # 4. Anclajes (Ejemplo: pequeños rectángulos en los extremos de los soportes)
        anchor_size = 10
        for i in range(num_supports):
            angle = 360 / num_supports * i
            import math
            x = (support_length + anchor_size) * math.cos(math.radians(angle))
            y = (support_length + anchor_size) * math.sin(math.radians(angle))
            anchor = scene.addRect(x - anchor_size / 2, y - anchor_size / 2,
                                         anchor_size, anchor_size, pen, QBrush(Qt.blue))
            anchor.setToolTip(cfg.DASHBOARD_GRAPH_ANCHOR_MSG)
        
        self._view.centralwidget.layout().addWidget(view,pos_i,pos_j)
    # --- Fin Grafico 2d ---

    # Función para actualizar la etiqueta de la fecha cuando cambia el valor del slider
    def _update_current_date(self,value,raft,lcurrentDate):
        start_date = raft.getStartDate()
        end_date = raft.getEndDate()
        # Variable boolean para determinar si se está en el periodo de pronóstico
        # Si no hay datos de pronóstico, no se necesita esta variable
        isForecast = False
        # Si tiene datos de pronóstico, extender un año más allá de la fecha final
        if not raft.getTemperatureForecast().empty:
            max_forecast_date = end_date + timedelta(days=365)
            # Ahora dividimos el rango del slider:
            # - De 0 a 75: periodo histórico (start_date a end_date)
            # - De 75 a 100: periodo de predicción (end_date a max_forecast_date)
            if value <= 75:  # Estamos en el periodo histórico
                # Mapear 0-75 al rango histórico completo
                historical_progress = value / 75
                delta_days = (end_date - start_date).days
                current_day_offset = int(delta_days * historical_progress)
                current_date = start_date + timedelta(days=current_day_offset)
            else:  # Estamos en el periodo de predicción
                # Mapear 75-100 al rango de predicción
                forecast_progress = (value - 75) / 25
                forecast_days = (max_forecast_date - end_date).days
                current_day_offset = int(forecast_days * forecast_progress)
                current_date = end_date + timedelta(days=current_day_offset)
                isForecast = True  # Estamos en el periodo de pronóstico
        else:
            # Si no hay datos de pronóstico, simplemente mapeamos 0-100 al rango histórico
            delta_days = (end_date - start_date).days
            if delta_days > 0:  # Protect against division by zero
                current_day_offset = int(delta_days * (value / 100))
                current_date = start_date + timedelta(days=current_day_offset)

        # Formatear la fecha y actualizar la etiqueta
        formatted_date = current_date.strftime("%d de %B de %Y")
        if isForecast:
            lcurrentDate.setText("Fecha actual (predicción): " + formatted_date)
        else:
            lcurrentDate.setText("Fecha actual: " + formatted_date)            

        # Actualizar la posición de la línea vertical si existe
        if hasattr(self, 'date_vline'):
            self.date_vline.setPos(datetime.combine(current_date, datetime.min.time()).timestamp())

        # Actualizar líneas verticales en todas las gráficas
        timestamp = datetime.combine(current_date, datetime.min.time()).timestamp()
        if hasattr(self, 'date_vline'):
            self.date_vline.setPos(timestamp)
        if hasattr(self, 'growth_vline'):
            self.growth_vline.setPos(timestamp)
        if hasattr(self, 'price_vline'):
            self.price_vline.setPos(timestamp)

        # Actualizar las líneas de predicción
        self._update_forecast_lines(value, raft, isForecast)

    # Datos de la balsa
    def _draw_infopanel(self,pos_i,pos_j,raft):
        # Crear un widget para mostrar información de la balsa
        view = QWidget()
        # Crear un layout vertical para organizar los QLabel
        layout = QVBoxLayout()
        view.setLayout(layout)
        # Mostrar información de la balsa
        lName = QLabel(raft.getName())
        lRegion = QLabel("Región del mar: {0}".format(raft.getSeaRegion()))
        lLocation = QLabel("Ubicación: 12.3456, -78.9012")
        lDepth = QLabel("Profundidad: 10 m")
        # Añadir un slider para simular la fecha actual de la balsa desde la fecha de inicio a la fecha final        
        sliderLayout = QHBoxLayout()
        sliderView = QWidget()
        sliderView.setLayout(sliderLayout)

        # El texto inicial será para la fecha de inicio de la balsa        
        lcurrentDate = QLabel("Fecha actual: " + raft.getStartDate().strftime("%d de %B de %Y"))
        sliderLayout.addWidget(lcurrentDate)

        # Configurar el slider con el rango 0-100
        dateSlider = QSlider(Qt.Horizontal)
        dateSlider.setMinimum(0)
        dateSlider.setMaximum(100)

        # Inicializar el slider al 25%
        dateSlider.setValue(25)
        self._update_current_date(25,raft,lcurrentDate)
       
        # Mostrar las fechas de inicio y fin en formato de idioma castellano        
        formatted_start_date = raft.getStartDate().strftime("%d de %B de %Y")
        formatted_end_date = raft.getEndDate().strftime("%d de %B de %Y")
        lFechas = QLabel("Fechas: {0} - {1}".format(formatted_start_date, formatted_end_date))
        if raft.getTemperature().empty:
            lTemperature = QLabel("Temperatura: No disponible")
        else:
            # Calcular la temperatura promedio
            temp = np.mean(raft.getTemperature()['y'])
            # Mostrar la temperatura promedio            
            lTemperature = QLabel("Temperatura: {0:.2f} °C".format(temp))

        # Estilo para los QLabel
        label_style = """
            QLabel {
                font-size: 18px; /* Tamaño de la letra */
                background-color: rgba(200, 200, 200, 150); /* Fondo semitransparente */
                color: black; /* Color del texto */
                border: 1px solid gray; /* Opcional: borde */
                padding: 5px; /* Margen interno */
            }
        """
        lName.setStyleSheet(label_style)
        lRegion.setStyleSheet(label_style)
        lLocation.setStyleSheet(label_style)
        lDepth.setStyleSheet(label_style)
        lFechas.setStyleSheet(label_style)
        lTemperature.setStyleSheet(label_style)
        lcurrentDate.setStyleSheet(label_style)        

        # Añadir los QLabel al layout
        layout.addWidget(lName)
        layout.addWidget(lRegion)
        layout.addWidget(lLocation)
        layout.addWidget(lDepth)        
        layout.addWidget(lTemperature)
        layout.addWidget(lFechas)
        # Añadir el slider al layout
        layout.addWidget(sliderView)        
        layout.addWidget(dateSlider)

        # Conectar el evento de cambio de valor del slider
        dateSlider.valueChanged.connect(lambda value: self._update_current_date(value, raft, lcurrentDate))

        self._view.centralwidget.layout().addWidget(view,pos_i,pos_j)

    # Formatear los ticks del eje X con el formato 'día/mes/año'
    def _format_date(self, value):
        date = datetime.fromtimestamp(value)
        return date.strftime('%d/%m/%Y')
      
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

    # Graficar una serie temporal
    def _draw_graph_temperature(self,pos_i,pos_j,raft):        
        # Crear un PlotItem para representar la gráfica
        if raft is None:
            region = "------"
        else:
            region = raft.getSeaRegion()    
        plot_widget = pg.PlotWidget(title="Temperatura del mar en {0}".format(region))
        plot_widget.setLabels(left="Grados ºC", bottom="Fechas")        
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setBackground((0, 0, 0, 140))        

        # Agregar datos al gráfico si existen
        if raft is None or raft.getTemperature().empty:
            # Mostrar una 'X' roja si no hay datos de temperatura
            plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
        else:
            # Obtener los datos de predicción de temperatura de la balsa si hay
            if not raft.getTemperatureForecast().empty:
                df_temperature_forecast = raft.getTemperatureForecast()
                # Convertir la columna 'ds' a formato timestamp si no está ya en datetime
                df_temperature_forecast['ds'] = pd.to_datetime(df_temperature_forecast['ds'], errors='coerce')
                # Eliminar valores NaT antes de filtrar
                df_temperature_forecast = df_temperature_forecast.dropna(subset=['ds'])
                # Filtrar los datos de temperatura con la fecha inicial y final de la balsa
                # Pero permitiendo ver un año más allá de la fecha final de la balsa si los datos existen
                # Esto es para que la predicción no se corte en la fecha final de la balsa
                extended_end_date = raft.getEndDate() + timedelta(days=365)
                df_temperature_forecast = df_temperature_forecast[(df_temperature_forecast['ds'].dt.date >= raft.getStartDate()) & 
                                                                  (df_temperature_forecast['ds'].dt.date <= extended_end_date)]
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
            else:
                # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
                x = df_temperature['ds'].map(pd.Timestamp.timestamp).values
                y = df_temperature['y'].values                           
            
                # Filtros dinámicos para los ticks
                interval = max(1, len(x) // 7) 
                ticks = [(x[i], self._format_date(x[i])) for i in range(0, len(x), interval)]

                # Personalizar los ticks del eje X
                axis = plot_widget.getAxis('bottom')
                axis.setTicks([ticks])
                # Cambiar el label del eje X de manera específica
                axis.setLabel("", units="")

                # Crear un ScatterPlotItem para permitir el tooltip
                scatter = pg.ScatterPlotItem(x=x, y=y, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120), size=7)
                plot_widget.addItem(scatter)

                # Crear una línea vertical
                vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='y', style=Qt.DashLine))
                plot_widget.addItem(vline)

                # Graficar los datos de temperatura
                plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2), name="Histórico")
                if df_temperature_forecast is not None and not df_temperature_forecast.empty:
                    # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
                    self.x_forecast = df_temperature_forecast['ds'].map(pd.Timestamp.timestamp).values
                    self.y_forecast = df_temperature_forecast['yhat'].values

                    # Añadir los items de las líneas pero no mostrarlos aún
                    # Se actualizarán cuando se mueva el slider
                    self.forecast_past_line = pg.PlotDataItem([], [], pen=pg.mkPen(color='r', width=2), name="Predicción pasada")
                    self.forecast_future_line = pg.PlotDataItem([], [], pen=pg.mkPen(color='r', width=2, style=Qt.DashLine), name="Predicción futura")

                    plot_widget.addItem(self.forecast_past_line)
                    plot_widget.addItem(self.forecast_future_line)

                    # Graficar los datos de predicción de temperatura
                    #plot_widget.plot(x_forecast, y_forecast, pen=pg.mkPen(color='r', width=2), name="Predicción")  

                # Ajustar los rangos de los ejes de manera dinámica
                plot_widget.setXRange(x.min(), x.max(), padding=0.1)
                plot_widget.setYRange(y.min(), y.max(), padding=0.1)

                # Conectar el evento de movimiento del ratón
                if df_temperature_forecast is not None:
                    def on_mouse_move(event):self._mouse_move_plot(event, plot_widget, x, y, self.y_forecast, vline)
                else:            
                    def on_mouse_move(event):self._mouse_move_plot(event, plot_widget, x, y, None, vline)

                # Conectar el evento de movimiento del ratón
                plot_widget.scene().sigMouseMoved.connect(on_mouse_move)

                # Añadir línea vertical para la fecha actual (usa un color diferente)
                self.date_vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='g', width=2, style=Qt.DashLine))
                plot_widget.addItem(self.date_vline)
        
                # Si es la primera vez, inicializar la línea en la posición del slider (25%)
                if hasattr(self, 'date_vline'):
                    initial_pos = x[0] + (x[-1] - x[0]) * 0.25  # 25% del rango
                    self.date_vline.setPos(initial_pos)
        
        self._view.centralwidget.layout().addWidget(plot_widget,pos_i,pos_j)
        return plot_widget
    
    # Actualizar las líneas de predicción según la posición del slider
    # Separa los datos en pasado y futuro, incluyendo el punto de conexión en ambos
    # y actualiza las líneas de predicción
    def _update_forecast_lines(self, slider_value, raft, isForecast=False):        
        if not hasattr(self, 'x_forecast') or not hasattr(self, 'y_forecast'):
            return
        
        # Calcular la fecha actual según el valor del slider
        start_date = raft.getStartDate()
        end_date = raft.getEndDate()

        # Si estamos en modo de predicción, usar el método modificado de cálculo
        if isForecast:
            max_forecast_date = end_date + timedelta(days=365)
            forecast_progress = (slider_value - 75) / 25
            forecast_days = (max_forecast_date - end_date).days
            current_day_offset = int(forecast_days * forecast_progress)
            current_date = end_date + timedelta(days=current_day_offset)
        else:
            # Comportamiento original para fechas históricas
            if slider_value <= 75:  # Rango histórico
                historical_progress = slider_value / 75
                delta_days = (end_date - start_date).days
                current_day_offset = int(delta_days * historical_progress)
                current_date = start_date + timedelta(days=current_day_offset)
            else:
                # Si no es predicción pero el slider está más allá del 75%, usar end_date
                current_date = end_date
    
        # Convertir a timestamp para comparar con x_forecast
        current_timestamp = datetime.combine(current_date, datetime.min.time()).timestamp()
    
        # Encontrar el índice del punto más cercano a la fecha actual
        closest_index = np.argmin(abs(self.x_forecast - current_timestamp))
    
        # Separar los datos en pasado y futuro, incluyendo el punto de conexión en ambos
        past_indices = np.where(self.x_forecast <= self.x_forecast[closest_index])[0]
        future_indices = np.where(self.x_forecast >= self.x_forecast[closest_index])[0]
    
        # Actualizar las líneas de predicción
        self.forecast_past_line.setData(self.x_forecast[past_indices], self.y_forecast[past_indices])
        self.forecast_future_line.setData(self.x_forecast[future_indices], self.y_forecast[future_indices])

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
