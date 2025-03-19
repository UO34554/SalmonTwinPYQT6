"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from PySide6.QtWidgets import QLabel, QDialog, QFileDialog, QGraphicsView, QGraphicsScene, QWidget, QVBoxLayout
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
from utility.utility import OptionsDialog, auxTools, DataLoader
from datetime import datetime

# Controlodador de la vista de dashboard
class dashBoardController:
    def __init__(self,view,raftController):
        self._view = view        
        self.lastError = None
        self.raftCon = raftController
        self.tempModel = DataTemperature()
        self.dataLoader = DataLoader()
        self.lastRaftName = None        
        # Configura el idioma a español (España) para las fechas
        locale.setlocale(locale.LC_TIME, 'es_ES.UTF-8')
        # --- Inicialización de la vista ---
        # Crear un QLabel para el mensaje de estado
        self.label_estado = QLabel()
        self._view.statusbar.addPermanentWidget(self.label_estado)
        # Cargar las balsas marinas
        self.load_rafts_from_controller()
        self.fish_items = []
        self.fish_count = 50
        self.timer = QTimer()

        # --- Conectar señales de la vista con manejadores de eventos ---
        self._view.actionConfigurar.triggered.connect(self.on_raft_config)
        self._view.actionVer.triggered.connect(self.on_raft_view)
        self._view.actionCSV.triggered.connect(self.on_temperature_load_csv)        
    
    # --- Eventos de la vista ---
    def show(self):
        # Mostrar mensaje permanente de información de las balsas marinas        
        self.label_estado.setText(cfg.RAFTS_LOADED_MESSAGE.format(count=self.raftCon.count_rafts()))
        self._view.show()

    def on_raft_view(self):        
        self.load_rafts_from_controller()
        # Ejemplo de datos que cambian las acciones del menú
        data = self.raftCon.get_name_rafts()
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
            cfg.DASHBOARD_LOAD_FILE_MSG,
            "",
            "CSV Files (*.csv)",
            options=options
            )
        if self.load_data_from_file("csv", file_name[0], ';'):            
            auxTools.show_info_dialog(cfg.DASHBOARD_LOAD_TEMP_FILE_SUCCESS)
            self._save_raft_temperature()
        else:
            auxTools.show_error_message(cfg.DASHBOARD_LOAD_TEMP_FILE_ERROR)

    # --- Métodos de la lógica de negocio
    def _save_raft_temperature(self):
        # Buscar la balsa seleccionada si hubiera
        if self.lastRaftName is not None:
            raft = self.raftCon.get_raft_by_name(self.lastRaftName)
            # Guardar los datos de temperatura en la balsa
            tempData = self.tempModel.getTemperatureData(raft.getSeaRegion())
            if tempData is not None:
                raft.setTemperature(tempData)
                # Actualizar la balsa en la lista de balsas
                if self.raftCon.update_rafts(raft.getId(), tempData):
                    self._draw_raft(self.lastRaftName)            

    def _clear_dashboard(self):
        # Borrar todos los widgets del layout
        for i in reversed(range(self._view.centralwidget.layout().count())): 
            self._view.centralwidget.layout().itemAt(i).widget().deleteLater()

    def _draw_raft(self,raftName):
        # Mostrar mensaje temporal
        self._view.statusbar.showMessage(cfg.DASHBOARD_RAFT_SELECTED_MESSAGE.format(raftName))
        self.lastRaftName = raftName
        # Buscar la balsa seleccionada
        raft = self.raftCon.get_raft_by_name(raftName)
        # Limpiar la vista
        self._clear_dashboard()        
        # Dibujar la balsa
        self._draw_graph_temperature(0,1,raft)
        self._draw_graph_temperature(1,1,raft)
        self._draw_graph_temperature(2,1,raft)
        self._draw_schematic(0,0)
        self._draw_infopanel(1,0,raft)
        self._draw_schematic_3d(2,0)

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
        self.timer.timeout.connect(self._update_fish_positions)
        # Actualizar la posición de los peces cada 500 ms
        self.timer.start(500)  

    def _update_fish_positions(self):
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
            fish.setData(pos=np.array([[new_x, new_y, new_z]]))
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
        support_brush = QBrush(support_color)

        # 1. Estructura Flotante (Círculo principal)
        floating_structure = scene.addEllipse(-cage_radius, -cage_radius,
                                                  2 * cage_radius, 2 * cage_radius,
                                                  pen, float_brush)
        floating_structure.setToolTip("Estructura Flotante Principal")

        # 2. Red de la Jaula
        net = scene.addEllipse(-cage_radius + 10, -cage_radius + 10,
                                     2 * cage_radius - 20, 2 * cage_radius - 20,
                                     pen, net_brush)
        net.setToolTip("Red de la Jaula")

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
            support.setToolTip("Soporte de la Balsa")

        # 4. Anclajes (Ejemplo: pequeños rectángulos en los extremos de los soportes)
        anchor_size = 10
        for i in range(num_supports):
            angle = 360 / num_supports * i
            import math
            x = (support_length + anchor_size) * math.cos(math.radians(angle))
            y = (support_length + anchor_size) * math.sin(math.radians(angle))
            anchor = scene.addRect(x - anchor_size / 2, y - anchor_size / 2,
                                         anchor_size, anchor_size, pen, QBrush(Qt.blue))
            anchor.setToolTip("Anclaje")
        
        self._view.centralwidget.layout().addWidget(view,pos_i,pos_j)
    # --- Fin Grafico 2d ---

    # Datos de la balsa
    def _draw_infopanel(self,pos_i,pos_j,raf):
        # Crear un widget para mostrar información de la balsa
        view = QWidget()
        # Crear un layout vertical para organizar los QLabel
        layout = QVBoxLayout()
        view.setLayout(layout)
        # Mostrar información de la balsa
        lName = QLabel(raf.getName())
        lRegion = QLabel("Región del mar: {0}".format(raf.getSeaRegion()))
        lLocation = QLabel("Ubicación: 12.3456, -78.9012")
        lDepth = QLabel("Profundidad: 10 m")
        # Mostrar las fechas de inicio y fin en formato de idioma castellano
        # Formatear las fechas al idioma castellano
        formatted_start_date = raf.getStartDate().strftime("%d de %B de %Y")
        formatted_end_date = raf.getEndDate().strftime("%d de %B de %Y")
        lFechas = QLabel("Fechas: {0} - {1}".format(formatted_start_date, formatted_end_date))
        if raf.getTemperature().empty:
            lTemperature = QLabel("Temperatura: No disponible")
        else:
            # Calcular la temperatura promedio
            temp = np.mean(raf.getTemperature()['y'])
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

        # Añadir los QLabel al layout
        layout.addWidget(lName)
        layout.addWidget(lRegion)
        layout.addWidget(lLocation)
        layout.addWidget(lDepth)
        layout.addWidget(lFechas)
        layout.addWidget(lTemperature)       

        self._view.centralwidget.layout().addWidget(view,pos_i,pos_j)

    # Formatear los ticks del eje X con el formato 'día/mes/año'
    def _format_date(self, value):
        date = datetime.fromtimestamp(value)
        return date.strftime('%d/%m/%Y')

    # Graficar una serie temporal
    def _draw_graph_temperature(self,pos_i,pos_j,raft):
        # Crear un PlotItem para representar la gráfica
        plot_widget = pg.PlotWidget(title="Temperatura del mar en {0}".format(raft.getSeaRegion()))
        plot_widget.setLabels(left="Grados ºC", bottom="Fechas")
        # Cambiar el label del eje X de manera específica
        plot_widget.getAxis('bottom').setLabel("Fechas", units=None)
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setBackground((0, 0, 0, 140))        

        # Agregar datos al gráfico si existen
        if raft.getTemperature().empty:
            # Mostrar una 'X' roja si no hay datos de temperatura
            plot_widget.plot([0], [0], pen=None, symbol='x', symbolSize=20, symbolPen='r', symbolBrush='r')
        else:
            # Convertir la columna 'ds' a formato timestamp si no está ya en datetime
            df_temperature = raft.getTemperature()
            df_temperature['ds'] = pd.to_datetime(df_temperature['ds'], errors='coerce')

            # Convertir fechas a valores numéricos (timestamps) para pyqtgraph
            x = df_temperature['ds'].map(pd.Timestamp.timestamp)
            y = df_temperature['y']            
            
            # Filtros dinámicos para los ticks
            interval = max(1, len(x) // 7) 
            ticks = [(x[i], self._format_date(x[i])) for i in range(0, len(x), interval)]

            # Personalizar los ticks del eje X
            axis = plot_widget.getAxis('bottom')
            axis.setTicks([ticks])

            # Graficar los datos de temperatura
            plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2))

            # Ajustar los rangos de los ejes de manera dinámica
            plot_widget.setXRange(x.min(), x.max(), padding=0.1)
            plot_widget.setYRange(y.min(), y.max(), padding=0.1)
        
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
        
    # Método para cargar datos de un archivo
    def load_data_from_file(self, file_type, filepath, separator):
        # Cargar los datos de temperatura desde un archivo CSV
        if file_type == "csv":            
                if self.dataLoader.load_from_csv(filepath, separator):
                    if self.tempModel.parseTemperature(self.dataLoader.getData()):
                        return True
                    else:
                        self.lastError = cfg.TEMPERATURE_PARSE_ERROR+ ":" + filepath
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
    