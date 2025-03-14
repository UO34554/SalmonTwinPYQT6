"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from PySide6.QtWidgets import QLabel, QDialog, QFileDialog, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene
import config as cfg
from model.seaTemperature import DataTemperature
from utility.utility import OptionsDialog, auxTools, DataLoader
import pyqtgraph as pg

# Controlodador de la vista de dashboard
class dashBoardController:
    def __init__(self,view,raftController):
        self._view = view        
        self.lastError = None
        self.raftCon = raftController
        self.tempModel = DataTemperature()
        self.dataLoader = DataLoader()

        # --- Inicialización de la vista ---
        # Crear un QLabel para el mensaje de estado
        self.label_estado = QLabel()
        self._view.statusbar.addPermanentWidget(self.label_estado)
        # Cargar las balsas marinas
        self.load_rafts_from_controller()

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
            self.draw_raft(option)
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
        else:
            auxTools.show_error_message(cfg.DASHBOARD_LOAD_TEMP_FILE_ERROR)

    # --- Métodos de la lógica de negocio
    def draw_raft(self,raftName):
        # Mostrar mensaje temporal
        self._view.statusbar.showMessage(cfg.DASHBOARD_RAFT_SELECTED_MESSAGE.format(raftName))
        self.draw_graph(0,0)
        self.draw_graph(0,1)
        self.draw_graph(0,2)
        self.draw_graph(1,0)
        self.draw_graph(1,1)
        self.draw_graph(1,2)
        self.draw_graph(2,0)
        self.draw_graph(2,1)
        self.draw_graph(2,2)

    def draw_graph(self,i,j):
        # Crear un PlotItem para representar la gráfica
        plot_widget = pg.PlotWidget(title="Serie Temporal")
        plot_widget.setLabels(left="Valor", bottom="Tiempo")
        plot_widget.showGrid(x=True, y=True)
        plot_widget.setBackground((0, 0, 0, 140))        

        # Agregar datos al gráfico
        x = [1, 2, 3, 4, 5]
        y = [20, 30, 25, 35, 40]
        plot_widget.plot(x, y, pen=pg.mkPen(color='b', width=2))

        # Ajustar los rangos de los ejes
        plot_widget.setXRange(0, 10, padding=0.1)
        plot_widget.setYRange(10, 50, padding=0.1)        
        
        self._view.centralwidget.layout().addWidget(plot_widget,i,j)

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
    