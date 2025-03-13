"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from PySide6.QtWidgets import QLabel,QDialog, QVBoxLayout, QPushButton, QListWidget
import config as cfg

# Controlodador de la vista de dashboard
class dashBoardController:
    def __init__(self,view,raftController):
        self._view = view        
        self.lastError = None
        self.raftCon = raftController

        # --- Inicialización de la vista ---
        # Crear un QLabel para el mensaje de estado
        self.label_estado = QLabel()
        self._view.statusbar.addPermanentWidget(self.label_estado)         
        # Cargar las balsas marinas
        self.load_rafts_from_controller()

        # --- Conectar señales de la vista con manejadores de eventos ---
        self._view.actionConfigurar.triggered.connect(self.on_raft_config)
        self._view.actionVer.triggered.connect(self.on_raft_view)        
    
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
            self.draw_raft(option)
        else:
            # Mostrar mensaje de error temporal
            self._view.statusbar.showMessage(cfg.DASHBOARD_SELECT_RAFT_ERORR_MESSAGE)            

    def on_raft_config(self):
        self.raftCon.show()
        # Actualizar el mensaje de estado permanente
        self.label_estado.setText(cfg.RAFTS_LOADED_MESSAGE.format(count=self.raftCon.count_rafts()))

    # --- Métodos de la lógica de negocio
    def draw_raft(self,raftName):
        # Mostrar mensaje temporal
        self._view.statusbar.showMessage(cfg.DASHBOARD_RAFT_SELECTED_MESSAGE.format(raftName))
        pass

    # Cargar las balsas marinas
    def load_rafts_from_controller(self):        
        if not self.raftCon.load_rafts():    
            self.lastError = self.raftCon.lastError        

    # Diálogo auxiliar para seleccionar una opción de una lista
    def aux_list_dialog(self, data):
        dialog = OptionsDialog(data)        
        if dialog.exec() == QDialog.Accepted:
            return dialog.get_selected_option()
            

# Diálogo de opciones auxiliar
class OptionsDialog(QDialog):
    def __init__(self, options): 
        super().__init__()       
        self.setWindowTitle(cfg.DASHBOARD_SELECT_RAFT_MESSAGE)
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        self.list_widget.addItems(options)
        layout.addWidget(self.list_widget)
        self.button_box = QVBoxLayout()
        self.select_button = QPushButton(cfg.DASHBOARD_LIST_TITLE, self)
        self.button_box.addWidget(self.select_button)
        layout.addLayout(self.button_box)
        self.select_button.clicked.connect(self.accept)

    def get_selected_option(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            return selected_items[0].text()
        return None    
    