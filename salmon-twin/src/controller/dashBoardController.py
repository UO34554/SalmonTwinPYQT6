"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from PySide6.QtWidgets import QLabel
import config as cfg

# Controlodador de la vista de dashboard
class dashBoardController:
    def __init__(self,view,raftController):
        self._view = view        
        self.lastError = None
        self.raftCon = raftController

        # --- Inicialización de la vista ---        
        # Cargar las balsas marinas
        if not self.raftCon.load_rafts():    
            self.lastError = self.raftCon.lastError        

        # --- Conectar señales de la vista con manejadores de eventos ---
        self._view.actionConfigurar.triggered.connect(self.on_raft_config)
        self._view.actionVer.triggered.connect(self.on_raft_view)        
    
    # --- Eventos de la vista ---
    def show(self):
        # Crear un QLabel para el mensaje de estado
        self.label_estado = QLabel(cfg.RAFTS_LOADED_MESSAGE.format(count=self.raftCon.count_rafts()))
        self._view.statusbar.addPermanentWidget(self.label_estado)        
        self._view.show()

    def on_raft_view(self):        
        pass

    def on_raft_config(self):
        self.raftCon.show_rafts()        

    # Cargar las balsas marinas
    def load_rafts(self,rafts):
        self.rafts = rafts
        
    