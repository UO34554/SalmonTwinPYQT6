"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import json
import config
import os
from  model.seaRaft import seaRaft
from PySide6.QtCore import QStringListModel

class raftConfigController:
    def __init__(self,view):
        self.rafts = []        
        # Usar la constante específica para balsas
        self.config_file = config.RAFTS_CONFIG_FILE
        # Inicializar el último error
        self.lastError = None
        self.view = view
        # Conectar la señal de clic en la lista con el manejador de eventos        
        self.view.listView.clicked.connect(self.on_item_clicked)

    # --- Eventos de la vista ---
    # Manejador de eventos para clic en la lista
    def on_item_clicked(self, index):
        # Obtener el índice de la lista
        item = index.data()
        # Obtener la balsa seleccionada
        raft = self.get_raft_by_name(item)
        # Mostrar la balsa en la vista
        self.show_raft(raft)
    
    # --- Métodos de la lógica de negocio ---
    # Devuelve una balsa por su nombre
    def get_raft_by_name(self, name:str)->seaRaft:
        for raft in self.rafts:
            if raft.getName() == name:
                return raft
        return None
    
    # Muestra una balsa en la vista
    def show_raft(self, raft:seaRaft):
        if raft:
            self.view.id.setText(str(raft.getId()))
            self.view.name.setText(raft.getName())
            self.view.region.setText(raft.getSeaRegion())
            self.view.initialDate.setDate(raft.getStartDate())
            self.view.finalDate.setDate(raft.getEndDate())
        else:
            self.view.id.setText('')
            self.view.name.setText('')
            self.view.region.setText('')
            self.view.initialDate.setDate(self.view.initialDate.minimumDate())
            self.view.finalDate.setDate(self.view.finalDate.minimumDate())

    # Devuelve una lista de las regiones marítimas disponibles.    
    def get_sea_regions(self):        
        return config.SEA_REGIONS
    
    # Carga las balsas desde el archivo de configuración JSON
    def load_rafts(self)->bool:                
        # Asegurarse de que el directorio config existe
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        # Si el archivo de configuración existe y tiene un tamaño mayor a 0
        if os.path.exists(self.config_file) and os.path.getsize(self.config_file) > 0:
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    raft_data = json.load(f)
                    self.rafts = []
                    for data in raft_data:
                        raft, self.lastError = seaRaft.from_dict(data)
                        if self.lastError is not None:
                            return False
                        if raft:
                            self.rafts.append(raft)
                return True                
            except json.JSONDecodeError as e:
                self.lastError = config.RAFTS_JSON_DECODE_ERROR_MESSAGE.format(error=e)
                self.rafts = []
                return False
            except Exception as e:
                self.lastError = config.RAFTS_LOAD_ERROR_MESSAGE.format(error=e)
                self.rafts = []
                return False
        else:
            self.lastError = config.RAFTS_EMPTY_CONFIG_MESSAGE
            self.rafts = []
            
            # Crear un archivo de configuración vacío
            self.save_empty_config()
            return True

    # Guarda una configuración vacía para inicializar el archivo
    def save_empty_config(self):        
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            self.lastError = config.RAFTS_EMPTY_CONFIG_CREATED_MESSAGE
        except Exception as e:
            self.lastError = config.RAFTS_EMPTY_CONFIG_ERROR_MESSAGE.format(error=str(e))

    # Muestra las balsas en la vista
    def show_rafts(self):
        # Crear un modelo de lista
        model = QStringListModel()
        # Obtener la lista existente
        items = model.stringList()
        # Añadir el nuevo elemento        
        for raft in self.rafts:            
            items.append(raft.getName())
        model.setStringList(items)
        # Configurar el modelo en la vista
        self.view.listView.setModel(model)
        
    