"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import json
import config
import os
from  model.seaRaft import seaRaft

class raftConfigController:
    def __init__(self,view):
        self.rafts = []        
        # Usar la constante específica para balsas
        self.config_file = config.RAFTS_CONFIG_FILE
        # Inicializar el último error
        self.lastError = None
        self.view = view
    
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