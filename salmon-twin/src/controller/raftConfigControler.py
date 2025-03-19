"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from datetime import datetime
import json
import config as cfg
import os
from model.seaRaft import seaRaft
from utility.utility import auxTools
from PySide6.QtCore import QStringListModel
from PySide6.QtGui import QFont

# Clase controladora de la vista de configuración de balsas
class raftConfigController:
    def __init__(self,view):
        self.rafts = []        
        # Usar la constante específica para balsas
        self.config_file = cfg.RAFTS_CONFIG_FILE
        # Inicializar el último error
        self.lastError = None
        # Asignar la vista a una propiedad privada de la clase
        self._view = view

        # --- Conectar señales de la vista con manejadores de eventos ---        
        # Conectar la señal de clic en un item de la lista con el método on_item_clicked
        self._view.listView.clicked.connect(self._on_item_clicked)
        # Conectar la señal de clic en el botón de guardar con el método save_raft
        self._view.saveRaft.clicked.connect(self._on_saveRaft_clicked)
        # Conectar la se ñal de clic en el bóton de añadir con el método on_addRaft_clicked
        self._view.addRaft.clicked.connect(self._on_addRaft_clicked)
        # Conectar la señal de clic en el botón de eliminar con el método remove_raft
        self._view.removeRaft.clicked.connect(self._on_remove_raft)

        # --- Initialización de la vista ---
        # Rellena combobox con las regiones marítimas
        self._view.regions.addItems(self.get_sea_regions())
        # Limpiar la vista
        self._clear_view()
        # Deshabilitar el botón de eliminar
        self._view.removeRaft.setEnabled(False)
        # Aplicar tipo de letra a los items de la lista
        font = QFont("Segoe UI", 12, QFont.Bold)
        self._view.listView.setFont(font)

    # --- Eventos de la vista ---
    # Mostrar la vista
    def show(self):
        self._show_rafts()        
        self._view.show()
    
    # Manejador de eventos para clic en la lista
    def _on_item_clicked(self, index):
        # Obtener el índice de la lista
        item = index.data()
        # Obtener la balsa seleccionada
        raft = self.get_raft_by_name(item)
        # Mostrar la balsa en la vista
        self._display_raft_details(raft)
        # Habilitar el botón de eliminar
        self._view.removeRaft.setEnabled(True)

    # Manejador de eventos para clic en el botón de guardar
    def _on_saveRaft_clicked(self):
        # Deshabilitar el botón de eliminar
        self._view.removeRaft.setEnabled(False)
        self._save_raft()

    # Manejador de eventos para clic en el botón de añadir
    def _on_addRaft_clicked(self):
        # Deshabilitar el botón de eliminar
        self._view.removeRaft.setEnabled(False)
        self._clear_view()

    # Manejador de eventos para clic en el botón de eliminar
    def _on_remove_raft(self):
        # Deshabilitar el botón de eliminar
        self._view.removeRaft.setEnabled(False)
        self._remove_raft()

    # --- Métodos de la vista ---
    def _clear_view(self):
        self._view.id.setText('')
        self._view.name.setText('')
        self._view.regions.setCurrentIndex(0)
        self._view.initialDate.setDate(datetime.today())
        self._view.finalDate.setDate(datetime.today())

    # Muestra una balsa en la vista
    def _display_raft_details(self, raft:seaRaft):
        if raft:
            self._view.id.setText(str(raft.getId()))
            self._view.name.setText(raft.getName())
            self._view.regions.setCurrentText(raft.getSeaRegion())
            self._view.initialDate.setDate(raft.getStartDate())
            self._view.finalDate.setDate(raft.getEndDate())
        else:
            self._view.id.setText('')
            self._view.name.setText('')
            self._view.regions.setCurrentIndex(0)
            self._view.initialDate.setDate(datetime.today())
            self._view.finalDate.setDate(datetime.today())

    # Muestra las balsas en la vista
    def _show_rafts(self):
        # Crear un modelo de lista
        model = QStringListModel()
        # Obtener la lista existente
        items = model.stringList()
        # Añadir el nuevo elemento        
        for raft in self.rafts:            
            items.append(raft.getName())
        model.setStringList(items)
        # Configurar el modelo en la vista
        self._view.listView.setModel(model)

    # --- Métodos de la lógica de negocio ---
    # Contar el número de balsas
    def count_rafts(self):
        return len(self.rafts)
    
    # Obtener los nombres de las balsas
    def get_name_rafts(self):
        return [raft.getName() for raft in self.rafts]

    # Guardar cambios en la balsa
    def _save_raft(self):
        # Obtener los datos de la vista
        id = self._view.id.text()
        name = self._view.name.text()
        region = self._view.regions.currentText()
        # Convierte QDate a datetime        
        initialDate = self._view.initialDate.date().toPython()
        # Convierte QDate a datetime
        finalDate = self._view.finalDate.date().toPython()
        # Crear un objeto balsa
        raft = self.get_raft_by_id(id)
        # Si la balsa no exite se pregunta si se quiere crear una nueva
        if raft is None:
            # Pregunta si quiere crear una balsa nueva
            if not auxTools.show_question('Crear nueva balsa', '¿Desea crear una nueva balsa?'):
                self._clear_view()
                return
            else:
                raft = seaRaft()
                # Buscar el primer id libre menor en los ids de la lista de balsas y asignar ese id
                # Si no hay ids, se asigna el id 1 por defecto que es el inicial
                existing_ids = sorted([raft.getId() for raft in self.rafts if raft.getId() is not None])
                new_id = 1
                for i in range(1, len(existing_ids) + 2):
                    if i not in existing_ids:
                        new_id = i
                        break
                id = new_id
                # Establecer los datos de la balsa
                raft.setId(id)
                # Mostrar el id en la vista
                self._view.id.setText(str(raft.getId()))
                if not name:
                    auxTools.show_error_message(cfg.RAFTS_NAME_ERROR_MESSAGE)
                    return
                raft.setName(name)
                raft.setSeaRegion(region)
                raft.setStartDate(initialDate)
                raft.setEndDate(finalDate)                
                # Añadir la balsa a la lista
                self.rafts.append(raft)
        else:        
            # Si la balsa existe se pregunta si se quiere sobreescribir
            if not auxTools.show_question('Sobreescribir balsa', '¿Desea sobreescribir la balsa existente?'):                                
                return             
            # Establecer los datos de la balsa        
            if not name:
                auxTools.show_error_message(cfg.RAFTS_NAME_ERROR_MESSAGE)
                return
            raft.setName(name)
            raft.setSeaRegion(region)
            raft.setStartDate(initialDate)
            raft.setEndDate(finalDate)
        # Guardar la lista de balsas
        if not self._save_raft_list_data():
            auxTools.show_error_message(cfg.RAFTS_SAVE_ERROR_MESSAGE.format(error=self.lastError))
        else:
            self._show_rafts()

    # Eliminar una balsa
    def _remove_raft(self):
         # Obtener la balsa seleccionada
        item = self._view.listView.currentIndex().data()
        if item is None:
            auxTools.show_error_message('Seleccione una balsa para eliminar.')
            return
        # Obtener la balsa por su nombre
        raft = self.get_raft_by_name(item)   
        # Si la balsa no existe
        if raft is None:
            auxTools.show_error_message(cfg.RAFTS_ID_NOT_FOUND.format(id=id))
            return
        # Pregrunta si se quiere eliminar la balsa
        if not auxTools.show_question('Eliminar balsa', '¿Desea eliminar la balsa seleccionada?'):
            return
        # Eliminar la balsa de la lista
        self.rafts.remove(raft)
        # Guardar la lista de balsas
        if not self._save_raft_list_data():
            auxTools.show_error_message(cfg.RAFTS_SAVE_ERROR_MESSAGE.format(error=self.lastError))
        else:
            self._show_rafts()

    # Devuelve balsa por su ID
    def get_raft_by_id(self, id:int)->seaRaft:
        for raft in self.rafts:
            try:
                if raft.getId() == int(id):
                    return raft
            except ValueError:
                return None            
        return None

    # Devuelve una balsa por su nombre
    def get_raft_by_name(self, name:str)->seaRaft:
        for raft in self.rafts:
            if raft.getName() == name:
                return raft
        return None

    # Devuelve una lista de las regiones marítimas disponibles.    
    def get_sea_regions(self):        
        return cfg.SEA_REGIONS
    
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
                self.lastError = cfg.RAFTS_JSON_DECODE_ERROR_MESSAGE.format(error=e)
                self.rafts = []
                return False
            except Exception as e:
                self.lastError = cfg.RAFTS_LOAD_ERROR_MESSAGE.format(error=e)
                self.rafts = []
                return False
        else:
            self.lastError = cfg.RAFTS_EMPTY_CONFIG_MESSAGE
            self.rafts = []
            
            # Crear un archivo de configuración vacío
            self._save_empty_config()
            return True

    # Guarda una configuración vacía para inicializar el archivo
    def _save_empty_config(self):        
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            self.lastError = cfg.RAFTS_EMPTY_CONFIG_CREATED_MESSAGE
        except Exception as e:
            self.lastError = cfg.RAFTS_EMPTY_CONFIG_ERROR_MESSAGE.format(error=str(e))    

    # Guarda la lista de balsas en el archivo de configur
    def _save_raft_list_data(self)->bool:        
        try:
            # Convertir las balsas a un diccionario
            raft_data = [raft.to_dict() for raft in self.rafts]
            # Guardar los datos en el archivo de configuración
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(raft_data, f, indent=4)            
            return True
        except Exception as e:
            self.lastError = e
            return False

    # Actualiza las balsas en el archivo de configuración    
    def update_rafts(self, ID, tempData)->bool:
        if not self.load_rafts():
            auxTools.show_error_message(cfg.RAFTS_LOAD_ERROR_MESSAGE.format(error=self.lastError))
        else:
           raft = self.get_raft_by_id(ID)
           if raft is not None:               
               raft.setTemperature(tempData)
               if not self._save_raft_list_data():
                   auxTools.show_error_message(cfg.RAFTS_SAVE_ERROR_MESSAGE.format(error=self.lastError))
               else:
                   return True
        return False
# End of raftConfigController.py
        
    