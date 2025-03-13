"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from datetime import datetime
import json
import config
import os
from  model.seaRaft import seaRaft
from PySide6.QtCore import QStringListModel
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QFont

class raftConfigController:
    def __init__(self,view):
        self.rafts = []        
        # Usar la constante específica para balsas
        self.config_file = config.RAFTS_CONFIG_FILE
        # Inicializar el último error
        self.lastError = None
        self.view = view

        # --- Conectar señales de la vista con manejadores de eventos ---
        # Conectar la señal de clic en un item de la lista con el método on_item_clicked
        self.view.listView.clicked.connect(self.on_item_clicked)
        # Conectar la señal de clic en el botón de guardar con el método save_raft
        self.view.saveRaft.clicked.connect(self.on_saveRaft_clicked)
        # Conectar la se ñal de clic en el bóton de añadir con el método on_addRaft_clicked
        self.view.addRaft.clicked.connect(self.on_addRaft_clicked)
        # Conectar la señal de clic en el botón de eliminar con el método remove_raft
        self.view.removeRaft.clicked.connect(self.on_remove_raft)

        # --- Initialización de la vista ---
        # Rellena combobox con las regiones marítimas
        self.view.regions.addItems(self.get_sea_regions())
        # Limpiar la vista
        self.clear_view()
        # Deshabilitar el botón de eliminar
        self.view.removeRaft.setEnabled(False)
        # Aplicar tipo de letra a los items de la lista
        font = QFont("Segoe UI", 12, QFont.Bold)
        self.view.listView.setFont(font)

    # --- Eventos de la vista ---
    # Manejador de eventos para clic en la lista
    def on_item_clicked(self, index):
        # Obtener el índice de la lista
        item = index.data()
        # Obtener la balsa seleccionada
        raft = self.get_raft_by_name(item)
        # Mostrar la balsa en la vista
        self.display_raft_details(raft)
        # Habilitar el botón de eliminar
        self.view.removeRaft.setEnabled(True)

    # Manejador de eventos para clic en el botón de guardar
    def on_saveRaft_clicked(self):
        # Deshabilitar el botón de eliminar
        self.view.removeRaft.setEnabled(False)
        self.save_raft()

    # Manejador de eventos para clic en el botón de añadir
    def on_addRaft_clicked(self):
        # Deshabilitar el botón de eliminar
        self.view.removeRaft.setEnabled(False)
        self.clear_view()

    # Manejador de eventos para clic en el botón de eliminar
    def on_remove_raft(self):
        # Deshabilitar el botón de eliminar
        self.view.removeRaft.setEnabled(False)
        self.remove_raft()

    # --- Métodos de la vista ---
    # Mostrar un mensaje de error
    def show_error_message(self,msg):
        error_dialog = QMessageBox()
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(msg)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.exec()

    def show_question(self,tittle,question):
        # Crea el cuadro de mensaje
        msg_box = QMessageBox()
        msg_box.setWindowTitle(tittle)
        msg_box.setText(question)
        msg_box.setIcon(QMessageBox.Question)        
        # Añade los botones y cambia el texto
        yes_button = msg_box.addButton('Sí', QMessageBox.YesRole)
        msg_box.addButton('No', QMessageBox.NoRole)        
        # Ejecuta el cuadro de diálogo y obtiene la respuesta
        msg_box.exec()

        return msg_box.clickedButton() == yes_button        

    def clear_view(self):
        self.view.id.setText('')
        self.view.name.setText('')
        self.view.regions.setCurrentIndex(0)
        self.view.initialDate.setDate(datetime.today())
        self.view.finalDate.setDate(datetime.today())

    # Muestra una balsa en la vista
    def display_raft_details(self, raft:seaRaft):
        if raft:
            self.view.id.setText(str(raft.getId()))
            self.view.name.setText(raft.getName())
            self.view.regions.setCurrentText(raft.getSeaRegion())
            self.view.initialDate.setDate(raft.getStartDate())
            self.view.finalDate.setDate(raft.getEndDate())
        else:
            self.view.id.setText('')
            self.view.name.setText('')
            self.view.regions.setCurrentIndex(0)
            self.view.initialDate.setDate(datetime.today())
            self.view.finalDate.setDate(datetime.today())

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
        # Mostrar la vista
        self.view.show()

    # --- Métodos de la lógica de negocio ---
    # Guardar cambios en la balsa
    def save_raft(self):
        # Obtener los datos de la vista
        id = self.view.id.text()
        name = self.view.name.text()
        region = self.view.regions.currentText()
        # Convierte QDate a datetime        
        initialDate = self.view.initialDate.date().toPython()
        # Convierte QDate a datetime
        finalDate = self.view.finalDate.date().toPython()
        # Crear un objeto balsa
        raft = self.get_raft_by_id(id)
        # Si la balsa no exite se pregunta si se quiere crear una nueva
        if raft is None:
            # Pregunta si quiere crear una balsa nueva
            if not self.show_question('Crear nueva balsa', '¿Desea crear una nueva balsa?'):
                self.clear_view()
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
                self.view.id.setText(str(raft.getId()))
                if not name:
                    self.show_error_message(config.RAFTS_NAME_ERROR_MESSAGE)
                    return
                raft.setName(name)
                raft.setSeaRegion(region)
                raft.setStartDate(initialDate)
                raft.setEndDate(finalDate)                
                # Añadir la balsa a la lista
                self.rafts.append(raft)
        else:        
            # Si la balsa existe se pregunta si se quiere sobreescribir
            if not self.show_question('Sobreescribir balsa', '¿Desea sobreescribir la balsa existente?'):                                
                return             
            # Establecer los datos de la balsa        
            if not name:
                self.show_error_message(config.RAFTS_NAME_ERROR_MESSAGE)
                return
            raft.setName(name)
            raft.setSeaRegion(region)
            raft.setStartDate(initialDate)
            raft.setEndDate(finalDate)
        # Guardar la lista de balsas
        if not self.save_raft_list_data():
            self.show_error_message(config.RAFTS_SAVE_ERROR_MESSAGE.format(error=self.lastError))
        else:
            self.show_rafts()

    # Eliminar una balsa
    def remove_raft(self):
         # Obtener la balsa seleccionada
        item = self.view.listView.currentIndex().data()
        if item is None:
            self.show_error_message('Seleccione una balsa para eliminar.')
            return
        # Obtener la balsa por su nombre
        raft = self.get_raft_by_name(item)   
        # Si la balsa no existe
        if raft is None:
            self.show_error_message(config.RAFTS_ID_NOT_FOUND.format(id=id))
            return
        # Pregrunta si se quiere eliminar la balsa
        if not self.show_question('Eliminar balsa', '¿Desea eliminar la balsa seleccionada?'):
            return
        # Eliminar la balsa de la lista
        self.rafts.remove(raft)
        # Guardar la lista de balsas
        if not self.save_raft_list_data():
            self.show_error_message(config.RAFTS_SAVE_ERROR_MESSAGE.format(error=self.lastError))
        else:
            self.show_rafts()

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

    # Guarda la lista de balsas en el archivo de configur
    def save_raft_list_data(self)->bool:        
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
        
    