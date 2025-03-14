"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
from PySide6.QtWidgets import QMessageBox, QDialog, QVBoxLayout, QListWidget, QPushButton
import pandas as pd
import config as cfg

# Clase auxiliar con funciones comunes
class auxTools:
    # Método para mostrar un diálogo de información
    @staticmethod
    def show_info_dialog(text):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle("Información")
        msg_box.setText(text)
        msg_box.exec()
    # Función para mostrar un cuadro de diálogo con un mensaje
    @staticmethod
    def show_error_message(msg):
        error_dialog = QMessageBox()
        error_dialog.setWindowTitle("Error")
        error_dialog.setText(msg)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.exec()        
    # Función para mostrar un cuadro de diálogo con una pregunta   
    @staticmethod
    def show_question(tittle,question):
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

# Clase para diálogo de elección de opciones auxiliar
class OptionsDialog(QDialog):
    def __init__(self, options,windowTittle,buttonTitle): 
        super().__init__()       
        self.setWindowTitle(windowTittle)
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        self.list_widget.addItems(options)
        layout.addWidget(self.list_widget)
        self.button_box = QVBoxLayout()
        self.select_button = QPushButton(buttonTitle, self)
        self.button_box.addWidget(self.select_button)
        layout.addLayout(self.button_box)
        self.select_button.clicked.connect(self.accept)

    def get_selected_option(self):
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            return selected_items[0].text()
        return None

class DataLoader:
    def __init__(self):
        self._dataRaw = None
        self.lastError = None        

    def load_from_excel(self, file_path):
        return pd.read_excel(file_path)

    # This method is not implemented
    # Load data from a database
    def load_from_database(self, db_url, query):
        pass

    def load_from_csv(self, file_path, separator):
        try:
            self._dataRaw = pd.read_csv(file_path, sep=separator, encoding='latin1')
            self.lastError = None
            return True
        except FileNotFoundError:
            self.lastError = cfg.UTILITY_DATA_LOAD_FILE_NOT_FOUND_MESSAGE
            return False
        except Exception as e:
            self.lastError = cfg.UTILITY_DATA_LOAD_EXCEPTION_MENSSAGE + e.__str__()
            return False
        
    def getData(self):
        return self._dataRaw   
    
