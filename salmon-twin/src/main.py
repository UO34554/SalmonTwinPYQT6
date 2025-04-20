# -*- coding: utf-8 -*-
"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
import sys
import config as cfg
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader
from controller.raftConfigControler import raftConfigController
from controller.dashBoardController import dashBoardController

# Se importan los recursos compilados de salmonResources.qrc
# para compilar: 
# cd salmon-twin
# pyside6-rcc salmonResources.qrc -o src/resources.py
# *********************
# pip list --outdated --format=json | ConvertFrom-Json | ForEach-Object { pip install --upgrade $_.name }

import resources

if __name__ == "__main__":
    result = 0
    try:
        # --- Configurar la salida estándar ---
        # Se configura la salida estándar para que acepte caracteres UTF-8
        sys.stdout.reconfigure(encoding='utf-8')
        print(cfg.APP_START_MESSAGE)      
        # --- Inicializar la aplicación ---
        # sys.argv es una lista de argumentos pasados al script de Python
        # no se utilizan en este caso
        app = QtWidgets.QApplication(sys.argv)

        # --- Cargar los archivos de UI ---
        loader = QUiLoader()        
        dashboard_view = loader.load(cfg.UI_DASHBOARD_FILE, None)
        raftConfig_view = loader.load(cfg.UI_RAFTCONFIG_FILE, None)

        # --- Instanciar los controladores de las vistas ---
        # Se usa el patrón de diseño MVC
        raftCon = raftConfigController(raftConfig_view)
        dashCon = dashBoardController(dashboard_view,raftCon)

        # --- Mostrar la vista principal ---    
        dashCon.show()

        # --- Ejecutar la aplicación ---
        result=app.exec()
        print(cfg.APP_EXIT_MESSAGE.format(result))
        sys.exit(result)
    except Exception as e:
        print(cfg.APP_ERROR_MESSAGE.format(e))
    
        