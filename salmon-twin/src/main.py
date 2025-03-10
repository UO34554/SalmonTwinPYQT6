# -*- coding: utf-8 -*-
"""
@author: Pedro López Treitiño
"""
import sys
import config as cfg
from PySide6 import QtWidgets
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt
from controller.raftConfigControler import raftConfigController

# Se importan los recursos compilados de salmonResources.qrc
# para compilar: 
# cd salmon-twin
# pyside6-rcc salmonResources.qrc -o src/resources.py
import resources

if __name__ == "__main__":
    loader = QUiLoader()
    app = QtWidgets.QApplication(sys.argv)    
    dashboard_view = loader.load(cfg.UI_DASHBOARD_FILE, None)
    raftConfig_view = loader.load(cfg.UI_RAFTCONFIG_FILE, None)
    raftConfig_control = raftConfigController(raftConfig_view)
    if not raftConfig_control.load_rafts():    
        print(raftConfig_control.lastError)
    raftConfig_control.view.show()
    #dashboard_view.show()    
    #raftConfig_view.show()
    app.exec()