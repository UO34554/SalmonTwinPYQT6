# -*- coding: utf-8 -*-
"""
@author: Pedro López Treitiño
"""
import sys
import config as cfg

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py
#from ui_form import Ui_DashBoard

class DashBoard(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        #self.ui = Ui_DashBoard()
        #self.ui.setupUi(self)
        loader = QUiLoader()
        ui_file = QFile(cfg.UI_DASHBOARD_FILE)
        if not ui_file.exists():           
            QMessageBox.critical(self, "Error", cfg.UI_DASHBOARD_FILE_NOT_FOUND_MESSAGE.format(cfg.UI_DASHBOARD_FILE))
            sys.exit(-1)
        ui_file.open(QFile.ReadOnly)
        self.ui = loader.load(ui_file, self)
        ui_file.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = DashBoard()
    widget.show()
    sys.exit(app.exec())
