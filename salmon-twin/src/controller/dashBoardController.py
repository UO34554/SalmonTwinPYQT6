"""
@author: Pedro López Treitiño
Gestor unificado de balsas marinas para el sistema Salmon Twin
"""
# Controlodador de la vista de dashboard
class dashBoardController:
    def __init__(self,view):
        self.view = view
        self.rafts = []
        self.lastError = None

        # --- Conectar señales de la vista con manejadores de eventos ---
        

    # --- Eventos de la vista ---


    # Cargar las balsas marinas
    def load_rafts(self,rafts):
        self.rafts = rafts
        
    