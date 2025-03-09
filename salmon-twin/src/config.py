import os

#--- Directorio base de la aplicación (asumiendo que este archivo está en src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#--- Directorios de assets
ASSETS_DIR = os.path.join(BASE_DIR, "..", "assets")
ICONS_DIR = os.path.join(ASSETS_DIR, "icons")
IMAGES_DIR = os.path.join(ASSETS_DIR, "images")

#--- Configuración directorio para archivos de configuración
CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")

#--- Rutas específicas de configuración de la ui
UI_DASHBOARD_FILE = os.path.join(BASE_DIR, "ui", "dashboard.ui")
UI_DASHBOARD_FILE_NOT_FOUND_MESSAGE = "No se encontró el archivo de interfaz de usuario en {0}"

#--- Rutas específicas de configuración
RAFTS_CONFIG_FILE = os.path.join(CONFIG_DIR, "rafts.json")

#--- Textos para la configuración de balsas
RAFT_CONFIG_TITLE = "Configuración de Balsas"
RAFT_CONFIG_ID = "ID:"
RAFT_CONFIG_NAME = "Nombre:"
RAFT_CONFIG_REGION = "Región Marítima:"
RAFT_CONFIG_START_DATE = "Fecha de Inicio:"
RAFT_CONFIG_END_DATE = "Fecha de Finalización:"
RAFT_CONFIG_ADD = "Añadir"
RAFT_CONFIG_DELETE = "Eliminar"
RAFT_CONFIG_SAVE = "Guardar"
RAFT_CONFIG_CANCEL = "Cancelar"
RAFT_CONFIG_IMPORT = "Importar"
RAFT_CONFIG_EXPORT = "Exportar"
RAFT_CONFIG_SUCCESS = "Configuración guardada correctamente"
RAFT_CONFIG_ERROR = "Error al guardar la configuración"
RAFT_CONFIG_FILE_NOT_FOUND ="No se encontró archivo de configuración de balsas. Se iniciará con lista vacía."
RAFTS_LOADED_MESSAGE = "Cargadas {count} balsas desde configuración"
RAFTS_LOAD_ERROR_MESSAGE = "Error al cargar balsas desde configuración: {error}"
RAFTS_SAVED_MESSAGE = "Guardadas {count} balsas en configuración"
RAFTS_SAVE_ERROR_MESSAGE = "Error al guardar balsas en configuración: {error}"
RAFT_ADDED_MESSAGE = "Añadida balsa con ID {id}"
RAFT_UPDATED_MESSAGE = "Actualizada balsa con ID {id}"
RAFT_NOT_FOUND_UPDATE_MESSAGE = "No se encontró balsa con ID {id} para actualizar"
RAFT_DELETED_MESSAGE = "Eliminada balsa con ID {id}"
RAFT_NOT_FOUND_DELETE_MESSAGE = "No se encontró balsa con ID {id} para eliminar"
RAFTS_EXPORTED_MESSAGE = "Balsas exportadas a {file_path}"
RAFTS_EXPORT_ERROR_MESSAGE = "Error al exportar balsas a {file_path}"
RAFTS_IMPORTED_MESSAGE = "Balsas importadas desde {file_path}"
RAFTS_IMPORT_ERROR_MESSAGE = "Error al importar balsas desde {file_path}"
RAFT_CONFIG_CONFIGURED_RAFTS = "Balsas configuradas"
RAFT_CONFIG_DETAILS = "Detalles"
RAFT_CONFIG_ERROR_LOADING_REGIONS = "Error al cargar regiones marítimas"
RAFTS_JSON_DECODE_ERROR_MESSAGE = "Error al decodificar archivo JSON de balsas: {error}"
RAFTS_LOAD_ERROR_MESSAGE = "Error al cargar balsas desde configuración: {error}"
RAFTS_EMPTY_CONFIG_MESSAGE = "No se encontraron balsas en la configuración"
RAFTS_EMPTY_CONFIG_CREATED_MESSAGE = "Archivo de configuración de balsas vacío creado"
RAFTS_EMPTY_CONFIG_ERROR_MESSAGE = "Error al crear archivo de configuración de balsas vacío: {error}"

# Mensajes de log para gestión de balsas
RAFT_CREATED_MESSAGE = "Balsa '{name}' creada en la región '{region}'"
RAFT_CREATE_ERROR_MESSAGE = "Error al crear la balsa: {error}"

#--- Cadenas para la vista de login_view.py
LOGIN_WINDOW_TITLE = "Acceso - Salmon Twin"
USERNAME_LABEL = "Usuario:"
PASSWORD_LABEL = "Contraseña:"
LOGIN_BUTTON_TEXT = "Valida"
INVALID_CREDENTIALS_MESSAGE = "Credenciales inválidas"

#--- Cadenas para la clase de DataLoader en data_loader.py
DATA_LOAD_FILE_NOT_FOUND_MESSAGE = "No se encontró el archivo."
DATA_LOAD_EXCEPTION_MENSSAGE = "Se ha producido un error:"

#--- Cadenas para la clase de DataTemperature en data_temperature.py
TEMPERATURE_PARSE_ERROR = "Error de parsear al procesar los datos de temperatura."
FIT_TEMPERATURE_ERROR_REGION_NOT_FOUND = "El parámetro Region no se encuentra al predecir la temperatura."
# Nombre de las columnas de los datos de temperatura
TEMP_COLUMN_NAMES = [
    'Año',
    'Mes',
    'Finnmark',
    'Troms',
    'Nordland',
    'Nord-Trøndelag',
    'Sør-Trøndelag',
    'Møre og Romsdal',
    'Sogn og Fjordane',
    'Hordaland',
    'Rogaland og Agder'
]
# Se crea un diccionario con el nombre del mes en sueco y el número del mes
SWEDISH_MONTH_NAMES = {
    'Jan': '01',
    'Feb': '02',
    'Mar': '03',
    'Apr': '04',
    'Mai': '05',            
    'Jun': '06',
    'Jul': '07',
    'Aug': '08',
    'Sep': '09',
    'Okt': '10',
    'Nov': '11',
    'Des': '12'
}

# Cadenas para el archivo main.py
MAIN_START_MESSAGE = "Iniciando la aplicación..."
MAIN_EXIT_MESSAGE = "Saliendo de la aplicación... con codigo de resultado:"

# Cadenas para la vista de dashboard de dashboard_view.py
DASHBOARD_WINDOW_TITLE = "Cuadro de Mando - Salmon Twin"
DASHBOARD_WELCOME_MESSAGE = "Bienvenido al cuadro de mando de Salmon Twin"
DASHBOARD_ERROR_DIALOG_TITTLE = "Error"
DASHBOARD_ERROR_DIALOG_MSG = "Se ha producido un error"
DASHBOARD_INFO_DIALOG_TITTLE = "Información"
DASHBOARD_LOAD_FILE_MSG = "Seleccionar fichero de datos"
DASHBOARD_LOAD_FILE_SUCCESS = "Datos cargados correctamente."
DASHBOARD_LOAD_FILE_ERROR = "Error al cargar los datos."

# Cadenas para el modelo de temperatura en data_temperature.py
PARSER_ERROR_COLUMN_NAME_NOT_FOUND = "La columna requerida no se encuentra en los datos:"

# Rutas a archivos específicos
FISH_BREEDING_ICON = os.path.join(ICONS_DIR, "fish-breeding.ico")
FISH_BREEDING_IMAGE = os.path.join(IMAGES_DIR, "fish-breeding.png")
FISH_BACKGROUND = os.path.join(IMAGES_DIR, "salmon-back.jpg")
FISH_BACKGROUND_MED = os.path.join(IMAGES_DIR, "salmon-back-medium.jpg")
FISH_BACKGROUND_MINI = os.path.join(IMAGES_DIR, "salmon-back-mini.jpg")
FISH_RED_BACKGROUND_MINI = os.path.join(IMAGES_DIR, "salmon-red-mini.jpeg")

# Configuración del log
LOG_FILENAME = os.path.join(BASE_DIR, "salmon_monitor.log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "DEBUG"

# Listado de regiones marítimas
SEA_REGIONS = [
    'Finnmark',
    'Troms',
    'Nordland',
    'Nord-Trøndelag',
    'Sør-Trøndelag',
    'Møre og Romsdal',
    'Sogn og Fjordane',
    'Hordaland',
    'Rogaland og Agder'
]

# Se crea un diccionario con el índice de la región y el nombre de la región sueca
INDEX_SEA_REGIONS = {
            0:'Finnmark',
            1:'Troms',
            2:'Nordland',
            3:'Nord-Trøndelag',
            4:'Sør-Trøndelag',
            5:'Møre og Romsdal',
            6:'Sogn og Fjordane',
            7:'Hordaland',
            8:'Rogaland og Agder'
        }