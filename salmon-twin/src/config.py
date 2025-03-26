import os

#--- Configuración de la aplicación
APP_NAME = "SalmonTwin"
APP_VERSION = "0.1"
APP_START_MESSAGE = "Iniciando {0} Versión {1}".format(APP_NAME, APP_VERSION)
APP_EXIT_MESSAGE = "Saliendo de la aplicación {0} con código de salida {1}".format(APP_NAME, "{0}")
APP_ERROR_MESSAGE = "Error no controlado de la aplicación: {0}"

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

#--- Datos de configuración de panel de mando
DASHBOARD_SELECT_RAFT_MESSAGE = "Seleccionar balsa marina"
DASHBOARD_LIST_TITLE = "Seleccionar"
DASHBOARD_SELECT_RAFT_ERORR_MESSAGE = "Error al seleccionar la balsa marina"
DASHBOARD_RAFT_SELECTED_MESSAGE = "Datos de la balsa marina: {0}"
DASHBOARD_LOAD_TEMP_FILE_SUCCESS = "Datos de temperatura cargados correctamente."
DASHBOARD_LOAD_TEMP_FILE_ERROR = "Error al cargar los datos de temperatura."
DASHBOARD_LOAD_FILE_MSG = "Cargar archivo de datos de temperatura"
DASHBOARD_NO_TEMP_DATA_ERROR = "No hay datos de temperatura para la balsa seleccionada."
DASHBOARD_PREDICT_TEMP_SUCCESS = "Predicción de temperatura exitosa."
DASHBOARD_PREDICT_TEMP_ERROR = "Error al predecir la temperatura."

#--- Datos de configuración de utilidades
UTILITY_DATA_LOAD_FILE_NOT_FOUND_MESSAGE = "Error al cargar los datos: archivo no encontrado."
UTILITY_DATA_LOAD_EXCEPTION_MENSSAGE = "Error al cargar los datos: "

#--- Datos de configuración de balsas marinas
RAFTS_CONFIG_FILE = os.path.join(CONFIG_DIR, "rafts.json")
UI_RAFTCONFIG_FILE = os.path.join(BASE_DIR, "ui", "raftConfig.ui")
RAFTS_LOADED_MESSAGE = "Se cargaron {count} balsas marinas."
RAFTS_JSON_DECODE_ERROR_MESSAGE = "Error al decodificar el archivo de configuración JSON: {error}"
RAFTS_LOAD_ERROR_MESSAGE = "Error al cargar el archivo de configuración: {error}"
RAFTS_EMPTY_CONFIG_MESSAGE = "No se encontraron balsas marinas en el archivo de configuración."
RAFTS_EMPTY_CONFIG_CREATED_MESSAGE = "Se creó un archivo de configuración vacío para balsas marinas."
RAFTS_EMPTY_CONFIG_ERROR_MESSAGE = "Error al crear un archivo de configuración vacío: {error}"
RAFTS_ID_NOT_FOUND = "La balsa con id={id} no existe"
RAFTS_SAVE_ERROR_MESSAGE = "Error al guardar los datos de la balsa marina: {error}"
RAFTS_NAME_ERROR_MESSAGE = "El nombre de la balsa marina no puede estar vacío."

#--- Cadenas para el modelo de temperatura marina en seaTemperature.py
PARSER_ERROR_COLUMN_NAME_NOT_FOUND = "No se encontró la columna {columnName} en los datos de temperatura."
TEMPERATURE_PARSE_ERROR = "Error de parseo al procesar los datos de temperatura."
FIT_TEMPERATURE_ERROR_REGION_NOT_FOUND = "El parámetro Region no se encuentra al predecir la temperatura."
REGION_NOT_FOUND = "La región {0} no existe en los datos de temperatura."
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
# Configuración del log (NO SE USA)
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