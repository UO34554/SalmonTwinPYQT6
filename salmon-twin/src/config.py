import os

#--- Configuración de la aplicación
APP_NAME = "SalmonTwin"
APP_VERSION = "0.1"
APP_START_MESSAGE = "Iniciando {0} Versión {1}".format(APP_NAME, APP_VERSION)
APP_EXIT_MESSAGE = "Saliendo de la aplicación {0} con código de salida {1}".format(APP_NAME, "{0}")

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