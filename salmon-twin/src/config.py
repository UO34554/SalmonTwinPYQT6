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
DASHBOARD_LOAD_INITIAL_DATA_ERROR = "Error al cargar datos iniciales: {error}"
DASHBOARD_LOAD_TEMP_FILE_SUCCESS = "Datos de temperatura cargados correctamente."
DASHBOARD_LOAD_TEMP_FILE_ERROR = "Error al cargar los datos de temperatura."
DASHBOARD_LOAD_TEMP_FILE_MSG = "Cargar archivo de datos de temperatura"
DASHBOARD_LOAD_PRICE_FILE_MSG = "Cargar archivo de datos de precios de salmón"
DASHBOARD_LOAD_PRICE_FILE_SUCCESS = "Datos de precios de salmón cargados correctamente."
DASHBOARD_LOAD_PRICE_FILE_ERROR = "Error al cargar los datos de precios de salmón."
DASHBOARD_NO_TEMP_DATA_ERROR = "No hay datos de temperatura para la balsa seleccionada."
DASHBOARD_NO_TEMP_FORECAST_DATA_ERROR = "No hay datos de predicción de temperatura para la balsa seleccionada."
DASHBOARD_NO_FORESCAST_PERIOD_ERROR = "No se ha seleccionado un periodo de predicción."
DASHBOARD_PREDICT_TEMP_SUCCESS = "Predicción de temperatura exitosa."
DASHBOARD_PREDICT_TEMP_ERROR = "Error al predecir la temperatura. {error}"
DASHBOARD_PREDICT_GROWTH_SUCCESS = "Predicción de crecimiento exitosa."
DASHBOARD_PREDICT_GROWTH_ERROR = "Error al predecir el crecimiento."
DASHBOARD_FISH_3D_ERROR =  "El pez {0} no es visible, anulando la actualización."
DASHBOARD_GRAPH_MAINSTRUCTURE_MSG = "Estructura Flotante Principal"
DASHBOARD_GRAPH_NET_MSG = "Red de la Jaula"
DASHBOARD_GRAPH_PILLARS_MSG = "Soporte de la Balsa"
DASHBOARD_GRAPH_ANCHOR_MSG = "Anclaje"
DASHBOARD_TEMPERATURE_PARSE_ERROR = "Error de parseo al procesar los datos de temperatura."
DASHBOARD_PRICE_PARSE_ERROR = "Error de parseo al procesar los datos de precios."
DASHBOARD_PRICE_DATA_SAVE_OK = "Datos de precios guardados correctamente."
DASHBOARD_PRICE_DATA_SAVE_ERROR = "Error al guardar los datos de precios: {error}"
DASHBOARD_PREDICT_PRICE_SUCCESS = "Predicción de precios exitosa."
DASHBOARD_PREDICT_PRICE_ERROR = "Error al predecir los precios. {error}"
DASHBOARD_DATE_VLINE_HIS_ERROR = "Error al actualizar las líneas verticales de los históricos. {error}"
DASHBOARD_DATE_VLINE_FOR_ERROR = "Error al actualizar las líneas verticales de las predicciones. {error}"
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
RAFTS_ERROR_PARSER_START_DATE = "Error al convertir fecha de inicio: {e}"
RAFTS_ERROR_PARSER_END_DATE = "Error al convertir fecha de fin: {e}"
RAFTS_ERROR_PARSER_TEMPERATURE = "Error al convertir los datos de temperatura: {e}"
RAFTS_ERROR_PARSER_TEMPERATURE_FORECAST = "Error al convertir los datos de predicción de temperatura: {e}"
RAFTS_ERROR_PARSER_PRICE = "Error al convertir los datos de precios: {e}"
RAFTS_ERROR_PARSER_PRICE_FORECAST = "Error al convertir los datos de predicción de precios: {e}"
RAFTS_ERROR_FROM_DICT_TO_RAFT = "Error al crear balsa desde diccionario: {e}"
RAFTS_ERROR_PARSER_GROWTH = "Error al convertir los datos de crecimiento: {e}"
RAFTS_ERROR_PARSER_GROWTH_FORECAST = "Error al convertir los datos de predicción de crecimiento: {e}"
RAFTS_ERROR_PARSER_FISHES_NUMBER = "Error al convertir el número de peces: {e}"
#--- Cadenas para el modelo de precios en priceModel.py
PRICEMODEL_ERROR_PARSER_DATE = "No fue posible convertir a una fecha válida"
PRICEMODEL_ERROR_PARSER_PRICE = "No fue posible convertir el precio a un número real válido"
PRICEMODEL_ERROR_PARSER_COLUMNS_ERROR = "Las columnas:'Year', 'Week', 'Month' y 'EUR_kg' no se encuentran en los datos."
PRICEMODEL_PRICE_EMPTY_DATA_SAVE_ERROR = "No hay datos de precios para guardar"
PRICEMODEL_PRICEFORECAST_EMPTY_DATA_SAVE_ERROR = "No hay datos de predicción de precios para guardar"
PRICEMODEL_NOT_ENOUGHT_DATA = "No hay suficientes datos para el rango de fechas seleccionado"
PRICEMODEL_FIT_NO_DATA = "Los datos de precio no se han cargado. Se cancela ARIMA."
PRICEMODEL_FIT_ERROR = "Error: {e}"
#--- Cadenas para el modelo de temperatura marina en seaTemperature.py
PARSER_ERROR_COLUMN_NAME_NOT_FOUND = "No se encontró la columna {columnName} en los datos de temperatura."
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