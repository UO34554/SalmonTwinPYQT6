# SalmonTwin

## Descripción
Gestor unificado de balsas marinas para el sistema Salmon Twin. Esta aplicación permite la configuración, monitorización y análisis de datos de balsas marinas para la acuicultura de salmón.

## Autor
Pedro López Treitiño

## Características principales
- Configuración de balsas marinas
- Dashboard para monitorización
- Predicción de precios utilizando modelos ARIMA
- Interfaz gráfica desarrollada con PySide6 (Qt)

## Requisitos
Para instalar las dependencias necesarias:
```bash
pip install -r requirements.txt
```

## Estructura del proyecto
```
salmon-twin/
├── assets/
│   ├── icons/
│   └── images/
├── config/
│   ├── prices.json
│   └── rafts.json
└── src/
    ├── controller/
    ├── model/
    ├── ui/
    ├── utility/
    ├── config.py
    ├── main.py
    └── resources.py
```

## Instalación y ejecución
1. Clona el repositorio:
   ```bash
   git clone [URL_DEL_REPOSITORIO]
   ```

2. Instala las dependencias:
   ```bash
   cd salmon-twin
   pip install -r requirements.txt
   ```

3. Compila los recursos (si es necesario):
   ```bash
   cd salmon-twin
   pyside6-rcc salmonResources.qrc -o src/resources.py
   ```

4. Ejecuta la aplicación:
   ```bash
   python src/main.py
   ```

## Configuración
- La aplicación utiliza archivos de configuración ubicados en el directorio `config/`:
  - `prices.json`: Datos de precios históricos
  - `rafts.json`: Configuración de balsas marinas

## Modelos de predicción
La aplicación incluye capacidades de predicción de precios utilizando modelos estadísticos ARIMA con las siguientes características:
- Análisis de series temporales
- Predicción de precios futuros
- Visualización de datos históricos y predicciones

## Licencia
Este proyecto está protegido por derechos de autor.
