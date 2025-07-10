# SalmonTwin

## Descripción
Prototipo de gemelo digital de jaulas marinas (balsas). Esta aplicación permite la configuración, monitorización y simulación de datos de balsas marinas para la acuicultura de salmón, con capacidades avanzadas de predicción y optimización de modelos.

## Autor
Pedro López Treitiño

## Características principales
- **Configuración de balsas marinas**: Gestión completa de múltiples balsas con datos específicos por región
- **Dashboard interactivo**: Monitorización en tiempo real con gráficos 2D y visualización 3D
- **Modelos de predicción avanzados**:
  - Predicción de precios con modelos LightGBM y optimización automática de parámetros
  - Predicción de temperatura marina usando Prophet
  - Modelo de crecimiento de biomasa basado en Thyholdt (2014)
- **Interfaz gráfica moderna**: Desarrollada con PySide6 (Qt6) y PyQtGraph
- **Análisis de series temporales**: Con capacidades de interpolación y extrapolación
- **Sistema de pruebas unitarias**: Testing automatizado con pytest

## Requisitos del sistema
- Python 3.11.9
- Sistema operativo: Windows/Linux/macOS
- Memoria RAM: 4GB mínimo recomendado
- Espacio en disco: 500MB para instalación completa

## Instalación

### 1. Clonar el repositorio
```bash
git clone [URL_DEL_REPOSITORIO]
cd SalmonTwinPYQT6
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
cd salmon-twin
pip install -r requirements.txt
```

### 4. Compilar recursos gráficos
```bash
pyside6-rcc salmonResources.qrc -o src/resources.py
```

### 5. Ejecutar la aplicación
```bash
python src/main.py
```

## Estructura del proyecto
```
salmon-twin/
├── assets/
│   ├── icons/          # Iconos de la interfaz
│   └── images/         # Imágenes y recursos gráficos
├── config/
│   ├── estimators.json # Configuración de estimadores optimizados
│   └── rafts.json     # Configuración de balsas marinas
└── src/
    ├── controller/     # Controladores MVC
    ├── model/         # Modelos de datos y predicción
    ├── tests/         # Pruebas unitarias
    ├── ui/            # Archivos de interfaz Qt
    ├── utility/       # Utilidades auxiliares
    ├── config.py      # Configuración global
    ├── main.py        # Punto de entrada
    └── resources.py   # Recursos compilados
```

## Dependencias principales
- **PySide6**: Interfaz gráfica moderna
- **PyQtGraph**: Gráficos interactivos de alto rendimiento
- **Pandas**: Manipulación y análisis de datos
- **NumPy**: Computación numérica
- **StatsForecast**: Modelos estadísticos avanzados
- **Prophet**: Predicción de series temporales
- **LightGBM**: Algoritmos de machine learning
- **scikit-learn**: Métricas y preprocesamiento

## Modelos de predicción

### 1. Modelo de Precios
- **Algoritmo**: LightGBM con características de ventana deslizante
- **Optimización**: Búsqueda aleatoria de hiperparámetros
- **Métricas**: MAE, RMSE, MAPE, precisión direccional
- **Funcionalidades**: Guardado automático de mejores configuraciones

### 2. Modelo de Temperatura Marina
- **Algoritmo**: Prophet con estacionalidad anual
- **Características**: Detección automática de puntos de cambio
- **Interpolación**: Lineal entre puntos de datos

### 3. Modelo de Crecimiento (Thyholdt 2014)
- **Base científica**: Función logística modificada
- **Parámetros**: Temperatura, tiempo, peso asintótico
- **Mortalidad**: Modelo exponencial integrado

## Configuración

### Archivos de configuración
La aplicación utiliza archivos JSON en el directorio `config/`:
- **rafts.json**: Configuración de balsas, temperaturas, precios y predicciones
- **estimators.json**: Mejores configuraciones de modelos encontradas

### Regiones marítimas soportadas
- Finnmark
- Troms  
- Nordland
- Nord-Trøndelag
- Sør-Trøndelag
- Møre og Romsdal
- Sogn og Fjordane
- Hordaland
- Rogaland og Agder

## Testing
Ejecutar las pruebas unitarias:
```bash
# Desde el directorio salmon-twin
pytest src/tests/ -v
```

## Desarrollo

### Actualizar dependencias
```bash
pip list --outdated --format=json | ConvertFrom-Json | ForEach-Object { pip install --upgrade $_.name }
```

### Compilar recursos tras cambios en .qrc
```bash
pyside6-rcc salmonResources.qrc -o src/resources.py
```

## Características técnicas avanzadas

### Dashboard interactivo
- Zoom dinámico con recálculo automático de etiquetas de tiempo
- Líneas verticales sincronizadas entre gráficos
- Visualización 3D de biomasa con animación de peces
- Panel de información en tiempo real

### Optimización de modelos
- Búsqueda automática de hiperparámetros
- Validación cruzada temporal
- Ranking de configuraciones por score compuesto
- Interfaz de progreso en tiempo real

### Gestión de datos
- Interpolación automática entre fechas
- Validación de integridad de datos
- Manejo robusto de errores
- Soporte para datos históricos extendidos

## Solución de problemas

### Error: "No se encontró el archivo de interfaz"
Verifique que los archivos .ui estén en el directorio correcto y recompile los recursos.

### Error: "Datos insuficientes para entrenamiento"
Amplíe el rango de fechas o cargue más datos históricos.

### Problemas de rendimiento
Considere reducir el número de iteraciones en la optimización de modelos.

## Licencia
Este proyecto está protegido por derechos de autor.

## Contacto
Pedro López Treitiño - Desarrollo y mantenimiento del sistema SalmonTwin.
