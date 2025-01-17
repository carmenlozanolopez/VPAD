# Visualización Hotelera
Este proyecto es una aplicación interactiva de visualización de datos y análisis predictivo para reservas hoteleras, construida con Streamlit y diversas bibliotecas de visualización y aprendizaje automático. El objetivo es proporcionar información detallada sobre las reservas, ocupación y cancelaciones de hoteles, además de predecir el comportamiento futuro.

## Requisitos
- Python 3.x
- Streamlit
- Pandas
- Plotly
- Seaborn
- Matplotlib
- scikit-learn

## Instalación

1. Clona este repositorio:
```
git clone https://github.com/tu_usuario/visualizacion-hotelera.git
```
2. Navega al directorio del proyecto:
```
cd visualizacion-hotelera
```
3. Instala las dependencias:
```
pip install -r requirements.txt```

## Ejecución de la Aplicación

Para ejecutar la aplicación, utiliza el siguiente comando:

streamlit run app.py```

## Características Principales

### 1. Visualización de Ocupación por Mes

Esta sección muestra la ocupación del hotel mes a mes, permitiendo identificar patrones de temporada alta y baja.
- Selección del año de interés.
- Visualización de la ocupación para dos tipos de hoteles.

### 2. Distribución Geográfica de Reservas

Un mapa interactivo que muestra el origen geográfico de las reservas.
- Visualización en un mapa coroplético.
- Datos agregados por país.

### 3. Heatmap Interactivo
Permite visualizar las reservas o cancelaciones por mes y día de la semana.
- Filtrado por días de la semana y meses seleccionados.
- Heatmap interactivo con opciones de personalización.

### 4. Análisis de Cancelaciones
Analiza las cancelaciones según el tipo de reserva y muestra la tasa de cancelación por mes.
- Gráfico de barras que compara reservas y cancelaciones.
- Análisis detallado de cancelaciones por tipo de mercado.

### 5. Predicción de Cancelaciones
Modelo predictivo de cancelaciones utilizando regresión lineal.
- Entrenamiento del modelo con datos históricos.
- Predicción del número de cancelaciones o reservas futuras según tipo de hotel, mes y año.

## Uso de la Aplicación
- Selección de Datos: Elige qué tipo de información deseas visualizar utilizando los checkboxes en la interfaz principal.
- Interactividad: Usa los selectores y multiselect para filtrar los datos según tus necesidades.
- Predicción: Introduce los datos relevantes para hacer una predicción de futuras cancelaciones o reservas.
