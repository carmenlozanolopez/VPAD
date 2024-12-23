import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# page configuration
st.set_page_config(page_title="Visualización Hotelera", page_icon="🏨", layout="wide")

# data from file
df = pd.read_csv('hotelbookings.csv', delimiter=';')

#%% Gráfico de barras: Ocupación del hotel anual

# título y descripción 
titulo = "Visualización de la Ocupación Anual del Hotel"
descripcion = """Con esta visualización podemos responder a preguntas como, ¿cuáles son las tendencias de ocupación hotelera a lo largo del año?
        Analiza los picos de ocupación y tendencias a lo largo del tiempo."""
st.title(titulo)
st.write(descripcion)

# convertimos las variables a una sola columna de fecha
df['arrival_date'] = pd.to_datetime(df['arrival_date_year'].astype(str) + '-' +
                                    df['arrival_date_month'].str[:3] + '-01')
# separamos mes y año
df['arrival_month'] = df['arrival_date'].dt.month
df['arrival_year'] = df['arrival_date'].dt.year

# agrupamos el número de reservas por año y mes
occupancy_per_month = df.groupby(['arrival_year', 'arrival_month']).size().reset_index(name='booking_count')

# selector interactivo de streamlit de los diferentes años
years = sorted(occupancy_per_month['arrival_year'].unique())
selected_years = st.multiselect("Selecciona los años a visualizar:", options=years, default=years)
# filtramos datos según los años seleccionados
filtered_data = occupancy_per_month[occupancy_per_month['arrival_year'].isin(selected_years)]

# gráfico con Seaborn (personalizado y ajustando diseño)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='arrival_month', y='booking_count', hue='arrival_year', data=filtered_data, ax=ax)

ax.set_title("Ocupación Anual del Hotel por Mes", fontsize=12)
ax.set_xlabel("Mes", fontsize=12)
ax.set_ylabel("Número de Reservas", fontsize=10)
ax.set_xticks(range(12))
ax.set_xticklabels(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])
ax.legend(title="Año", loc='upper left')

plt.tight_layout()

st.pyplot(fig)


#%% Pie Chart Distribución de Cancelaciones por Tipo de Habitación

titulo = "Distribución de Cancelaciones por Tipo de Habitación y Año"
descripcion = """Este gráfico muestra la distribución porcentual de cancelaciones según el tipo de habitación reservada para cada año.
Puedes seleccionar el año para un análisis más detallado."""
st.title(titulo)
st.write(descripcion)

# filtramos datos para las cancelaciones y obtenemos los años disponibles
df_cancelations = df[df['is_canceled'] == 1]
years = sorted(df_cancelations['arrival_date_year'].unique())

# selector interactivo de streamlit para seleccionar el año
selected_year = st.selectbox("Selecciona el año:", options=years)

# filtramos datos según el año seleccionado
df_cancelations_year = df_cancelations[df_cancelations['arrival_date_year'] == selected_year]

# contamos cancelaciones por tipo de habitación
cancelation_counts = df_cancelations_year['reserved_room_type'].value_counts()
total = sum(cancelation_counts)     # Suma total de cancelaciones
pctdistance_values = [1 - (p / total)*0.6 for p in cancelation_counts]  # Ajuste en función del porcentaje

# gráfico con Matplotlib (personalizado y ajustando diseño/tamaño)
fig, ax = plt.subplots(figsize=(6, 4)) 

wedges, texts, autotexts = ax.pie(
    cancelation_counts, 
    autopct=lambda p: f'{p:.1f}%' if p > 4 else '',  # condición if para el ajuste de porcentajes pequeños
    startangle=90,
    textprops={'fontsize': 6},
    radius=0.9,
    pctdistance=0.7,
    colors=plt.cm.Set3(range(len(cancelation_counts))),  
    wedgeprops={'edgecolor': 'black'}  
)

ax.set_title(f"Distribución de Cancelaciones por Tipo de Habitación en {selected_year}", fontsize=8)
ax.set_ylabel('')  

# ajustamos la posición del texto en función de los valores de pctdistance
for t, d in zip(autotexts, pctdistance_values):
    xi, yi = t.get_position()           
    ri = np.sqrt(xi**2 +yi**2)          
    phi = np.arctan2(yi, xi)            
    x = d*ri*np.cos(phi)                
    y = d*ri*np.sin(phi)
    t.set_position((x, y))

# leyenda aporta una mejor legibilidad
ax.legend(cancelation_counts.index, title="Tipo de Habitación", title_fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6)

st.pyplot(fig)


#%% Tipo de gráfico: Precio Promedio por Noche según Tipo de Habitación

titulo = "Visualización del Precio Promedio por Noche según Tipo de Habitación"
descripcion = """Con esta visualización puedes ver cómo varía el precio promedio por noche (ADR) según el tipo de habitación reservado en el hotel.
        Analiza las diferencias en los precios según las preferencias de los huéspedes."""
st.title(titulo)
st.write(descripcion)

# filtramos columnas relevantes y calculamos el precio promedio por tipo de habitación
df_room_price = df[['hotel', 'adr', 'reserved_room_type']]
avg_price_by_room = df_room_price.groupby('reserved_room_type')['adr'].mean().reset_index()

# selector interactivo para elegir el tipo de habitación
room_types = sorted(avg_price_by_room['reserved_room_type'].unique())
selected_rooms = st.multiselect("Selecciona los tipos de habitación a visualizar:", options=room_types, default=room_types)

# filtramos datos según los tipos de habitación 
filtered_data = avg_price_by_room[avg_price_by_room['reserved_room_type'].isin(selected_rooms)]

# gráfico con Seaborn
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='reserved_room_type', y='adr', data=filtered_data, ax=ax, palette='viridis')

ax.set_title("Precio Promedio por Noche según el Tipo de Habitación", fontsize=12)
ax.set_xlabel("Tipo de Habitación", fontsize=12)
ax.set_ylabel("Precio Promedio por Noche (ADR)", fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()

st.pyplot(fig)


#%% Tipo de gráfico: Tasa de Cancelación por Tipo de Cliente y Antigüedad de la Reserva

titulo = "Análisis de la Tasa de Cancelación por Tipo de Cliente"
descripcion = """Con estas visualizaciones podemos analizar las tasas de cancelación por tipo de cliente y cómo la antigüedad de la reserva (lead time) 
influye en dicha tasa. Utilizamos dos tipos de gráficos: uno de barras para la tasa de cancelación por tipo de cliente y uno de cajas para la variación de la tasa de cancelación según la antigüedad."""
st.title(titulo)
st.write(descripcion)

# selector interactivo para elegir los tipos de cliente que se quieren visualizar
customer_types = df['customer_type'].map({
    'Transient': 'Nuevo',
    'Contract': 'Repetitivo',
    'Group': 'Repetitivo',
    'Transient-Party': 'Nuevo'
}).unique()

selected_types = st.multiselect("Selecciona los tipos de cliente a visualizar:", options=customer_types, default=customer_types)

# filtramos datos según los tipos de cliente seleccionados
df_new_vs_repeated = df[['customer_type', 'is_canceled']]
df_new_vs_repeated['customer_type'] = df_new_vs_repeated['customer_type'].map({
    'Transient': 'Nuevo',
    'Contract': 'Repetitivo',
    'Group': 'Repetitivo',
    'Transient-Party': 'Nuevo'
})
df_new_vs_repeated_filtered = df_new_vs_repeated[df_new_vs_repeated['customer_type'].isin(selected_types)]

# tasa de cancelación por tipo de cliente
cancel_rate_by_type = df_new_vs_repeated_filtered.groupby('customer_type')['is_canceled'].mean().reset_index()

# gráfico con Seaborn
fig, ax = plt.subplots(figsize=(8, 4))  # Tamaño ajustado
sns.barplot(data=cancel_rate_by_type, x='customer_type', y='is_canceled', palette='Purples', ax=ax)

ax.set_title('Tasa de cancelación por tipo de cliente', fontsize=14)
ax.set_xlabel('Tipo de Cliente', fontsize=10)
ax.set_ylabel('Tasa de Cancelación', fontsize=10)

plt.tight_layout()

st.pyplot(fig)


#%% Tipo de gráfico: Variación de la tasa de cancelación según la antigüedad del cliente (lead time)

titulo = "Análisis de la Tasa de Cancelación por Antigüedad del Cliente"
descripcion = """Con estas visualizaciones podemos analizar las tasas de cancelación y cómo la antigüedad de la reserva (lead time) 
influye en dicha tasa para diferentes tipos de cliente. Utilizamos un gráfico de cajas (boxplot) para mostrar la distribución de las cancelaciones
en función de la antigüedad (lead time) de la reserva."""
st.title(titulo)
st.write(descripcion)

# selector interactivo para el rango de antigüedad del cliente (lead time)
min_lead_time = int(df['lead_time'].min())
max_lead_time = int(df['lead_time'].max())

# selección de rango de antigüedad del cliente (lead time)
lead_time_range = st.slider("Selecciona el rango de antigüedad (Lead Time) de la reserva:", 
                            min_value=min_lead_time, max_value=max_lead_time, 
                            value=(min_lead_time, max_lead_time), step=1)

# filtramos datos según el rango de antigüedad (lead time) seleccionado
df_filtered_lead_time = df[(df['lead_time'] >= lead_time_range[0]) & (df['lead_time'] <= lead_time_range[1])]

# gráfico con Seaborn
fig, ax = plt.subplots(figsize=(8, 4))  
# boxplot por tener algo más diferente pero quizá haya otro mejor
sns.boxplot(data=df_filtered_lead_time, x='customer_type', y='lead_time', hue='is_canceled', palette='coolwarm', ax=ax) 

ax.set_title(f'Variación de la tasa de cancelación según la antigüedad del cliente (Lead Time) \n'
             f'Rango de Antigüedad: {lead_time_range[0]} a {lead_time_range[1]} días', fontsize=14)
ax.set_xlabel('Tipo de Cliente', fontsize=10)
ax.set_ylabel('Lead Time (Días de Antigüedad)', fontsize=10)

plt.tight_layout()

st.pyplot(fig)


#%% Tipo de gráfico: Heatmap Interactivo de Reservas Realizadas

titulo = "Reservas Realizadas por Mes y Día de la Semana"
descripcion = """Con esta visualización interactiva, puedes explorar las reservas realizadas en el hotel según el día de la semana y el mes."""
st.title(titulo)
st.write(descripcion)

# formato correcto para la fecha
try:
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], format='%d/%m/%Y')
except ValueError:
    # inferencia automática
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], dayfirst=True, errors='coerce')

# verificamos si hay valores 'isna' después de la conversión
if df['reservation_status_date'].isna().sum() > 0:
    st.error("Algunas fechas no se pudieron convertir correctamente. Verifica el formato de las fechas en tu archivo.")
    st.stop()

# columnas para día de la semana y mes
df['day_of_week'] = df['reservation_status_date'].dt.dayofweek  # 0 = lunes, 6 = domingo
df['month'] = df['reservation_status_date'].dt.month

# filtramos reservas no canceladas
df_bookings = df[df['is_canceled'] == 0]

# selector interactivo
st.sidebar.title("Filtros")
selected_days = st.sidebar.multiselect(
    "Selecciona los Días de la Semana:", 
    ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'], 
    default=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
)
selected_months = st.sidebar.multiselect(
    "Selecciona los Meses:", 
    ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], 
    default=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
)

# pasamos días y mes seleccionados a índices
day_indices = [i for i, day in enumerate(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']) if day in selected_days]
month_indices = [i + 1 for i, month in enumerate(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']) if month in selected_months]

# filtramos datos según la selección
filtered_data = df_bookings[
    (df_bookings['day_of_week'].isin(day_indices)) & 
    (df_bookings['month'].isin(month_indices))
]

# contamos las reservas
booking_counts = filtered_data.groupby(['month', 'day_of_week']).size().unstack(fill_value=0)

# ajustamos los ejes según datos seleccionados
xticks_labels = [day for i, day in enumerate(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']) if i in day_indices]
yticks_labels = [month for i, month in enumerate(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']) if i + 1 in month_indices]

# heatmap
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(booking_counts, annot=True, fmt='d', cmap='Blues', linewidths=0.5, cbar_kws={'label': 'Número de Reservas'}, ax=ax)

ax.set_xlabel('Día de la Semana', fontsize=12)
ax.set_ylabel('Mes', fontsize=12)
ax.set_xticks(range(len(xticks_labels)))
ax.set_xticklabels(xticks_labels)
ax.set_yticks(range(len(yticks_labels)))
ax.set_yticklabels(yticks_labels, rotation=0)

st.pyplot(fig)
