import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import pandas as pd

st.set_page_config(page_title="TRABAJO VPAD",page_icon="❄️",layout="wide")

st.markdown('---')
st.header("ANÁLISIS HOTELERO")

#Carga de datos
df_hotel = pd.read_csv('hotel_bookings.csv')
import seaborn as sns
import matplotlib.pyplot as plt


# Convertir los meses a un orden correcto
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
df_hotel['arrival_date_month'] = pd.Categorical(
    df_hotel['arrival_date_month'], categories=month_order, ordered=True
)

# Título
st.title("Visualización de la Ocupación del Hotel")
texto='''Con esta visualización podemos responder a preguntas como: ¿Cuándo se alcanzan los picos de ocupación?
        ¿Cuál es la diferencia de ocupación entre los dos tipos de hotel?, ¿Cuándo es la temporada baja?,
        ¿Qué meses necesitaremos más personal?'''
st.text(texto)

# Filtro interactivo para seleccionar el año
selected_year = st.selectbox("Selecciona el año:", sorted(df_hotel['arrival_date_year'].unique()))

# Filtrar datos según el año seleccionado
df_filtered = df_hotel[(df_hotel['is_canceled'] == 0) & (df_hotel['arrival_date_year'] == selected_year)]

# Agrupar los datos por mes y tipo de hotel
occupancy_data = df_filtered.groupby(['arrival_date_month', 'hotel']).size().reset_index(name='reservations')

# Crear el gráfico
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(
    data=occupancy_data,
    x='arrival_date_month',
    y='reservations',
    hue='hotel',
    palette='viridis',
    ax=ax,
    markers=True
)

# Personalizar el gráfico
ax.set_title(f'Ocupación Anual del Hotel por Mes y Tipo de Hotel ({selected_year})', fontsize=16)
ax.set_xlabel('Mes', fontsize=12)
ax.set_ylabel('Número de Reservas', fontsize=12)
ax.legend(title='Tipo de Hotel', loc='upper right')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)
