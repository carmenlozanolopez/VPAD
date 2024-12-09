import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Configuración de la página
st.set_page_config(page_title="TRABAJO VPAD", page_icon="❄️", layout="wide")

st.markdown('---')
st.header("ANÁLISIS HOTELERO")

# Carga de datos
df_hotel = pd.read_csv('hotel_bookings.csv')

# Convertir los meses a un orden correcto
month_order = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
df_hotel['arrival_date_month'] = pd.Categorical(
    df_hotel['arrival_date_month'], categories=month_order, ordered=True
)

# Visualización: Ocupación por mes
st.title("Visualización de la Ocupación del Hotel")
texto = '''Con esta visualización podemos responder a preguntas como: ¿Cuándo se alcanzan los picos de ocupación?
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

# Modelo de regresión lineal para predecir ocupación
st.header("Modelo de Predicción de cancelaciones")

texto_prediccion='''Con los datos que disponemos podemos tratar de predecir cuántas cancelaciones vamos a tener en un mes y año determinado.
 Esto nos puede ser útil para planificar la cantidad de personal necesario en un mes concreto o para anticiparnos y tratar de planterar estrategias 
 para reducir el número de cancelaciones. '''
 
st.text(texto_prediccion)

# Seleccionar variables de interés
data=df_hotel[['hotel','arrival_date_month','arrival_date_year','is_canceled']]
data=data[data['is_canceled']==1]
data=data.groupby(['hotel','arrival_date_month', 'arrival_date_year']).count().reset_index()
data=data.rename(columns={'is_canceled':'cancelaciones'})

X=data[['hotel','arrival_date_month','arrival_date_year']]
y=data['cancelaciones']

# Crear variables dummy para el modelo
X = pd.get_dummies(X)
original_columns = X.columns

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Predicción interactiva
st.subheader("Hacer una Predicción")
hotel = st.selectbox("Selecciona el tipo de hotel:", df_hotel['hotel'].unique())
month = st.selectbox("Selecciona el mes de llegada:", month_order)
year = st.number_input("Introduce el año de llegada:", min_value=2000, max_value=2030, step=1)

# Función para predecir
def make_prediction(hotel, month, year):
    data = {'hotel': [hotel], 'arrival_date_month': [month], 'arrival_date_year': [year]}
    df = pd.DataFrame(data)
    df = pd.get_dummies(df)
    
    # Asegurar que las columnas coincidan con las del modelo
    for col in original_columns:
        if col not in df:
            df[col] = 0
    df = df[original_columns]

    prediction = model.predict(df)[0]
    return prediction

# Botón para generar predicción
if st.button("Predecir cancelaciones"):
    prediction = make_prediction(hotel, month, year)
    st.success(f"El número de cancelaciones estimadas para {hotel} en {month} del {year} es: {prediction:.2f}")
    
st.header("Mapa de Calor: Cancelaciones vs Tipo de Cliente")

texto_correlaciones='''Con este gráfico podemos tratar de responder a preguntas como: ¿Qué tipo de cliente tiene más o menos probabilidad de cancelar una reserva?
o ¿qué tipo de cliente debería ser el foco de estrategias para reducir cancelaciones? '''

st.text(texto_correlaciones)

# Convertir 'customer_type' en variables numéricas (codificación dummy)
df_hotel_encoded = pd.get_dummies(df_hotel, columns=['customer_type'], drop_first=True)

# Seleccionar columnas relevantes
correlation_matrix = df_hotel_encoded[['is_canceled'] + [col for col in df_hotel_encoded.columns if 'customer_type_' in col]].corr()

column_labels = {
    'is_canceled': 'Cancelaciones',
    'customer_type_Contract': 'Contract',
    'customer_type_Group': 'Group',
    'customer_type_Transient': 'Transient',
    'customer_type_Transient-Party': 'Transient Party'
}
correlation_matrix.rename(columns=column_labels, index=column_labels, inplace=True)


# Crear Heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

# Personalizar
plt.tight_layout()

# Mostrar en Streamlit
st.pyplot(fig)

