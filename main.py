import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Visualización Hotelera", page_icon="🏨", layout="wide")

# Carga de datos
#@st.cache
def load_data():
    df = pd.read_csv('hotelbookings.csv')
    month_order = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    df['arrival_date_month'] = pd.Categorical(df['arrival_date_month'], categories=month_order, ordered=True)
    return df

df = load_data()

#########################
# Función: Visualización de ocupación por mes
def visualizacion_ocupacion(df):
    st.title("Visualización de la Ocupación del Hotel")
    texto = '''Con esta visualización podemos responder a preguntas como: ¿Cuándo se alcanzan los picos de ocupación?
               ¿Cuál es la diferencia de ocupación entre los dos tipos de hotel?, ¿Cuándo es la temporada baja?,
               ¿Qué meses necesitaremos más personal?'''
    st.text(texto)

    selected_year = st.selectbox("Selecciona el año:", sorted(df['arrival_date_year'].unique()))
    df_filtered = df[(df['is_canceled'] == 0) & (df['arrival_date_year'] == selected_year)]

    occupancy_data = df_filtered.groupby(['arrival_date_month', 'hotel']).size().reset_index(name='reservations')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=occupancy_data, x='arrival_date_month', y='reservations', hue='hotel', palette='viridis', ax=ax)
    ax.set_title(f'Ocupación Anual del Hotel por Mes y Tipo de Hotel ({selected_year})', fontsize=16)
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Número de Reservas', fontsize=12)
    ax.legend(title='Tipo de Hotel', loc='upper right')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

#########################
# Función: Distribución geográfica de reservas
def distribucion_geografica(df):
    st.header("Distribución Geográfica de Reservas")

    country_data = df['country'].value_counts().reset_index()
    country_data.columns = ['country', 'reservations']
    fig = px.choropleth(
        country_data,
        locations='country',
        locationmode='ISO-3',
        color='reservations',
        color_continuous_scale='Blues',
        title='Distribución de Reservas por País',
        labels={'reservations': 'Reservas'},
    )
    fig.update_geos(
        resolution=50,
        showcountries=True,
        showcoastlines=True,
        showland=True,
        landcolor="white",
        projection_type="mercator",
        center={"lat": 50, "lon": 10},
        lataxis_range=[30, 75],
        lonaxis_range=[-25, 40],
    )
    fig.update_layout(height=800, width=1200)
    st.plotly_chart(fig)

#########################
# Función: Heatmap interactivo
def heatmap_interactivo(df,cancel):
    st.title("Reservas Realizadas por Mes y Día de la Semana")

    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
    df['day_of_week'] = df['reservation_status_date'].dt.dayofweek
    df['month'] = df['reservation_status_date'].dt.month
    if cancel:
        df_bookings = df[df['is_canceled'] == 1]
        color='Reds'
    else:
        df_bookings = df[df['is_canceled'] == 0]
        color='Blues'
    selected_days = st.multiselect("Selecciona los Días de la Semana:", ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'], default=['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'])
    selected_months = st.multiselect("Selecciona los Meses:", ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'], default=['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'])

    day_indices = [i for i, day in enumerate(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']) if day in selected_days]
    month_indices = [i + 1 for i, month in enumerate(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']) if month in selected_months]

    filtered_data = df_bookings[(df_bookings['day_of_week'].isin(day_indices)) & (df_bookings['month'].isin(month_indices))]
    booking_counts = filtered_data.groupby(['month', 'day_of_week']).size().unstack(fill_value=0)

    xticks_labels = [day for i, day in enumerate(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']) if i in day_indices]
    yticks_labels = [month for i, month in enumerate(['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']) if i + 1 in month_indices]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(booking_counts, annot=True, fmt='d', cmap=color, linewidths=0.5, cbar_kws={'label': 'Número de Reservas'}, ax=ax)
    ax.set_xlabel('Día de la Semana', fontsize=12)
    ax.set_ylabel('Mes', fontsize=12)
    ax.set_xticks(range(len(xticks_labels)))
    ax.set_xticklabels(xticks_labels)
    ax.set_yticks(range(len(yticks_labels)))
    ax.set_yticklabels(yticks_labels, rotation=0)
    st.pyplot(fig)


def cancelaciones(df):
    # Mapeo para los tipos de reservas y meses
    map_reserva = {
        'Direct': 'Directa', 'Corporate': 'Empresa', 'Online TA': 'Agencia Online', 
        'Offline TA/TO': 'Agencia no Online', 'Complementary': 'Complementario', 
        'Groups': 'Grupo', 'Undefined': 'Indefinido', 'Aviation': 'Avión'}
    map_meses = {
        'July': 'Jul', 'August': 'Ago', 'September': 'Sep', 'October': 'Oct', 
        'November': 'Nov', 'December': 'Dec', 'January': 'Ene', 'February': 'Feb', 
        'March': 'Mar', 'April': 'Abr', 'June': 'Jun'}
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
    df['day_of_week'] = df['reservation_status_date'].dt.dayofweek
    df['month'] = df['reservation_status_date'].dt.month
    # Título y descripción
    titulo = "Análisis de las cancelaciones por tipo de reserva"
    descripcion = """Se van a buscar las relaciones entre los diversos tipos de reservas y cancelaciones de las mismas"""
    st.title(titulo)
    st.write(descripcion)
    # Selección de tipo de cancelación que quiere consultar
    reserva_hecha = list(df['market_segment'].unique())
    reserva_hecha.append('Todos')
    market = st.selectbox('Tipo de reserva para consultar', reserva_hecha)
    # Filtrar los datos según el tipo de reserva seleccionado
    if market == 'Todos':
        client_selec = df
    else:
        client_selec = df[df['market_segment'] == market]
    # Asegúrate de que hay datos después de filtrar
    if not client_selec.empty:
        # Índices de años y meses
        months = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]               ]
        meses = [ 'Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
        # Filtrar cancelaciones
        #Datos por mes
        years_=['2015','2016','2017']
        info_y= st.selectbox ('Selecciona el año:', years_)         
        selecbymonth = [client_selec[client_selec['arrival_date_month']==i] [client_selec['arrival_date_year']==int(info_y)]['hotel'].count() for i in months]
        selecbymonth_cancel = [client_selec[client_selec['arrival_date_month']==i][client_selec['arrival_date_year']==int(info_y)][client_selec['is_canceled']==1]['hotel'].count() for i in months]
        tasas_mont=[100*selecbymonth_cancel[i]/selecbymonth[i] for i in range(len(selecbymonth))]
        # # Gráfico de barras de cancelaciones vs reservas
        fig2, ax2 = plt.subplots()
        ax2.bar(meses, selecbymonth_cancel, color="red") 
        ax2.bar(meses, selecbymonth, width=0.5, align="edge", color="skyblue")
        ax2.set_title("Reservas por meses en el año {0}".format(info_y))
        ax2.set_xlabel("Meses")
        ax2.set_ylabel("Nº Clientes hicieron la reserva con {0} ".format(market))
        ax2_2 = ax2.twinx()  # Crear un segundo eje Y
        ax2_2.plot(meses, tasas_mont, color="orange", marker="o", label="Tasa de Cancelación (%)")
        ax2_2.set_ylabel("Tasa de Cancelación (%)", color="orange")
        ax2_2.tick_params(axis="y", labelcolor="orange")
        st.pyplot(fig2)     
                
    

#########################
# Función: Predicción de cancelaciones
def prediccion(df,cancel):
    st.header("Modelo de Predicción de Reservas y Cancelaciones")

    data = df[['hotel', 'arrival_date_month', 'arrival_date_year', 'is_canceled']]
    if cancel:
        data = data[data['is_canceled'] == 1]
        texto='cancelaciones'
    else:
        data = data[data['is_canceled'] == 0]
        texto='reservas'
    data = data.groupby(['hotel', 'arrival_date_month', 'arrival_date_year']).count().reset_index()
    data = data.rename(columns={'is_canceled': 'cancelaciones'})

    X = pd.get_dummies(data[['hotel', 'arrival_date_month', 'arrival_date_year']])
    y = data['cancelaciones']
    original_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    hotel = st.selectbox("Selecciona el tipo de hotel:", df['hotel'].unique())
    month = st.selectbox("Selecciona el mes de llegada:", [month for month in X.columns if 'arrival_date_month' in month])
    year = st.number_input("Introduce el año de llegada:", min_value=2000, max_value=2030, step=1)

    def make_prediction(hotel, month, year):
        data = {'hotel': [hotel], 'arrival_date_month': [month], 'arrival_date_year': [year]}
        df_pred = pd.DataFrame(data)
        df_pred = pd.get_dummies(df_pred)
        for col in original_columns:
            if col not in df_pred:
                df_pred[col] = 0
        df_pred = df_pred[original_columns]
        return model.predict(df_pred)[0]

    if st.button(f"Predecir {texto}"):
        prediction = make_prediction(hotel, month, year)
        st.success(f"El número de {texto} estimadas para {hotel} en {month} del {year} es: {prediction:.2f}")


def toggle_checkbox_r():
    if st.session_state.checkbox_r:  # Si el primer checkbox está activado
        st.session_state.checkbox_c = False  # Desactiva el segundo

def toggle_checkbox_c():
    if st.session_state.checkbox_c:  # Si el segundo checkbox está activado
        st.session_state.checkbox_r = False  # Desactiva el primero


#########################
# Ejecución
st.header("ANÁLISIS HOTELERO")
Texto= '¿Qué información desea consultar primero?'
st.write(Texto)
col1, col2, col3,col4 = st.columns(4)
if "checkbox_r" not in st.session_state:
        st.session_state.checkbox_r = False
if "checkbox_c" not in st.session_state:
        st.session_state.checkbox_c = False
# Añadir checkboxes a cada columna
with col1:
    checkbox_ocupacion = st.checkbox("Ocupación anual total", value = False)
with col2:
    checkbox_mapa = st.checkbox("Mapa de reservas", value = False)
with col3:
    checkbox_reser = st.checkbox("Información sobre las reservas", value = st.session_state.checkbox_r,
                                 key="checkbox_r", on_change=toggle_checkbox_r)
with col4:
    checkbox_cancel = st.checkbox("Información sobre las cancelaciones", value = st.session_state.checkbox_c,
                                  key="checkbox_c", on_change=toggle_checkbox_c)
    if checkbox_cancel:
        checkbox_map_bar = st.selectbox("Muestra", ['Heatmap','Barplot'])



if checkbox_ocupacion:
    visualizacion_ocupacion(df)
if checkbox_mapa:
    distribucion_geografica(df)
if checkbox_reser:
    heatmap_interactivo(df,False)
    prediccion(df,False)
if checkbox_cancel:
    if checkbox_map_bar=='Heatmap':
        heatmap_interactivo(df,True)
    else:
        cancelaciones(df)
    prediccion(df,True)
