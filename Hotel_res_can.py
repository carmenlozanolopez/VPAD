import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
#%%

# Carga de datos y modificaciones
df_hotel = pd.read_csv('hotel_bookings.csv')
map_reserva= {'Direct':'Directa',
    'Corporate':'Empresa' ,
    'Online TA': 'Agencia Online',
    'Offline TA/TO': 'Agencia no Online',
    'Complementary': 'Complementario',
    'Groups': 'Grupo',
    'Undefined': 'Indefinido',
    'Aviation': 'Avión'}

map_meses= {'July':'Jul', 'August':'Ago', 'September':'Sep', 'October':'Oct', 'November':'Nov', 'December':'Dec',
       'January':'Ene', 'February':'Feb', 'March':'Mar', 'April':'Abr', 'June':'Jun'}

# Configuración de la página
st.set_page_config(page_title="TRABAJO VPAD", page_icon="❄️", layout="wide")

st.markdown('---')
st.header("ANÁLISIS HOTELERO")

titulo = "Análisis de las cancelaciones por tipo de reserva"
descripcion = """Se van a buscar las relaciones entre los diversos tipos de reservas y cancelaciones de las mismas"""
st.title(titulo)
st.write(descripcion)

#Bar plot con tipo de reserva por año y el número de ellas que se cancelan


#selección de tipo de cancelación que quiere consultar
reserva_hecha = list(df_hotel['market_segment'].unique())
reserva_hecha.append('Todos')
market = st.selectbox ('Tipo de reserva para consultar', reserva_hecha) 
if market=='Todos':
    client_selec= df_hotel['market_segment'].isnull()==False
else:
    client_selec = df_hotel['market_segment']==market

#indices
years = df_hotel['arrival_date_year'].unique()
months = df_hotel['arrival_date_month'].unique()
meses = ['Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic','Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun']


cancelado = df_hotel['is_canceled']==1
st.write('En rojo se muestran las cancelaciones para ese tipo de reserva')
selecbyyear = [df_hotel['hotel'][client_selec][df_hotel['arrival_date_year']==2015].count(),
               df_hotel['hotel'][client_selec][df_hotel['arrival_date_year']==2016].count(),
               df_hotel['hotel'][client_selec][df_hotel['arrival_date_year']==2017].count()] 

selecbyyearcanc = [df_hotel['hotel'][client_selec][cancelado][df_hotel['arrival_date_year']==2015].count(),
               df_hotel['hotel'][client_selec][cancelado][df_hotel['arrival_date_year']==2016].count(),
               df_hotel['hotel'][client_selec][cancelado][df_hotel['arrival_date_year']==2017].count()] 

fig, ax = plt.subplots()
ax.bar(years, selecbyyearcanc, color="red") 
ax.bar(years, selecbyyear, width=0.5, align="edge", color="skyblue")

ax.set_title("Cancelaciones frente a número de reservas")
ax.set_xlabel("Años")
ax.set_ylabel("Nº Clientes hicieron la reserva con "+market)

st.pyplot(fig)

#en porcentaje
tasas_ano = [int(100*selecbyyearcanc[0]/selecbyyear[0]),int(100*selecbyyearcanc[1]/selecbyyear[1]),int(100*selecbyyearcanc[2]/selecbyyear[2])]
porcentajes= 'TASA DE CANCELACIÓN DE CADA AÑO '

fig, ax = plt.subplots()
ax.bar(years, tasas_ano, width=0.6,color="red")
ax.set_title(porcentajes)
ax.set_xlabel("Años")
ax.set_ylabel("Tasa de clientes cancelaron la reserva con "+market)
st.pyplot(fig)


#Para dar más detalle de cada año:
st.write('Si desea más detalles de cada año:')
years_=['No','2015','2016','2017']
info_y= st.selectbox ('Año para consultar', years_) 

if info_y !="No":
    selecbymonth = [df_hotel['hotel'][client_selec][df_hotel['arrival_date_month']==i][df_hotel['arrival_date_year']==int(info_y)].count() for i in months]
    selecbymonth_cancel = [df_hotel['hotel'][client_selec][cancelado][df_hotel['arrival_date_month']==i][df_hotel['arrival_date_year']==int(info_y)].count() for i in months]
    fig2, ax2 = plt.subplots()
    ax2.bar(meses, selecbymonth_cancel, color="red") 
    ax2.bar(meses, selecbymonth, width=0.5, align="edge", color="skyblue")

    ax2.set_title("Reservas por meses en el año {0}".format(info_y))
    ax2.set_xlabel("Meses")
    ax2.set_ylabel("Nº Clientes hicieron la reserva con {0} ".format(market))
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    tasas_mont=[100*selecbymonth_cancel[i]/selecbymonth[i] for i in range(len(selecbymonth))]
    ax3.bar(meses, tasas_mont, color="red") 
    

    ax3.set_title("Tasa de cancelación por meses en el año {0}".format(info_y))
    ax3.set_xlabel("Meses")
    ax3.set_ylabel("Tasa de clientes cancelaron la reserva de {0} ".format(market))
    st.pyplot(fig3)


