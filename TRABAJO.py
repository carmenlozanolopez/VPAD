import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import pandas as pd

st.set_page_config(page_title="TRABAJO VPAD",page_icon="❄️",layout="wide")

st.markdown('---')
st.header("ANÁLISIS HOTELERO")

#Carga de datos
df_hotel = pd.read_csv('hotel_bookings.csv')
