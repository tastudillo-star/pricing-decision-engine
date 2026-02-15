import streamlit as st
from datetime import date
from backend.sku_set import SKUSet
from utils.auth import Auth


@st.cache_data(show_spinner=True, ttl=600)
def run_page(fecha_inicio, fecha_fin, id_competidor):

    skuset = SKUSet.from_ventas(
        fecha_inicio=str(fecha_inicio),
        fecha_fin=str(fecha_fin),
    )
    skuset.get_posicionamiento(id_competidor)
    skuset.get_reglas(override_reglas=True)

    pos_total = skuset.get_master_pos(id_competidor)
    pos_total_v2 = skuset.get_master_pos_v2(id_competidor)
    print(pos_total)
    print(pos_total_v2)
    st.metric("Posicionamiento promedio (max competitor)", f"{pos_total:.4f}")


    #st.header(skuset.name)
    #st.dataframe(skuset.df_skus)
    #st.dataframe(skuset.df_ventas)
    #st.dataframe(skuset.df_posicionamiento)
    #st.dataframe(skuset.df_reglas)




#=============================================================================
# CONFIGURACIÓN DE PÁGINA Y CONTROLES
#=============================================================================
st.header("Master Posicionamiento")

auth = Auth()
auth.require_page()

# Parámetros
st.sidebar.subheader("Parámetros")
st.set_page_config(
    page_title="Pricing Chiper – BI",
    page_icon="https://chiper.cl/wp-content/uploads/2023/06/cropped-favicon-192x192.png",
    layout="wide",
)
fecha_inicio = st.sidebar.date_input(
    "Fecha inicio",
    value=date.today(),
)
fecha_fin = st.sidebar.date_input(
    "Fecha fin",
    value=date.today(),
)
COMPETIDORES = {
    1: "Central Mayorista",
    2: "Alvi",
    3: "La Oferta",
    None: "Mejor precio (min entre 1–3)",
}
id_competidor = st.sidebar.selectbox(
    "Competidor",
    options=list(COMPETIDORES.keys()),
    format_func=lambda x: f"{x} – {COMPETIDORES.get(x, 'Competidor')}",
    index=3,
)

# Button para cargar datos
if st.sidebar.button("Cargar datos"):
    st.cache_data.clear()
    run_page(fecha_inicio, fecha_fin, id_competidor)
