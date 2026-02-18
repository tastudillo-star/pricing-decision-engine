import streamlit as st
from datetime import date, timedelta
from backend.sku_set import SKUSet
from front.figures.scatter_pos_margen import ScatterPosMargen, ScatterConfig
import pandas as pd

from utils.auth import Auth

# Configuración de página - debe ser lo primero
st.set_page_config(
    page_title="Pricing Chiper – BI",
    page_icon="https://chiper.cl/wp-content/uploads/2023/06/cropped-favicon-192x192.png",
    layout="wide",
)

@st.cache_data(show_spinner=True, ttl=600)
def load_data(fecha_fin, num_ventanas, tamano_ventana_dias, id_competidor):
    """
    Carga datos para múltiples ventanas de tiempo y consolida en un DataFrame maestro.

    Args:
        fecha_fin: Fecha final de la ventana más reciente
        num_ventanas: Número de ventanas a generar
        tamano_ventana_dias: Tamaño de cada ventana en días
        id_competidor: ID del competidor

    Returns:
        DataFrame consolidado con columna 'ventana' (1=más antigua, num_ventanas=más reciente)
    """


    print(f'Corriendo load_data_multiple_windows() con {num_ventanas} ventanas de {tamano_ventana_dias} días')

    df_consolidado = pd.DataFrame()
    skuset_ventanas = []

    for i in range(num_ventanas):
        # Calcular ventana (1 es la más antigua, num_ventanas es la más reciente)
        ventana_numero = num_ventanas - i

        # Calcular fechas para esta ventana
        # La ventana más reciente (i=0) termina en fecha_fin
        fin_ventana = fecha_fin - timedelta(days=i * tamano_ventana_dias)
        inicio_ventana = fin_ventana - timedelta(days=tamano_ventana_dias - 1)

        print(f'  Ventana {ventana_numero}: {inicio_ventana} a {fin_ventana}')

        try:
            # Cargar datos para esta ventana
            skuset = SKUSet.build_master(
                fecha_inicio=str(inicio_ventana),
                fecha_fin=str(fin_ventana),
                id_competidor=id_competidor,
            )

            skuset_ventanas.append(skuset)

            # Obtener el DataFrame master
            df_ventana = skuset.master.copy()

            # Agregar columna de ventana
            df_ventana['ventana'] = ventana_numero

            # Agregar fechas de la ventana
            df_ventana['fecha_inicio_ventana'] = inicio_ventana
            df_ventana['fecha_fin_ventana'] = fin_ventana

            # Consolidar
            df_consolidado = pd.concat([df_consolidado, df_ventana], ignore_index=True)

        except Exception as e:
            print(f'  Error en ventana {ventana_numero}: {e}')
            continue

    print(f'DataFrame consolidado: {len(df_consolidado)} registros, {df_consolidado["ventana"].nunique()} ventanas')

    return df_consolidado, skuset_ventanas

def display_page(df_consolidado, ventanas_skuset):
    """Muestra los widgets de Streamlit usando los datos cargados"""
    print('Corriendo display_page()')
    skuset = ventanas_skuset[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Venta total:', f"${skuset.venta_neta_inicial:,.0f}")
    with col2:
        st.metric('% Venta representada:', f"{skuset.rep_venta_neta:.2%}")
    with col3:
        st.metric('Posicionamiento ponderado:', f"{skuset.posicionamiento:.2%}")

    col11, col12, col13 = st.columns(3)
    with col11:
        st.metric('Venta representada:', f"${skuset.venta_neta:,.0f}")
    with col12:
        st.metric('Representatividad SKUs:', f"{skuset.rep_skus:.2%}")
    with col13:
        st.metric('Margen ponderado:', f"{skuset.margen:.2%}")

    st.dataframe(skuset.master)

    cfg = ScatterConfig(
        # optional overrides
        x_ref=100.0,
        y_ref=10.00,
        height=750,
    )
    component = ScatterPosMargen(df_consolidado, config=cfg)
    fig = component.render()
    st.plotly_chart(fig, width='stretch')



#=============================================================================
# CONFIGURACIÓN DE PÁGINA Y CONTROLES
#=============================================================================
print('Iniciando página Master Posicionamiento')
st.header("Master Posicionamiento")

#auth = Auth()
#auth.require_page()

# Parámetros
st.sidebar.subheader("Parámetros")

# Modo múltiples ventanas
def get_last_sunday(ref_date: date | None = None) -> date:
    ref = ref_date or date.today()
    days_since_sunday = (ref.weekday() + 1) % 7
    return ref - timedelta(days=days_since_sunday)
last_sunday = get_last_sunday()

fecha_fin = st.sidebar.date_input(
    "Fecha fin (ventana más reciente)",
    value=last_sunday,
)

num_ventanas = st.sidebar.number_input(
    "Número de ventanas",
    min_value=1,
    max_value=52,
    value=3,
    step=1,
    help="Número de ventanas hacia atrás desde la fecha fin"
)

tamano_ventana = st.sidebar.number_input(
    "Tamaño de ventana (días)",
    min_value=1,
    max_value=365,
    value=7,
    step=1,
    help="Duración de cada ventana en días"
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

# Modo múltiples ventanas
df_consolidado, ventanas_skuset = load_data(
    fecha_fin=fecha_fin,
    num_ventanas=int(num_ventanas),
    tamano_ventana_dias=int(tamano_ventana),
    id_competidor=id_competidor
)

display_page(df_consolidado, ventanas_skuset)